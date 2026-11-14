// ============================================================================
// VERSORC-MATH: GPU HARDWARE COMPUTE & VULKAN DISPATCH PRIMITIVES
// License: Apache-2.0
// ============================================================================

pub mod shaders;

pub use shaders::{
    CgaTransformPushConstants, GaCl3BatchPushConstants, PgaTransformPushConstants,
    PoincareBatchPushConstants, RmsNormPushConstants, RopePushConstants, SiluMulPushConstants,
    VecAddPushConstants, CGA_TRANSFORM_SPV, GA_CL3_BATCH_SPV, PGA_TRANSFORM_SPV,
    POINCARE_MOBIUS_BATCH_SPV, RMSNORM_SPV, ROPE_SPV, SILU_MUL_SPV, VEC_ADD_SPV,
};

#[cfg(feature = "vulkan")]
pub mod vulkan {
    use crate::gpu::shaders::*;
    use crate::pga::PgaMotor;
    use ash::{vk, Device, Entry, Instance};
    use std::collections::HashMap;
    use std::ffi::CStr;
    use std::sync::Mutex;

    const WORKGROUP_SIZE_1D: u32 = 64;
    const MAX_STAGING_POOL_BUFFERS: usize = 16;
    const MAX_KERNEL_BINDINGS: u32 = 3;

    /// Discovered Vulkan Compute Device.
    #[derive(Debug, Clone)]
    pub struct DiscoveredGpu {
        pub name: String,
        pub is_discrete: bool,
        pub driver_version: u32,
        pub api_version: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum ComputeKernel {
        PgaTransform,
        VecAdd,
        Cl3Batch,
        PoincareBatch,
        CgaTransform,
        Rope,
        RmsNorm,
        SiluMul,
    }

    struct KernelSpec {
        spirv: &'static [u8],
        binding_count: u32,
        push_constant_size: u32,
    }

    impl ComputeKernel {
        fn spec(self) -> KernelSpec {
            match self {
                Self::PgaTransform => KernelSpec {
                    spirv: PGA_TRANSFORM_SPV,
                    binding_count: 2,
                    push_constant_size: std::mem::size_of::<PgaTransformPushConstants>() as u32,
                },
                Self::VecAdd => KernelSpec {
                    spirv: VEC_ADD_SPV,
                    binding_count: 2,
                    push_constant_size: std::mem::size_of::<VecAddPushConstants>() as u32,
                },
                Self::Cl3Batch => KernelSpec {
                    spirv: GA_CL3_BATCH_SPV,
                    binding_count: 3,
                    push_constant_size: std::mem::size_of::<GaCl3BatchPushConstants>() as u32,
                },
                Self::PoincareBatch => KernelSpec {
                    spirv: POINCARE_MOBIUS_BATCH_SPV,
                    binding_count: 3,
                    push_constant_size: std::mem::size_of::<PoincareBatchPushConstants>() as u32,
                },
                Self::CgaTransform => KernelSpec {
                    spirv: CGA_TRANSFORM_SPV,
                    binding_count: 2,
                    push_constant_size: std::mem::size_of::<CgaTransformPushConstants>() as u32,
                },
                Self::Rope => KernelSpec {
                    spirv: ROPE_SPV,
                    binding_count: 1,
                    push_constant_size: std::mem::size_of::<RopePushConstants>() as u32,
                },
                Self::RmsNorm => KernelSpec {
                    spirv: RMSNORM_SPV,
                    binding_count: 3,
                    push_constant_size: std::mem::size_of::<RmsNormPushConstants>() as u32,
                },
                Self::SiluMul => KernelSpec {
                    spirv: SILU_MUL_SPV,
                    binding_count: 3,
                    push_constant_size: std::mem::size_of::<SiluMulPushConstants>() as u32,
                },
            }
        }
    }

    /// RAII guard preventing `vk::Instance` leaks on early errors during probe/init.
    struct InstanceGuard(Option<Instance>);

    impl Drop for InstanceGuard {
        fn drop(&mut self) {
            if let Some(instance) = self.0.take() {
                unsafe {
                    instance.destroy_instance(None);
                }
            }
        }
    }

    /// RAII guard preventing `vk::Device` leaks on early errors during runtime creation.
    struct DeviceGuard(Option<Device>);

    impl Drop for DeviceGuard {
        fn drop(&mut self) {
            if let Some(device) = self.0.take() {
                unsafe {
                    device.destroy_device(None);
                }
            }
        }
    }

    /// RAII guard ensuring partially constructed buffer/memory pairs are cleaned up on error.
    struct PartialBufferGuard<'a> {
        device: &'a Device,
        buffer: Option<vk::Buffer>,
        memory: Option<vk::DeviceMemory>,
    }

    impl Drop for PartialBufferGuard<'_> {
        fn drop(&mut self) {
            unsafe {
                if let Some(buf) = self.buffer.take() {
                    self.device.destroy_buffer(buf, None);
                }
                if let Some(mem) = self.memory.take() {
                    self.device.free_memory(mem, None);
                }
            }
        }
    }

    /// RAII guard ensuring temporary pipeline construction objects never leak on failure.
    struct PipelineBuildGuard<'a> {
        device: &'a Device,
        desc_layout: Option<vk::DescriptorSetLayout>,
        pipeline_layout: Option<vk::PipelineLayout>,
        shader_module: Option<vk::ShaderModule>,
    }

    impl Drop for PipelineBuildGuard<'_> {
        fn drop(&mut self) {
            unsafe {
                if let Some(module) = self.shader_module.take() {
                    self.device.destroy_shader_module(module, None);
                }
                if let Some(layout) = self.pipeline_layout.take() {
                    self.device.destroy_pipeline_layout(layout, None);
                }
                if let Some(layout) = self.desc_layout.take() {
                    self.device.destroy_descriptor_set_layout(layout, None);
                }
            }
        }
    }

    /// Cached compiled Vulkan compute pipeline and associated layouts.
    #[derive(Clone, Copy)]
    struct VulkanComputePipeline {
        desc_layout: vk::DescriptorSetLayout,
        pipeline_layout: vk::PipelineLayout,
        pipeline: vk::Pipeline,
    }

    impl VulkanComputePipeline {
        fn build(
            ctx: &GpuComputeContext,
            pipeline_cache: vk::PipelineCache,
            spec: &KernelSpec,
        ) -> Result<Self, String> {
            let mut guard = PipelineBuildGuard {
                device: &ctx.device,
                desc_layout: None,
                pipeline_layout: None,
                shader_module: None,
            };

            let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..spec.binding_count)
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();

            let desc_layout = unsafe {
                ctx.device
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                        None,
                    )
                    .map_err(|e| format!("vkCreateDescriptorSetLayout failed: {}", e))?
            };
            guard.desc_layout = Some(desc_layout);

            let push_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(spec.push_constant_size);
            let set_layouts = [desc_layout];
            let push_ranges = [push_range];

            let pipeline_layout = unsafe {
                ctx.device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&set_layouts)
                            .push_constant_ranges(&push_ranges),
                        None,
                    )
                    .map_err(|e| format!("vkCreatePipelineLayout failed: {}", e))?
            };
            guard.pipeline_layout = Some(pipeline_layout);

            let shader_module = ctx.create_shader_module(spec.spirv)?;
            guard.shader_module = Some(shader_module);

            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(c"main");
            let pipe_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(pipeline_layout);

            let pipeline = unsafe {
                ctx.device
                    .create_compute_pipelines(pipeline_cache, &[pipe_info], None)
                    .map_err(|(_, e)| format!("vkCreateComputePipelines failed: {:?}", e))?[0]
            };

            // Shader module is no longer needed once the compute pipeline is compiled.
            if let Some(module) = guard.shader_module.take() {
                unsafe {
                    ctx.device.destroy_shader_module(module, None);
                }
            }
            let pipeline_layout = guard.pipeline_layout.take().unwrap();
            let desc_layout = guard.desc_layout.take().unwrap();

            Ok(Self {
                desc_layout,
                pipeline_layout,
                pipeline,
            })
        }

        fn destroy(self, device: &Device) {
            unsafe {
                device.destroy_pipeline(self.pipeline, None);
                device.destroy_pipeline_layout(self.pipeline_layout, None);
                device.destroy_descriptor_set_layout(self.desc_layout, None);
            }
        }
    }

    struct PooledBuffer {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        capacity: vk::DeviceSize,
    }

    /// Reusable host-visible coherent storage buffer pool.
    struct StagingBufferPool {
        free: Vec<PooledBuffer>,
    }

    impl StagingBufferPool {
        fn new() -> Self {
            Self {
                free: Vec::with_capacity(MAX_STAGING_POOL_BUFFERS),
            }
        }

        fn acquire(
            &mut self,
            ctx: &GpuComputeContext,
            required_size: vk::DeviceSize,
        ) -> Result<PooledBuffer, String> {
            // Best-fit reuse: pick the smallest pooled buffer that fits `required_size`.
            if let Some((idx, _)) = self
                .free
                .iter()
                .enumerate()
                .filter(|(_, b)| b.capacity >= required_size)
                .min_by_key(|(_, b)| b.capacity)
            {
                return Ok(self.free.swap_remove(idx));
            }

            // Evict the smallest idle buffer when capping pool residency.
            if self.free.len() >= MAX_STAGING_POOL_BUFFERS {
                if let Some((min_idx, _)) =
                    self.free.iter().enumerate().min_by_key(|(_, b)| b.capacity)
                {
                    let evicted = self.free.swap_remove(min_idx);
                    unsafe {
                        ctx.device.destroy_buffer(evicted.buffer, None);
                        ctx.device.free_memory(evicted.memory, None);
                    }
                }
            }

            let (buffer, memory) = ctx.create_buffer(
                required_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            Ok(PooledBuffer {
                buffer,
                memory,
                capacity: required_size,
            })
        }

        fn release_all(&mut self, buffers: &mut Vec<PooledBuffer>) {
            self.free.append(buffers);
        }

        fn destroy_all(&mut self, device: &Device) {
            unsafe {
                for buf in self.free.drain(..) {
                    device.destroy_buffer(buf.buffer, None);
                    device.free_memory(buf.memory, None);
                }
            }
        }
    }

    /// Scoped lease guard returning acquired staging buffers to `StagingBufferPool` on exit.
    struct StagingLeaseGuard<'a> {
        pool: &'a mut StagingBufferPool,
        leased: Vec<PooledBuffer>,
    }

    impl Drop for StagingLeaseGuard<'_> {
        fn drop(&mut self) {
            self.pool.release_all(&mut self.leased);
        }
    }

    struct GpuRuntimeState {
        pipeline_cache: vk::PipelineCache,
        pipelines: HashMap<ComputeKernel, VulkanComputePipeline>,
        staging_pool: StagingBufferPool,
        cmd_pool: vk::CommandPool,
        cmd_buf: vk::CommandBuffer,
        desc_pool: vk::DescriptorPool,
        fence: vk::Fence,
    }

    impl GpuRuntimeState {
        fn init(device: &Device, queue_family_index: u32) -> Result<Self, String> {
            let pipeline_cache = unsafe {
                device
                    .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                    .map_err(|e| format!("vkCreatePipelineCache failed: {}", e))?
            };

            let cmd_pool = unsafe {
                device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(queue_family_index),
                        None,
                    )
                    .map_err(|e| {
                        device.destroy_pipeline_cache(pipeline_cache, None);
                        format!("vkCreateCommandPool failed: {}", e)
                    })?
            };

            let cmd_buf = match unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            } {
                Ok(bufs) => bufs[0],
                Err(e) => {
                    unsafe {
                        device.destroy_command_pool(cmd_pool, None);
                        device.destroy_pipeline_cache(pipeline_cache, None);
                    }
                    return Err(format!("vkAllocateCommandBuffers failed: {}", e));
                }
            };

            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(MAX_KERNEL_BINDINGS);
            let pool_sizes = [pool_size];
            let desc_pool = match unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
            } {
                Ok(pool) => pool,
                Err(e) => {
                    unsafe {
                        device.destroy_command_pool(cmd_pool, None);
                        device.destroy_pipeline_cache(pipeline_cache, None);
                    }
                    return Err(format!("vkCreateDescriptorPool failed: {}", e));
                }
            };

            let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            {
                Ok(f) => f,
                Err(e) => {
                    unsafe {
                        device.destroy_descriptor_pool(desc_pool, None);
                        device.destroy_command_pool(cmd_pool, None);
                        device.destroy_pipeline_cache(pipeline_cache, None);
                    }
                    return Err(format!("vkCreateFence failed: {}", e));
                }
            };

            Ok(Self {
                pipeline_cache,
                pipelines: HashMap::with_capacity(8),
                staging_pool: StagingBufferPool::new(),
                cmd_pool,
                cmd_buf,
                desc_pool,
                fence,
            })
        }

        fn destroy(&mut self, device: &Device) {
            unsafe {
                self.staging_pool.destroy_all(device);
                for (_, pipe) in self.pipelines.drain() {
                    pipe.destroy(device);
                }
                if self.pipeline_cache != vk::PipelineCache::null() {
                    device.destroy_pipeline_cache(self.pipeline_cache, None);
                    self.pipeline_cache = vk::PipelineCache::null();
                }
                if self.fence != vk::Fence::null() {
                    device.destroy_fence(self.fence, None);
                    self.fence = vk::Fence::null();
                }
                if self.desc_pool != vk::DescriptorPool::null() {
                    device.destroy_descriptor_pool(self.desc_pool, None);
                    self.desc_pool = vk::DescriptorPool::null();
                }
                if self.cmd_pool != vk::CommandPool::null() {
                    device.destroy_command_pool(self.cmd_pool, None);
                    self.cmd_pool = vk::CommandPool::null();
                }
            }
        }
    }

    enum BufferSlot<'a> {
        /// Read-only device input buffer. Any tail beyond `data.len()` up to `size` is zeroed.
        ReadOnly {
            data: &'a [u8],
            size: vk::DeviceSize,
        },
        /// In-place read-write buffer copied before execution and overwritten on completion.
        InOut(&'a mut [u8]),
        /// Output buffer filled with GPU results on completion.
        WriteOnly(&'a mut [u8]),
    }

    impl BufferSlot<'_> {
        fn byte_len(&self) -> Result<vk::DeviceSize, String> {
            let bytes = match self {
                Self::ReadOnly { size, .. } => return Ok(*size),
                Self::InOut(slice) => slice.len(),
                Self::WriteOnly(slice) => slice.len(),
            };
            vk::DeviceSize::try_from(bytes)
                .map_err(|_| format!("Buffer size {} exceeds vk::DeviceSize", bytes))
        }
    }

    fn checked_buffer_size<T>(len: usize) -> Result<vk::DeviceSize, String> {
        let elem_size = std::mem::size_of::<T>();
        let total = len.checked_mul(elem_size).ok_or_else(|| {
            format!(
                "Vulkan buffer byte size overflow for len={} and elem_size={}",
                len, elem_size
            )
        })?;
        vk::DeviceSize::try_from(total)
            .map_err(|_| format!("Buffer size {} exceeds vk::DeviceSize", total))
    }

    fn checked_u32(val: usize, label: &str) -> Result<u32, String> {
        u32::try_from(val)
            .map_err(|_| format!("{} ({}) exceeds u32 GPU dispatch limit", label, val))
    }

    /// Minimal zero-dependency Vulkan Compute Context with cached pipelines and staging pool.
    pub struct GpuComputeContext {
        pub entry: Entry,
        pub instance: Instance,
        pub physical_device: vk::PhysicalDevice,
        pub device: Device,
        pub compute_queue: vk::Queue,
        pub queue_family_index: u32,
        runtime: Mutex<GpuRuntimeState>,
    }

    impl GpuComputeContext {
        fn create_vulkan_instance() -> Result<(Entry, InstanceGuard), String> {
            let entry =
                unsafe { Entry::load().map_err(|e| format!("Vulkan loader error: {}", e))? };
            let app_info =
                vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
            let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
            let instance = unsafe {
                entry
                    .create_instance(&create_info, None)
                    .map_err(|e| format!("vkCreateInstance failed: {}", e))?
            };
            Ok((entry, InstanceGuard(Some(instance))))
        }

        /// Probes host for available Vulkan devices.
        pub fn probe_devices() -> Result<Vec<DiscoveredGpu>, String> {
            let (_entry, instance_guard) = Self::create_vulkan_instance()?;
            let instance = instance_guard.0.as_ref().unwrap();

            let pdevices = unsafe {
                instance
                    .enumerate_physical_devices()
                    .map_err(|e| format!("Failed to enumerate physical devices: {}", e))?
            };

            let mut list = Vec::with_capacity(pdevices.len());
            for pdev in pdevices {
                let props = unsafe { instance.get_physical_device_properties(pdev) };
                let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
                list.push(DiscoveredGpu {
                    name,
                    is_discrete,
                    driver_version: props.driver_version,
                    api_version: props.api_version,
                });
            }

            Ok(list)
        }

        /// Initializes compute context targeting preferred GPU (preferring discrete if available).
        pub fn init_compute() -> Result<Self, String> {
            let (entry, mut instance_guard) = Self::create_vulkan_instance()?;
            let instance = instance_guard.0.as_ref().unwrap();

            let pdevices = unsafe {
                instance
                    .enumerate_physical_devices()
                    .map_err(|e| format!("Failed to enumerate physical devices: {}", e))?
            };

            if pdevices.is_empty() {
                return Err("No Vulkan physical devices found".into());
            }

            let selected_pdev = pdevices
                .iter()
                .copied()
                .find(|&pdev| {
                    let props = unsafe { instance.get_physical_device_properties(pdev) };
                    props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .unwrap_or(pdevices[0]);

            let qprops =
                unsafe { instance.get_physical_device_queue_family_properties(selected_pdev) };
            let queue_family_index = qprops
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| "No compute queue family found".to_string())?
                as u32;

            let priorities = [1.0f32];
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);
            let dev_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info));

            let device = unsafe {
                instance
                    .create_device(selected_pdev, &dev_create_info, None)
                    .map_err(|e| format!("vkCreateDevice failed: {}", e))?
            };
            let mut device_guard = DeviceGuard(Some(device));
            let runtime =
                GpuRuntimeState::init(device_guard.0.as_ref().unwrap(), queue_family_index)?;
            let compute_queue = unsafe {
                device_guard
                    .0
                    .as_ref()
                    .unwrap()
                    .get_device_queue(queue_family_index, 0)
            };

            let device = device_guard.0.take().unwrap();
            let instance = instance_guard.0.take().unwrap();

            Ok(Self {
                entry,
                instance,
                physical_device: selected_pdev,
                device,
                compute_queue,
                queue_family_index,
                runtime: Mutex::new(runtime),
            })
        }

        /// Returns the number of compute pipelines currently cached on this context.
        pub fn cached_pipeline_count(&self) -> usize {
            self.runtime
                .lock()
                .map(|rt| rt.pipelines.len())
                .unwrap_or(0)
        }

        /// Creates a shader module from SPIR-V byte slice.
        pub fn create_shader_module(&self, spirv_bytes: &[u8]) -> Result<vk::ShaderModule, String> {
            if spirv_bytes.is_empty() || !spirv_bytes.len().is_multiple_of(4) {
                return Err(format!(
                    "SPIR-V byte length ({}) must be non-zero and a multiple of 4",
                    spirv_bytes.len()
                ));
            }
            let mut words = Vec::<u32>::with_capacity(spirv_bytes.len() / 4);
            for chunk in spirv_bytes.chunks_exact(4) {
                words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let info = vk::ShaderModuleCreateInfo::default().code(&words);
            unsafe {
                self.device
                    .create_shader_module(&info, None)
                    .map_err(|e| format!("vkCreateShaderModule failed: {}", e))
            }
        }

        /// Finds a host or device memory type satisfying filter flags and properties.
        pub fn find_memory_type(
            &self,
            type_filter: u32,
            properties: vk::MemoryPropertyFlags,
        ) -> Result<u32, String> {
            let mem_props = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical_device)
            };
            for (i, mem_type) in mem_props.memory_types[..mem_props.memory_type_count as usize]
                .iter()
                .enumerate()
            {
                if (type_filter & (1 << i)) != 0
                    && (mem_type.property_flags & properties) == properties
                {
                    return Ok(i as u32);
                }
            }
            Err("Failed to find suitable memory type".into())
        }

        /// Allocates a GPU buffer with dedicated memory, cleaning up safely on partial failure.
        pub fn create_buffer(
            &self,
            size: vk::DeviceSize,
            usage: vk::BufferUsageFlags,
            properties: vk::MemoryPropertyFlags,
        ) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
            let mut guard = PartialBufferGuard {
                device: &self.device,
                buffer: None,
                memory: None,
            };

            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = unsafe {
                self.device
                    .create_buffer(&buffer_info, None)
                    .map_err(|e| format!("vkCreateBuffer failed: {}", e))?
            };
            guard.buffer = Some(buffer);

            let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
            let mem_type_index = self.find_memory_type(mem_reqs.memory_type_bits, properties)?;

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type_index);

            let memory = unsafe {
                self.device
                    .allocate_memory(&alloc_info, None)
                    .map_err(|e| format!("vkAllocateMemory failed: {}", e))?
            };
            guard.memory = Some(memory);

            unsafe {
                self.device
                    .bind_buffer_memory(buffer, memory, 0)
                    .map_err(|e| format!("vkBindBufferMemory failed: {}", e))?;
            }

            Ok((guard.buffer.take().unwrap(), guard.memory.take().unwrap()))
        }

        fn upload_slot(
            &self,
            memory: vk::DeviceMemory,
            size: vk::DeviceSize,
            slot: &BufferSlot<'_>,
        ) -> Result<(), String> {
            let (src, pad_to_size) = match slot {
                BufferSlot::ReadOnly { data, .. } => (*data, true),
                BufferSlot::InOut(slice) => (&**slice, false),
                BufferSlot::WriteOnly(_) => return Ok(()),
            };

            unsafe {
                let ptr = self
                    .device
                    .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("vkMapMemory upload failed: {}", e))?
                    as *mut u8;

                let copy_len = src.len().min(size as usize);
                if copy_len > 0 {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, copy_len);
                }
                if pad_to_size && copy_len < size as usize {
                    std::ptr::write_bytes(ptr.add(copy_len), 0, size as usize - copy_len);
                }

                self.device.unmap_memory(memory);
            }
            Ok(())
        }

        fn readback_slot(
            &self,
            memory: vk::DeviceMemory,
            size: vk::DeviceSize,
            slot: &mut BufferSlot<'_>,
        ) -> Result<(), String> {
            let dst = match slot {
                BufferSlot::InOut(slice) | BufferSlot::WriteOnly(slice) => slice,
                BufferSlot::ReadOnly { .. } => return Ok(()),
            };

            unsafe {
                let ptr = self
                    .device
                    .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("vkMapMemory readback failed: {}", e))?
                    as *const u8;
                std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len());
                self.device.unmap_memory(memory);
            }
            Ok(())
        }

        fn execute_kernel(
            &self,
            kernel: ComputeKernel,
            slots: &mut [BufferSlot<'_>],
            push_constants: &[u8],
            workgroups: [u32; 3],
        ) -> Result<(), String> {
            let spec = kernel.spec();
            if slots.len() != spec.binding_count as usize {
                return Err(format!(
                    "Kernel {:?} expects {} buffer bindings, got {}",
                    kernel,
                    spec.binding_count,
                    slots.len()
                ));
            }
            if push_constants.len() != spec.push_constant_size as usize {
                return Err(format!(
                    "Kernel {:?} expects {} push-constant bytes, got {}",
                    kernel,
                    spec.push_constant_size,
                    push_constants.len()
                ));
            }

            let mut slot_sizes = Vec::with_capacity(slots.len());
            for slot in slots.iter() {
                let size = slot.byte_len()?;
                if size == 0 {
                    return Err("Zero-byte Vulkan buffer dispatch is invalid".into());
                }
                slot_sizes.push(size);
            }

            let mut rt_guard = self
                .runtime
                .lock()
                .map_err(|_| "GPU runtime mutex poisoned".to_string())?;
            let rt = &mut *rt_guard;

            let pipe = if let Some(&existing) = rt.pipelines.get(&kernel) {
                existing
            } else {
                let created = VulkanComputePipeline::build(self, rt.pipeline_cache, &spec)?;
                rt.pipelines.insert(kernel, created);
                created
            };

            let mut lease = StagingLeaseGuard {
                pool: &mut rt.staging_pool,
                leased: Vec::with_capacity(slots.len()),
            };
            for &size in &slot_sizes {
                lease.leased.push(lease.pool.acquire(self, size)?);
            }

            for ((buf, &size), slot) in lease.leased.iter().zip(&slot_sizes).zip(slots.iter()) {
                self.upload_slot(buf.memory, size, slot)?;
            }

            unsafe {
                self.device
                    .reset_descriptor_pool(rt.desc_pool, vk::DescriptorPoolResetFlags::empty())
                    .map_err(|e| format!("vkResetDescriptorPool failed: {}", e))?;
            }

            let set_layouts = [pipe.desc_layout];
            let desc_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(rt.desc_pool)
                .set_layouts(&set_layouts);
            let desc_set = unsafe {
                self.device
                    .allocate_descriptor_sets(&desc_alloc_info)
                    .map_err(|e| format!("vkAllocateDescriptorSets failed: {}", e))?[0]
            };

            let buf_infos: Vec<vk::DescriptorBufferInfo> = lease
                .leased
                .iter()
                .zip(&slot_sizes)
                .map(|(buf, &size)| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.buffer)
                        .offset(0)
                        .range(size)
                })
                .collect();

            let writes: Vec<vk::WriteDescriptorSet<'_>> = buf_infos
                .iter()
                .enumerate()
                .map(|(binding, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(desc_set)
                        .dst_binding(binding as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            unsafe {
                self.device.update_descriptor_sets(&writes, &[]);

                self.device
                    .reset_command_pool(rt.cmd_pool, vk::CommandPoolResetFlags::empty())
                    .map_err(|e| format!("vkResetCommandPool failed: {}", e))?;

                self.device
                    .begin_command_buffer(
                        rt.cmd_buf,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| format!("vkBeginCommandBuffer failed: {}", e))?;

                self.device.cmd_bind_pipeline(
                    rt.cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    pipe.pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    rt.cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    pipe.pipeline_layout,
                    0,
                    &[desc_set],
                    &[],
                );
                self.device.cmd_push_constants(
                    rt.cmd_buf,
                    pipe.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
                self.device
                    .cmd_dispatch(rt.cmd_buf, workgroups[0], workgroups[1], workgroups[2]);
                self.device
                    .end_command_buffer(rt.cmd_buf)
                    .map_err(|e| format!("vkEndCommandBuffer failed: {}", e))?;

                self.device
                    .reset_fences(&[rt.fence])
                    .map_err(|e| format!("vkResetFences failed: {}", e))?;

                let cmd_bufs = [rt.cmd_buf];
                let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
                self.device
                    .queue_submit(self.compute_queue, &[submit_info], rt.fence)
                    .map_err(|e| format!("vkQueueSubmit failed: {}", e))?;

                if let Err(e) = self.device.wait_for_fences(&[rt.fence], true, u64::MAX) {
                    let _ = self.device.device_wait_idle();
                    return Err(format!("vkWaitForFences failed: {}", e));
                }
            }

            for ((buf, &size), slot) in lease.leased.iter().zip(&slot_sizes).zip(slots.iter_mut()) {
                self.readback_slot(buf.memory, size, slot)?;
            }

            Ok(())
        }

        /// Dispatches 3D PGA motor rigid transformations over an `[N, 3]` tensor batch in GPU VRAM.
        pub fn dispatch_pga_transform(
            &self,
            points: &[[f32; 3]],
            motor: &PgaMotor,
        ) -> Result<Vec<[f32; 3]>, String> {
            let n = points.len();
            if n == 0 {
                return Ok(Vec::new());
            }

            let n_u32 = checked_u32(n, "points.len()")?;
            let _ = checked_buffer_size::<[f32; 4]>(n)?;
            let in_data: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect();
            let mut out_data = vec![[0.0f32; 4]; n];

            let unit_motor = motor.normalized();
            let pc = PgaTransformPushConstants::new(n_u32, &unit_motor);
            let mut slots = [
                BufferSlot::ReadOnly {
                    data: bytemuck::cast_slice(&in_data),
                    size: checked_buffer_size::<[f32; 4]>(n)?,
                },
                BufferSlot::WriteOnly(bytemuck::cast_slice_mut(&mut out_data)),
            ];

            self.execute_kernel(
                ComputeKernel::PgaTransform,
                &mut slots,
                bytemuck::bytes_of(&pc),
                [n_u32.div_ceil(WORKGROUP_SIZE_1D), 1, 1],
            )?;

            Ok(out_data.into_iter().map(|p| [p[0], p[1], p[2]]).collect())
        }

        /// Dispatches dense residual vector addition (`a + b`) on GPU.
        pub fn dispatch_vec_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
            if a.len() != b.len() {
                return Err(format!(
                    "Vectors a and b must have identical length (got {} vs {})",
                    a.len(),
                    b.len()
                ));
            }
            let n = a.len();
            if n == 0 {
                return Ok(Vec::new());
            }

            let n_u32 = checked_u32(n, "a.len()")?;
            let size = checked_buffer_size::<f32>(n)?;
            let mut out = b.to_vec();
            let pc = VecAddPushConstants::new(n_u32);
            let mut slots = [
                BufferSlot::ReadOnly {
                    data: bytemuck::cast_slice(a),
                    size,
                },
                BufferSlot::InOut(bytemuck::cast_slice_mut(&mut out)),
            ];

            self.execute_kernel(
                ComputeKernel::VecAdd,
                &mut slots,
                bytemuck::bytes_of(&pc),
                [n_u32.div_ceil(WORKGROUP_SIZE_1D), 1, 1],
            )?;

            Ok(out)
        }

        /// Dispatches a batch `Cl(3, 0)` multivector operation (`op_code` `0..=12`) in GPU VRAM.
        /// Unary ops (`8..=12`: Hodge Dual, Reversion, Involution, Conjugation, Undual) ignore `b`.
        /// Binary ops (`0..=7`: GP, Wedge, LeftContract, Hestenes, Meet, Commutator, Anticommutator, Sandwich) require `b.len() == a.len()`.
        pub fn dispatch_cl3_batch(
            &self,
            a: &[[f32; 8]],
            b: &[[f32; 8]],
            op_code: u32,
        ) -> Result<Vec<[f32; 8]>, String> {
            let n = a.len();
            if n == 0 {
                return Ok(Vec::new());
            }

            let is_unary = (8..=12).contains(&op_code);
            if !is_unary && b.len() != n {
                return Err(format!(
                    "Binary Cl(3, 0) operator op_code {} requires b.len() == a.len() (got {} vs {})",
                    op_code,
                    b.len(),
                    n
                ));
            }

            let n_u32 = checked_u32(n, "a.len()")?;
            let size = checked_buffer_size::<[f32; 8]>(n)?;
            let b_slice = if b.is_empty() {
                &[]
            } else {
                &b[..b.len().min(n)]
            };
            let mut results = vec![[0.0f32; 8]; n];
            let pc = GaCl3BatchPushConstants::new(n_u32, op_code);

            let mut slots = [
                BufferSlot::ReadOnly {
                    data: bytemuck::cast_slice(a),
                    size,
                },
                BufferSlot::ReadOnly {
                    data: bytemuck::cast_slice(b_slice),
                    size,
                },
                BufferSlot::WriteOnly(bytemuck::cast_slice_mut(&mut results)),
            ];

            self.execute_kernel(
                ComputeKernel::Cl3Batch,
                &mut slots,
                bytemuck::bytes_of(&pc),
                [n_u32.div_ceil(WORKGROUP_SIZE_1D), 1, 1],
            )?;

            Ok(results)
        }

        /// Dispatches Poincare/Hyperbolic batch operations (`0=Mobius Add`, `1=Geodesic Distance`,
        /// `2=Spherical Inversion`, `3=Poincare to Klein`, `4=Klein to Poincare`, `5=Einstein Midpoint`).
        /// Always returns `Vec<[f32; 3]>`; for distance (`op_code == 1`), result is in `[0]`.
        pub fn dispatch_poincare_batch(
            &self,
            u: &[[f32; 3]],
            v: &[[f32; 3]],
            op_code: u32,
            sphere_center: [f32; 3],
            sphere_radius: f32,
        ) -> Result<Vec<[f32; 3]>, String> {
            let n = u.len();
            if n == 0 {
                return Ok(Vec::new());
            }

            let is_binary = op_code == 0 || op_code == 1 || op_code == 5;
            if is_binary && v.len() != n {
                return Err(format!(
                    "Binary Poincare operator op_code {} requires v.len() == u.len() (got {} vs {})",
                    op_code,
                    v.len(),
                    n
                ));
            }

            let n_u32 = checked_u32(n, "u.len()")?;
            let size = checked_buffer_size::<[f32; 3]>(n)?;
            let v_slice = if v.is_empty() {
                &[]
            } else {
                &v[..v.len().min(n)]
            };
            let mut results = vec![[0.0f32; 3]; n];
            let pc = PoincareBatchPushConstants::new(n_u32, op_code, sphere_center, sphere_radius);

            let mut slots = [
                BufferSlot::ReadOnly {
                    data: bytemuck::cast_slice(u),
                    size,
                },
                BufferSlot::ReadOnly {
                    data: bytemuck::cast_slice(v_slice),
                    size,
                },
                BufferSlot::WriteOnly(bytemuck::cast_slice_mut(&mut results)),
            ];

            self.execute_kernel(
                ComputeKernel::PoincareBatch,
                &mut slots,
                bytemuck::bytes_of(&pc),
                [n_u32.div_ceil(WORKGROUP_SIZE_1D), 1, 1],
            )?;

            Ok(results)
        }

        /// Dispatches Poincare geodesic distance batch on GPU, returning a flat `Vec<f32>`.
        pub fn dispatch_poincare_distance(
            &self,
            u: &[[f32; 3]],
            v: &[[f32; 3]],
        ) -> Result<Vec<f32>, String> {
            let pts = self.dispatch_poincare_batch(u, v, 1, [0.0; 3], 1.0)?;
            Ok(pts.into_iter().map(|p| p[0]).collect())
        }

        /// Dispatches 3D Conformal Geometric Algebra (CGA) transformations
        /// (`0=Spherical Inversion`, `1=Dilation`, `2=Translation`) in GPU VRAM.
        pub fn dispatch_cga_transform(
            &self,
            points: &[[f32; 3]],
            pc: &CgaTransformPushConstants,
        ) -> Result<Vec<[f32; 3]>, String> {
            let n = points.len();
            if n == 0 {
                return Ok(Vec::new());
            }

            let n_u32 = checked_u32(n, "points.len()")?;
            let size = checked_buffer_size::<[f32; 4]>(n)?;
            let in_data: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect();
            let mut out_data = vec![[0.0f32; 4]; n];

            let mut bound_pc = *pc;
            bound_pc.num_points = n_u32;

            let mut slots = [
                BufferSlot::ReadOnly {
                    data: bytemuck::cast_slice(&in_data),
                    size,
                },
                BufferSlot::WriteOnly(bytemuck::cast_slice_mut(&mut out_data)),
            ];

            self.execute_kernel(
                ComputeKernel::CgaTransform,
                &mut slots,
                bytemuck::bytes_of(&bound_pc),
                [n_u32.div_ceil(WORKGROUP_SIZE_1D), 1, 1],
            )?;

            Ok(out_data.into_iter().map(|p| [p[0], p[1], p[2]]).collect())
        }

        /// Dispatches Rotary Position Embedding (RoPE) / Planar Spinor Bank in-place in GPU VRAM.
        /// Supports both NeoX layout (split-half) and LLaMA layout (interleaved).
        pub fn dispatch_rope(
            &self,
            v: &mut [f32],
            pos: u32,
            n_heads: u32,
            head_dim: u32,
            rope_theta: f32,
            is_neox: bool,
        ) -> Result<(), String> {
            let total_elements = (n_heads as usize)
                .checked_mul(head_dim as usize)
                .ok_or_else(|| "RoPE element count overflow".to_string())?;
            if v.len() != total_elements {
                return Err(format!(
                    "RoPE vector length mismatch: expected {} (n_heads * head_dim), got {}",
                    total_elements,
                    v.len()
                ));
            }
            if !head_dim.is_multiple_of(2) {
                return Err(format!(
                    "head_dim ({}) must be even for 2D spinor rotations",
                    head_dim
                ));
            }
            if total_elements == 0 {
                return Ok(());
            }

            let _ = checked_buffer_size::<f32>(total_elements)?;
            let total_pairs = n_heads
                .checked_mul(head_dim / 2)
                .ok_or_else(|| "RoPE pair count overflow".to_string())?;
            let pc = RopePushConstants::new(pos, n_heads, head_dim, rope_theta, is_neox);
            let mut slots = [BufferSlot::InOut(bytemuck::cast_slice_mut(v))];

            self.execute_kernel(
                ComputeKernel::Rope,
                &mut slots,
                bytemuck::bytes_of(&pc),
                [total_pairs.div_ceil(WORKGROUP_SIZE_1D), 1, 1],
            )
        }

        /// Dispatches Root Mean Square Normalization (RMSNorm) with subgroup parallel reduction on GPU.
        pub fn dispatch_rmsnorm(
            &self,
            x: &[f32],
            gamma: &[f32],
            eps: f32,
        ) -> Result<Vec<f32>, String> {
            if x.len() != gamma.len() {
                return Err(format!(
                    "Vector x and gamma must have identical dimension (got {} vs {})",
                    x.len(),
                    gamma.len()
                ));
            }
            let n = x.len();
            if n == 0 {
                return Ok(Vec::new());
            }

            let d_model = checked_u32(n, "x.len()")?;
            let size = checked_buffer_size::<f32>(n)?;
            let mut out = vec![0.0f32; n];
            let pc = RmsNormPushConstants::new(d_model, eps, 0);

            let mut slots = [
                BufferSlot::ReadOnly {
                    data: bytemuck::cast_slice(x),
                    size,
                },
                BufferSlot::ReadOnly {
                    data: bytemuck::cast_slice(gamma),
                    size,
                },
                BufferSlot::WriteOnly(bytemuck::cast_slice_mut(&mut out)),
            ];

            // Single-workgroup subgroup reduction inside rmsnorm.comp (local_size_x = 128).
            self.execute_kernel(
                ComputeKernel::RmsNorm,
                &mut slots,
                bytemuck::bytes_of(&pc),
                [1, 1, 1],
            )?;

            Ok(out)
        }

        /// Dispatches SiLU-Mul / SwiGLU activation fusion (`act = silu(gate) * up`) on GPU.
        pub fn dispatch_silu_mul(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
            if gate.len() != up.len() {
                return Err(format!(
                    "Gate and Up vector lengths must match (got {} vs {})",
                    gate.len(),
                    up.len()
                ));
            }
            let n = gate.len();
            if n == 0 {
                return Ok(Vec::new());
            }

            let n_u32 = checked_u32(n, "gate.len()")?;
            let size = checked_buffer_size::<f32>(n)?;
            let mut out = vec![0.0f32; n];
            let pc = SiluMulPushConstants::new(n_u32);

            let mut slots = [
                BufferSlot::ReadOnly {
                    data: bytemuck::cast_slice(gate),
                    size,
                },
                BufferSlot::ReadOnly {
                    data: bytemuck::cast_slice(up),
                    size,
                },
                BufferSlot::WriteOnly(bytemuck::cast_slice_mut(&mut out)),
            ];

            self.execute_kernel(
                ComputeKernel::SiluMul,
                &mut slots,
                bytemuck::bytes_of(&pc),
                [n_u32.div_ceil(WORKGROUP_SIZE_1D), 1, 1],
            )?;

            Ok(out)
        }
    }

    impl Drop for GpuComputeContext {
        fn drop(&mut self) {
            unsafe {
                let _ = self.device.device_wait_idle();
                let rt = self.runtime.get_mut().unwrap_or_else(|e| e.into_inner());
                rt.destroy(&self.device);
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }
}
