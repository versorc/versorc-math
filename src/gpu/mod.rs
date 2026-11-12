// ============================================================================
// VERSORC-MATH: GPU HARDWARE COMPUTE & VULKAN DISPATCH PRIMITIVES
// License: Apache-2.0 OR MIT
// ============================================================================

pub mod shaders;

pub use shaders::{
    CgaTransformPushConstants, HotStratumE8PushConstants, PgaTransformPushConstants,
    RmsNormPushConstants, RopePushConstants, SiluMulPushConstants, VecAddPushConstants,
    CGA_TRANSFORM_SPV, HOT_STRATUM_E8_SPV, PGA_TRANSFORM_SPV, RMSNORM_SPV, ROPE_SPV,
    SILU_MUL_SPV, VEC_ADD_SPV,
};

#[cfg(feature = "vulkan")]
pub mod vulkan {
    use ash::{vk, Device, Entry, Instance};
    use std::ffi::CStr;
    use crate::pga::PgaMotor;
    use crate::gpu::shaders::*;

    /// Discovered Vulkan Compute Device.
    #[derive(Debug, Clone)]
    pub struct DiscoveredGpu {
        pub name: String,
        pub is_discrete: bool,
        pub driver_version: u32,
        pub api_version: u32,
    }

    /// Minimal zero-dependency Vulkan Compute Context for math dispatch.
    pub struct GpuComputeContext {
        pub entry: Entry,
        pub instance: Instance,
        pub physical_device: vk::PhysicalDevice,
        pub device: Device,
        pub compute_queue: vk::Queue,
        pub queue_family_index: u32,
    }

    impl GpuComputeContext {
        /// Probes host for available Vulkan devices.
        pub fn probe_devices() -> Result<Vec<DiscoveredGpu>, String> {
            let entry = unsafe {
                Entry::load().map_err(|e| format!("Vulkan loader error: {}", e))?
            };
            let app_info = vk::ApplicationInfo::default()
                .api_version(vk::make_api_version(0, 1, 3, 0));
            let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
            let instance = unsafe {
                entry.create_instance(&create_info, None)
                    .map_err(|e| format!("vkCreateInstance failed: {}", e))?
            };

            let pdevices = unsafe {
                instance.enumerate_physical_devices()
                    .map_err(|e| format!("Failed to enumerate physical devices: {}", e))?
            };

            let mut list = Vec::new();
            for &pdev in &pdevices {
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

            unsafe {
                instance.destroy_instance(None);
            }

            Ok(list)
        }

        /// Initializes compute context targeting preferred GPU (preferring discrete if available).
        pub fn init_compute() -> Result<Self, String> {
            let entry = unsafe {
                Entry::load().map_err(|e| format!("Vulkan loader error: {}", e))?
            };
            let app_info = vk::ApplicationInfo::default()
                .api_version(vk::make_api_version(0, 1, 3, 0));
            let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
            let instance = unsafe {
                entry.create_instance(&create_info, None)
                    .map_err(|e| format!("vkCreateInstance failed: {}", e))?
            };

            let pdevices = unsafe {
                instance.enumerate_physical_devices()
                    .map_err(|e| format!("Failed to enumerate physical devices: {}", e))?
            };

            if pdevices.is_empty() {
                return Err("No Vulkan physical devices found".into());
            }

            // Pick discrete GPU if present, else first
            let mut selected_pdev = pdevices[0];
            for &pdev in &pdevices {
                let props = unsafe { instance.get_physical_device_properties(pdev) };
                if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    selected_pdev = pdev;
                    break;
                }
            }

            // Find compute queue family
            let qprops = unsafe { instance.get_physical_device_queue_family_properties(selected_pdev) };
            let queue_family_index = qprops
                .iter()
                .enumerate()
                .position(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| "No compute queue family found".to_string())? as u32;

            let priorities = [1.0f32];
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);

            let dev_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info));

            let device = unsafe {
                instance.create_device(selected_pdev, &dev_create_info, None)
                    .map_err(|e| format!("vkCreateDevice failed: {}", e))?
            };

            let compute_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

            Ok(Self {
                entry,
                instance,
                physical_device: selected_pdev,
                device,
                compute_queue,
                queue_family_index,
            })
        }

        /// Creates a shader module from SPIR-V byte slice.
        pub fn create_shader_module(&self, spirv_bytes: &[u8]) -> Result<vk::ShaderModule, String> {
            assert_eq!(
                spirv_bytes.len() % 4,
                0,
                "SPIR-V byte length must be a multiple of 4"
            );
            let mut words = Vec::<u32>::with_capacity(spirv_bytes.len() / 4);
            for chunk in spirv_bytes.chunks_exact(4) {
                words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let info = vk::ShaderModuleCreateInfo::default().code(&words);
            unsafe {
                self.device.create_shader_module(&info, None)
                    .map_err(|e| format!("vkCreateShaderModule failed: {}", e))
            }
        }

        /// Finds a host or device memory type satisfying filter flags and properties.
        pub fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Result<u32, String> {
            let mem_props = unsafe { self.instance.get_physical_device_memory_properties(self.physical_device) };
            for i in 0..mem_props.memory_type_count {
                if (type_filter & (1 << i)) != 0 && (mem_props.memory_types[i as usize].property_flags & properties) == properties {
                    return Ok(i);
                }
            }
            Err("Failed to find suitable memory type".into())
        }

        /// Allocates a GPU buffer with dedicated memory.
        pub fn create_buffer(
            &self,
            size: vk::DeviceSize,
            usage: vk::BufferUsageFlags,
            properties: vk::MemoryPropertyFlags,
        ) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = unsafe {
                self.device.create_buffer(&buffer_info, None)
                    .map_err(|e| format!("vkCreateBuffer failed: {}", e))?
            };

            let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
            let mem_type_index = self.find_memory_type(mem_reqs.memory_type_bits, properties)?;

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type_index);

            let memory = unsafe {
                self.device.allocate_memory(&alloc_info, None)
                    .map_err(|e| format!("vkAllocateMemory failed: {}", e))?
            };

            unsafe {
                self.device.bind_buffer_memory(buffer, memory, 0)
                    .map_err(|e| format!("vkBindBufferMemory failed: {}", e))?;
            }

            Ok((buffer, memory))
        }

        /// Dispatches 3D PGA motor rigid transformations over an [N, 3] tensor batch directly in GPU VRAM.
        pub fn dispatch_pga_transform(
            &self,
            points: &[[f32; 3]],
            motor: &PgaMotor,
        ) -> Result<Vec<[f32; 3]>, String> {
            let n = points.len();
            if n == 0 {
                return Ok(Vec::new());
            }

            let mut in_data = Vec::<[f32; 4]>::with_capacity(n);
            for p in points {
                in_data.push([p[0], p[1], p[2], 1.0]);
            }

            let size = (n * std::mem::size_of::<[f32; 4]>()) as vk::DeviceSize;

            let (in_buf, in_mem) = self.create_buffer(
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let (out_buf, out_mem) = self.create_buffer(
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            // Copy input points to GPU VRAM
            unsafe {
                let ptr = self.device.map_memory(in_mem, 0, size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("vkMapMemory failed: {}", e))? as *mut [f32; 4];
                std::ptr::copy_nonoverlapping(in_data.as_ptr(), ptr, n);
                self.device.unmap_memory(in_mem);
            }

            let binding_in = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE);

            let binding_out = vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE);

            let bindings = [binding_in, binding_out];
            let desc_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let desc_layout = unsafe {
                self.device.create_descriptor_set_layout(&desc_layout_info, None)
                    .map_err(|e| format!("vkCreateDescriptorSetLayout failed: {}", e))?
            };

            let push_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<PgaTransformPushConstants>() as u32);

            let push_ranges = [push_range];
            let set_layouts = [desc_layout];
            let pipe_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_ranges);

            let pipeline_layout = unsafe {
                self.device.create_pipeline_layout(&pipe_layout_info, None)
                    .map_err(|e| format!("vkCreatePipelineLayout failed: {}", e))?
            };

            let shader_module = self.create_shader_module(PGA_TRANSFORM_SPV)?;
            let entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(entry_name);

            let pipe_create_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(pipeline_layout);

            let pipeline = unsafe {
                self.device.create_compute_pipelines(vk::PipelineCache::null(), &[pipe_create_info], None)
                    .map_err(|(_, e)| format!("create_compute_pipelines failed: {:?}", e))?[0]
            };

            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2);

            let pool_sizes = [pool_size];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);

            let desc_pool = unsafe {
                self.device.create_descriptor_pool(&pool_info, None)
                    .map_err(|e| format!("vkCreateDescriptorPool failed: {}", e))?
            };

            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&set_layouts);

            let desc_set = unsafe {
                self.device.allocate_descriptor_sets(&alloc_info)
                    .map_err(|e| format!("vkAllocateDescriptorSets failed: {}", e))?[0]
            };

            let buf_in_info = vk::DescriptorBufferInfo::default()
                .buffer(in_buf)
                .offset(0)
                .range(size);
            let buf_out_info = vk::DescriptorBufferInfo::default()
                .buffer(out_buf)
                .offset(0)
                .range(size);

            let write_in = vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_in_info));

            let write_out = vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_out_info));

            unsafe {
                self.device.update_descriptor_sets(&[write_in, write_out], &[]);
            }

            let cmd_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family_index);
            let cmd_pool = unsafe {
                self.device.create_command_pool(&cmd_pool_info, None)
                    .map_err(|e| format!("vkCreateCommandPool failed: {}", e))?
            };

            let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let cmd_buf = unsafe {
                self.device.allocate_command_buffers(&cmd_alloc_info)
                    .map_err(|e| format!("vkAllocateCommandBuffers failed: {}", e))?[0]
            };

            let pc = PgaTransformPushConstants::new(n as u32, motor);

            unsafe {
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.device.begin_command_buffer(cmd_buf, &begin_info)
                    .map_err(|e| format!("vkBeginCommandBuffer failed: {}", e))?;

                self.device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline_layout,
                    0,
                    &[desc_set],
                    &[],
                );

                self.device.cmd_push_constants(
                    cmd_buf,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&pc),
                );

                let workgroups = ((n as u32) + 63) / 64;
                self.device.cmd_dispatch(cmd_buf, workgroups, 1, 1);

                self.device.end_command_buffer(cmd_buf)
                    .map_err(|e| format!("vkEndCommandBuffer failed: {}", e))?;
            }

            let fence = unsafe {
                self.device.create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(|e| format!("vkCreateFence failed: {}", e))?
            };

            let cmd_bufs = [cmd_buf];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
            unsafe {
                self.device.queue_submit(self.compute_queue, &[submit_info], fence)
                    .map_err(|e| format!("vkQueueSubmit failed: {}", e))?;
                self.device.wait_for_fences(&[fence], true, u64::MAX)
                    .map_err(|e| format!("vkWaitForFences failed: {}", e))?;
            }

            let mut results = Vec::<[f32; 3]>::with_capacity(n);
            unsafe {
                let ptr = self.device.map_memory(out_mem, 0, size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("vkMapMemory failed: {}", e))? as *const [f32; 4];
                for i in 0..n {
                    let p4 = *ptr.add(i);
                    results.push([p4[0], p4[1], p4[2]]);
                }
                self.device.unmap_memory(out_mem);
            }

            unsafe {
                self.device.destroy_fence(fence, None);
                self.device.destroy_command_pool(cmd_pool, None);
                self.device.destroy_descriptor_pool(desc_pool, None);
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_pipeline_layout(pipeline_layout, None);
                self.device.destroy_shader_module(shader_module, None);
                self.device.destroy_descriptor_set_layout(desc_layout, None);
                self.device.destroy_buffer(in_buf, None);
                self.device.free_memory(in_mem, None);
                self.device.destroy_buffer(out_buf, None);
                self.device.free_memory(out_mem, None);
            }

            Ok(results)
        }

        /// Dispatches dense residual vector addition on GPU.
        pub fn dispatch_vec_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
            assert_eq!(a.len(), b.len(), "Vectors a and b must have identical length");
            let n = a.len();
            if n == 0 {
                return Ok(Vec::new());
            }

            let size = (n * std::mem::size_of::<f32>()) as vk::DeviceSize;

            let (in_buf, in_mem) = self.create_buffer(
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let (out_buf, out_mem) = self.create_buffer(
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            unsafe {
                let ptr_in = self.device.map_memory(in_mem, 0, size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("vkMapMemory in failed: {}", e))? as *mut f32;
                std::ptr::copy_nonoverlapping(a.as_ptr(), ptr_in, n);
                self.device.unmap_memory(in_mem);

                let ptr_out = self.device.map_memory(out_mem, 0, size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("vkMapMemory out failed: {}", e))? as *mut f32;
                std::ptr::copy_nonoverlapping(b.as_ptr(), ptr_out, n);
                self.device.unmap_memory(out_mem);
            }

            let binding0 = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE);

            let binding1 = vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE);

            let desc_layout = unsafe {
                self.device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[binding0, binding1]),
                    None,
                ).map_err(|e| format!("create_descriptor_set_layout failed: {}", e))?
            };

            let push_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<VecAddPushConstants>() as u32);

            let pipeline_layout = unsafe {
                self.device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[desc_layout])
                        .push_constant_ranges(&[push_range]),
                    None,
                ).map_err(|e| format!("create_pipeline_layout failed: {}", e))?
            };

            let shader_module = self.create_shader_module(VEC_ADD_SPV)?;
            let entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
            let pipeline = unsafe {
                self.device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(shader_module)
                            .name(entry_name))
                        .layout(pipeline_layout)],
                    None,
                ).map_err(|(_, e)| format!("create_compute_pipelines failed: {:?}", e))?[0]
            };

            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2);
            let desc_pool = unsafe {
                self.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[pool_size]),
                    None,
                ).map_err(|e| format!("create_descriptor_pool failed: {}", e))?
            };

            let desc_set = unsafe {
                self.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&[desc_layout]),
                ).map_err(|e| format!("allocate_descriptor_sets failed: {}", e))?[0]
            };

            let buf0_info = vk::DescriptorBufferInfo::default().buffer(in_buf).offset(0).range(size);
            let buf1_info = vk::DescriptorBufferInfo::default().buffer(out_buf).offset(0).range(size);
            let write0 = vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&buf0_info));
            let write1 = vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&buf1_info));

            unsafe {
                self.device.update_descriptor_sets(&[write0, write1], &[]);
            }

            let cmd_pool = unsafe {
                self.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family_index),
                    None,
                ).map_err(|e| format!("create_command_pool failed: {}", e))?
            };

            let cmd_buf = unsafe {
                self.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
                ).map_err(|e| format!("allocate_command_buffers failed: {}", e))?[0]
            };

            let pc = VecAddPushConstants::new(n as u32);

            unsafe {
                self.device.begin_command_buffer(cmd_buf, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                    .map_err(|e| format!("begin_command_buffer failed: {}", e))?;
                self.device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.device.cmd_bind_descriptor_sets(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[desc_set], &[]);
                self.device.cmd_push_constants(cmd_buf, pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc));
                let workgroups = ((n as u32) + 63) / 64;
                self.device.cmd_dispatch(cmd_buf, workgroups, 1, 1);
                self.device.end_command_buffer(cmd_buf)
                    .map_err(|e| format!("end_command_buffer failed: {}", e))?;
            }

            let fence = unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("create_fence failed: {}", e))? };
            unsafe {
                self.device.queue_submit(self.compute_queue, &[vk::SubmitInfo::default().command_buffers(&[cmd_buf])], fence)
                    .map_err(|e| format!("queue_submit failed: {}", e))?;
                self.device.wait_for_fences(&[fence], true, u64::MAX)
                    .map_err(|e| format!("wait_for_fences failed: {}", e))?;
            }

            let mut results = vec![0.0f32; n];
            unsafe {
                let ptr = self.device.map_memory(out_mem, 0, size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("map_memory failed: {}", e))? as *const f32;
                std::ptr::copy_nonoverlapping(ptr, results.as_mut_ptr(), n);
                self.device.unmap_memory(out_mem);
            }

            unsafe {
                self.device.destroy_fence(fence, None);
                self.device.destroy_command_pool(cmd_pool, None);
                self.device.destroy_descriptor_pool(desc_pool, None);
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_pipeline_layout(pipeline_layout, None);
                self.device.destroy_shader_module(shader_module, None);
                self.device.destroy_descriptor_set_layout(desc_layout, None);
                self.device.destroy_buffer(in_buf, None);
                self.device.free_memory(in_mem, None);
                self.device.destroy_buffer(out_buf, None);
                self.device.free_memory(out_mem, None);
            }

            Ok(results)
        }
    }

    impl Drop for GpuComputeContext {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }
}
