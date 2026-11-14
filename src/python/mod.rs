// ============================================================================
// VERSORC-MATH: PURE PYO3 NATIVE BINDINGS & PEP 3118 BUFFER BRIDGE
// License: Apache-2.0
// ============================================================================

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBufferError, PyIndexError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PySlice, PyTuple, PyType};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use crate::cga::{CgaCircle, CgaConformalMotor, CgaPoint, CgaSphere};
#[cfg(feature = "vulkan")]
use crate::gpu::shaders::CgaTransformPushConstants;
#[cfg(feature = "vulkan")]
use crate::gpu::vulkan::GpuComputeContext;
use crate::lattice::E8Lattice;
use crate::multivector::{Cl2, Cl3};
use crate::pga::{PgaLine, PgaMotor, PgaPlane, PgaPoint};
use crate::poincare::cayley_klein::CayleyKleinSpace;
use crate::poincare::PoincarePoint;
use crate::quantization::clifford_3bit::{
    Clifford3BitCacheLine64, Clifford3BitTensor, LLOYD_MAX_3BIT_CENTROIDS,
};
use crate::sta::{LounestoClass, StaSpinor, StaVector};
use crate::versor::{DualQuaternion, DyadicWalshVersor, Quaternion};

// ============================================================================
// PEP 3118 BUFFER PROTOCOL & ARRAY INTERFACE HELPERS
// ============================================================================

struct ExportedBufferMeta {
    shape: Box<[isize]>,
    strides: Box<[isize]>,
}

trait BufferElement: Copy {
    const FORMAT: &'static CStr;
    const NUMPY_TYPESTR: &'static str;
    const DTYPE_NAME: &'static str;

    fn itemsize() -> isize {
        std::mem::size_of::<Self>() as isize
    }
}

impl BufferElement for f32 {
    const FORMAT: &'static CStr = c"f";
    #[cfg(target_endian = "little")]
    const NUMPY_TYPESTR: &'static str = "<f4";
    #[cfg(target_endian = "big")]
    const NUMPY_TYPESTR: &'static str = ">f4";
    const DTYPE_NAME: &'static str = "float32";
}

impl BufferElement for i32 {
    const FORMAT: &'static CStr = c"i";
    #[cfg(target_endian = "little")]
    const NUMPY_TYPESTR: &'static str = "<i4";
    #[cfg(target_endian = "big")]
    const NUMPY_TYPESTR: &'static str = ">i4";
    const DTYPE_NAME: &'static str = "int32";
}

unsafe fn fill_readonly_py_buffer<T: BufferElement>(
    obj: Py<PyAny>,
    data: &[T],
    shape: &[usize],
    view: *mut ffi::Py_buffer,
    flags: c_int,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("Py_buffer view pointer is null"));
    }
    if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
        return Err(PyBufferError::new_err(
            "VersorC buffer objects are read-only",
        ));
    }

    let expected_elems = shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, d| acc.checked_mul(d))
        .ok_or_else(|| PyBufferError::new_err("Buffer shape overflow"))?;
    if expected_elems != data.len() {
        return Err(PyBufferError::new_err(
            "Buffer shape does not match backing storage length",
        ));
    }

    let itemsize = T::itemsize();
    let byte_len = (data.len() as isize)
        .checked_mul(itemsize)
        .ok_or_else(|| PyBufferError::new_err("Buffer byte length overflow"))?;

    let mut shape_vec = Vec::with_capacity(shape.len());
    for &d in shape {
        shape_vec.push(d as isize);
    }
    let mut strides_vec = vec![0isize; shape.len()];
    let mut stride = itemsize;
    for i in (0..shape.len()).rev() {
        strides_vec[i] = stride;
        stride = stride.saturating_mul(shape_vec[i].max(1));
    }

    let mut meta = Box::new(ExportedBufferMeta {
        shape: shape_vec.into_boxed_slice(),
        strides: strides_vec.into_boxed_slice(),
    });

    unsafe {
        (*view).buf = data.as_ptr() as *mut c_void;
        (*view).obj = obj.into_ptr();
        (*view).len = byte_len;
        (*view).readonly = 1;
        (*view).itemsize = itemsize;
        (*view).format = T::FORMAT.as_ptr() as *mut c_char;
        (*view).ndim = shape.len() as c_int;
        (*view).shape = meta.shape.as_mut_ptr();
        (*view).strides = meta.strides.as_mut_ptr();
        (*view).suboffsets = ptr::null_mut();
        (*view).internal = Box::into_raw(meta) as *mut c_void;
    }

    Ok(())
}

unsafe fn release_py_buffer(view: *mut ffi::Py_buffer) {
    if !view.is_null() {
        let internal = unsafe { (*view).internal };
        if !internal.is_null() {
            unsafe {
                drop(Box::from_raw(internal as *mut ExportedBufferMeta));
                (*view).internal = ptr::null_mut();
                (*view).shape = ptr::null_mut();
                (*view).strides = ptr::null_mut();
            }
        }
    }
}

fn build_array_interface<T: BufferElement>(
    py: Python<'_>,
    data: &[T],
    shape: &[usize],
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let py_shape = PyTuple::new(py, shape)?;
    dict.set_item("shape", py_shape)?;
    dict.set_item("typestr", T::NUMPY_TYPESTR)?;
    dict.set_item("data", (data.as_ptr() as usize, true))?;
    dict.set_item("version", 3)?;
    Ok(dict.unbind())
}

fn extract_f32_1d(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        if buf.dimensions() != 1 {
            return Err(PyValueError::new_err(format!(
                "Expected a 1D float32 buffer, got {}D",
                buf.dimensions()
            )));
        }
        let len = buf.item_count();
        let mut out = vec![0.0f32; len];
        if let Some(slice) = buf.as_slice(py) {
            for (dst, cell) in out.iter_mut().zip(slice.iter()) {
                *dst = cell.get();
            }
        } else {
            buf.copy_to_slice(py, &mut out)?;
        }
        return Ok(out);
    }

    // Reject incompatible PEP 3118 typed buffers explicitly rather than coercing via iteration.
    if unsafe { ffi::PyObject_CheckBuffer(obj.as_ptr()) } == 1 {
        let buf = PyBuffer::<f32>::get(obj)?;
        return Err(PyValueError::new_err(format!(
            "Expected a 1D float32 buffer, got {}D",
            buf.dimensions()
        )));
    }

    obj.extract::<Vec<f32>>()
}

fn extract_f32_2d(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, usize, usize)> {
    if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        let shape = buf.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Expected a 2D float32 buffer, got {}D",
                shape.len()
            )));
        }
        let (rows, cols) = (shape[0], shape[1]);
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| PyValueError::new_err("2D buffer size overflow"))?;
        let mut out = vec![0.0f32; total];
        if !out.is_empty() {
            buf.copy_to_slice(py, &mut out)?;
        }
        return Ok((out, rows, cols));
    }

    if unsafe { ffi::PyObject_CheckBuffer(obj.as_ptr()) } == 1 {
        let buf = PyBuffer::<f32>::get(obj)?;
        return Err(PyValueError::new_err(format!(
            "Expected a 2D float32 buffer, got {}D",
            buf.shape().len()
        )));
    }

    let rows_vec: Vec<Vec<f32>> = obj.extract()?;
    let rows = rows_vec.len();
    let cols = rows_vec.first().map(|r| r.len()).unwrap_or(0);
    let mut flat = Vec::with_capacity(rows.saturating_mul(cols));
    for (r_idx, row) in rows_vec.iter().enumerate() {
        if row.len() != cols {
            return Err(PyValueError::new_err(format!(
                "Ragged 2D sequence at row {}: expected {} columns, got {}",
                r_idx,
                cols,
                row.len()
            )));
        }
        flat.extend_from_slice(row);
    }
    Ok((flat, rows, cols))
}

fn normalize_index(idx: isize, len: usize) -> PyResult<usize> {
    let len_isize = len as isize;
    let resolved = if idx < 0 { len_isize + idx } else { idx };
    if resolved < 0 || resolved >= len_isize {
        Err(PyIndexError::new_err(format!(
            "Index {} is out of bounds for axis with size {}",
            idx, len
        )))
    } else {
        Ok(resolved as usize)
    }
}

fn collect_stepped_range(
    slice: &Bound<'_, PySlice>,
    len: usize,
) -> PyResult<(isize, isize, usize)> {
    let indices = slice.indices(len as isize)?;
    Ok((indices.start, indices.step, indices.slicelength))
}

// ============================================================================
// NATIVE ZERO-DEPENDENCY BUFFER OUTPUT TYPES (PEP 3118 + __array_interface__)
// ============================================================================

/// Read-only 1D or 2D `float32` buffer exposing PEP 3118 (`memoryview`) and `__array_interface__`.
#[pyclass(name = "VersorBufferF32")]
#[derive(Clone)]
pub struct VersorBufferF32 {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl VersorBufferF32 {
    pub fn from_vec_1d(data: Vec<f32>) -> Self {
        let len = data.len();
        Self {
            data,
            shape: vec![len],
        }
    }

    pub fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> PyResult<Self> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(PyValueError::new_err("2D buffer shape mismatch"));
        }
        Ok(Self {
            data,
            shape: vec![rows, cols],
        })
    }
}

#[pymethods]
impl VersorBufferF32 {
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, &self.shape)?.unbind())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[getter]
    fn size(&self) -> usize {
        self.data.len()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        f32::DTYPE_NAME
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    #[getter]
    fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        build_array_interface(py, &self.data, &self.shape)
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.shape.len() == 2 {
            let (rows, cols) = (self.shape[0], self.shape[1]);
            let py_rows: Vec<Bound<'_, PyList>> = (0..rows)
                .map(|r| PyList::new(py, &self.data[r * cols..(r + 1) * cols]))
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, py_rows)?.into_any().unbind())
        } else {
            Ok(PyList::new(py, &self.data)?.into_any().unbind())
        }
    }

    fn to_list(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.tolist(py)
    }

    fn tobytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for &v in &self.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        PyBytes::new(py, &bytes)
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        self.tobytes(py)
    }

    fn __len__(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list_obj = self.tolist(py)?;
        Ok(list_obj.bind(py).call_method0("__iter__")?.unbind())
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let outer = self.shape.first().copied().unwrap_or(0);
        if self.shape.len() == 2 {
            let cols = self.shape[1];
            if let Ok((r_raw, c_raw)) = key.extract::<(isize, isize)>() {
                let r = normalize_index(r_raw, outer)?;
                let c = normalize_index(c_raw, cols)?;
                return Ok(self.data[r * cols + c]
                    .into_pyobject(py)?
                    .into_any()
                    .unbind());
            }
            if let Ok(r_raw) = key.extract::<isize>() {
                let r = normalize_index(r_raw, outer)?;
                let row = self.data[r * cols..(r + 1) * cols].to_vec();
                return Ok(Py::new(py, Self::from_vec_1d(row))?.into_any());
            }
            if let Ok(slice) = key.cast::<PySlice>() {
                let (start, step, count) = collect_stepped_range(slice, outer)?;
                let mut sliced = Vec::with_capacity(count * cols);
                for k in 0..count {
                    let r = (start + (k as isize) * step) as usize;
                    sliced.extend_from_slice(&self.data[r * cols..(r + 1) * cols]);
                }
                return Ok(Py::new(py, Self::from_vec_2d(sliced, count, cols)?)?.into_any());
            }
            return Err(PyTypeError::new_err(
                "Indices must be int, slice, or (row, col) tuple",
            ));
        }

        if let Ok(idx_raw) = key.extract::<isize>() {
            let idx = normalize_index(idx_raw, outer)?;
            return Ok(self.data[idx].into_pyobject(py)?.into_any().unbind());
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            let (start, step, count) = collect_stepped_range(slice, outer)?;
            let mut sliced = Vec::with_capacity(count);
            for k in 0..count {
                let idx = (start + (k as isize) * step) as usize;
                sliced.push(self.data[idx]);
            }
            return Ok(Py::new(py, Self::from_vec_1d(sliced))?.into_any());
        }
        Err(PyTypeError::new_err("Indices must be integers or slices"))
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let borrow = slf.borrow();
        let data_ptr = borrow.data.as_ptr();
        let data_len = borrow.data.len();
        let shape = borrow.shape.clone();
        drop(borrow);
        let slice = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
        unsafe { fill_readonly_py_buffer(slf.into_any().unbind(), slice, &shape, view, flags) }
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe { release_py_buffer(view) };
    }

    fn __repr__(&self) -> String {
        format!("VersorBufferF32(shape={:?}, dtype='float32')", self.shape)
    }
}

/// Read-only 1D or 2D `int32` buffer exposing PEP 3118 (`memoryview`) and `__array_interface__`.
#[pyclass(name = "VersorBufferI32")]
#[derive(Clone)]
pub struct VersorBufferI32 {
    data: Vec<i32>,
    shape: Vec<usize>,
}

impl VersorBufferI32 {
    pub fn from_vec_1d(data: Vec<i32>) -> Self {
        let len = data.len();
        Self {
            data,
            shape: vec![len],
        }
    }
}

#[pymethods]
impl VersorBufferI32 {
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, &self.shape)?.unbind())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[getter]
    fn size(&self) -> usize {
        self.data.len()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        i32::DTYPE_NAME
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<i32>()
    }

    #[getter]
    fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        build_array_interface(py, &self.data, &self.shape)
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyList::new(py, &self.data)?.into_any().unbind())
    }

    fn to_list(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.tolist(py)
    }

    fn tobytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for &v in &self.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        PyBytes::new(py, &bytes)
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        self.tobytes(py)
    }

    fn __len__(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list_obj = self.tolist(py)?;
        Ok(list_obj.bind(py).call_method0("__iter__")?.unbind())
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let len = self.data.len();
        if let Ok(idx_raw) = key.extract::<isize>() {
            let idx = normalize_index(idx_raw, len)?;
            return Ok(self.data[idx].into_pyobject(py)?.into_any().unbind());
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            let (start, step, count) = collect_stepped_range(slice, len)?;
            let mut sliced = Vec::with_capacity(count);
            for k in 0..count {
                let idx = (start + (k as isize) * step) as usize;
                sliced.push(self.data[idx]);
            }
            return Ok(Py::new(py, Self::from_vec_1d(sliced))?.into_any());
        }
        Err(PyTypeError::new_err("Indices must be integers or slices"))
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let borrow = slf.borrow();
        let data_ptr = borrow.data.as_ptr();
        let data_len = borrow.data.len();
        let shape = borrow.shape.clone();
        drop(borrow);
        let slice = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
        unsafe { fill_readonly_py_buffer(slf.into_any().unbind(), slice, &shape, view, flags) }
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe { release_py_buffer(view) };
    }

    fn __repr__(&self) -> String {
        format!("VersorBufferI32(shape={:?}, dtype='int32')", self.shape)
    }
}

// ============================================================================
// FORMATTING HELPERS
// ============================================================================

fn format_cl3(c: &[f32; 8]) -> String {
    let names = ["", "e1", "e2", "e12", "e3", "e13", "e23", "e123"];
    let mut parts = Vec::new();
    for (i, (&coeff, &name)) in c.iter().zip(names.iter()).enumerate() {
        if coeff.abs() > 1e-6 {
            if i == 0 {
                parts.push(format!("{:.4}", coeff));
            } else if (coeff - 1.0).abs() < 1e-6 {
                parts.push(name.to_string());
            } else if (coeff + 1.0).abs() < 1e-6 {
                parts.push(format!("-{}", name));
            } else {
                parts.push(format!("{:.4}*{}", coeff, name));
            }
        }
    }
    if parts.is_empty() {
        "0.0".to_string()
    } else {
        parts.join(" + ").replace("+ -", "- ")
    }
}

fn format_cl2(c: &[f32; 4]) -> String {
    let names = ["", "e1", "e2", "e12"];
    let mut parts = Vec::new();
    for (i, (&coeff, &name)) in c.iter().zip(names.iter()).enumerate() {
        if coeff.abs() > 1e-6 {
            if i == 0 {
                parts.push(format!("{:.4}", coeff));
            } else if (coeff - 1.0).abs() < 1e-6 {
                parts.push(name.to_string());
            } else if (coeff + 1.0).abs() < 1e-6 {
                parts.push(format!("-{}", name));
            } else {
                parts.push(format!("{:.4}*{}", coeff, name));
            }
        }
    }
    if parts.is_empty() {
        "0.0".to_string()
    } else {
        parts.join(" + ").replace("+ -", "- ")
    }
}

// ============================================================================
// PYO3 WRAPPER: Cl3 (3D EUCLIDEAN GEOMETRIC ALGEBRA)
// ============================================================================

#[pyclass(name = "Cl3")]
#[derive(Clone, Copy)]
pub struct PyCl3 {
    pub inner: Cl3,
}

#[pymethods]
impl PyCl3 {
    #[new]
    #[pyo3(signature = (coords = None))]
    fn new(coords: Option<Vec<f32>>) -> PyResult<Self> {
        let mut arr = [0.0f32; 8];
        if let Some(c) = coords {
            if c.len() > 8 {
                return Err(PyValueError::new_err(format!(
                    "Cl3 expects at most 8 coordinates, got {}",
                    c.len()
                )));
            }
            for (i, &v) in c.iter().enumerate() {
                arr[i] = v;
            }
        }
        Ok(Self {
            inner: Cl3 { coords: arr },
        })
    }

    #[classmethod]
    fn from_scalar(_cls: &Bound<'_, PyType>, s: f32) -> Self {
        Self {
            inner: Cl3::from_scalar(s),
        }
    }

    #[classmethod]
    fn vector(_cls: &Bound<'_, PyType>, x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: Cl3::from_vector(&[x, y, z]),
        }
    }

    #[classmethod]
    fn from_vector(_cls: &Bound<'_, PyType>, x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: Cl3::from_vector(&[x, y, z]),
        }
    }

    #[classmethod]
    fn bivector(_cls: &Bound<'_, PyType>, e12: f32, e13: f32, e23: f32) -> Self {
        let mut arr = [0.0f32; 8];
        arr[3] = e12;
        arr[5] = e13;
        arr[6] = e23;
        Self {
            inner: Cl3 { coords: arr },
        }
    }

    #[classmethod]
    fn e1_basis(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Cl3::basis(1),
        }
    }

    #[classmethod]
    #[pyo3(name = "e1")]
    fn cls_e1(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Cl3::basis(1),
        }
    }

    #[classmethod]
    #[pyo3(name = "e2")]
    fn cls_e2(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Cl3::basis(2),
        }
    }

    #[classmethod]
    #[pyo3(name = "e3")]
    fn cls_e3(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Cl3::basis(3),
        }
    }

    #[classmethod]
    #[allow(non_snake_case)]
    fn I(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Cl3::pseudoscalar(),
        }
    }

    #[classmethod]
    fn from_buffer(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        buf: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let slice = extract_f32_1d(py, buf)?;
        if slice.len() != 8 {
            return Err(PyValueError::new_err(format!(
                "Cl3 requires 8 coordinates, got {}",
                slice.len()
            )));
        }
        let coords: [f32; 8] = slice.try_into().unwrap();
        Ok(Self {
            inner: Cl3 { coords },
        })
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_buffer(&self) -> VersorBufferF32 {
        VersorBufferF32::from_vec_1d(self.inner.coords.to_vec())
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_list(&self) -> Vec<f32> {
        self.inner.coords.to_vec()
    }

    fn tolist(&self) -> Vec<f32> {
        self.inner.coords.to_vec()
    }

    #[getter]
    fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        build_array_interface(py, &self.inner.coords, &[8])
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let borrow = slf.borrow();
        let ptr = borrow.inner.coords.as_ptr();
        drop(borrow);
        let slice = unsafe { std::slice::from_raw_parts(ptr, 8) };
        unsafe { fill_readonly_py_buffer(slf.into_any().unbind(), slice, &[8], view, flags) }
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe { release_py_buffer(view) };
    }

    // Dunder operators
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl3>() {
            Ok(Self {
                inner: self.inner + other_cl.inner,
            })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self {
                inner: self.inner + Cl3::from_scalar(s),
            })
        } else {
            Err(PyValueError::new_err(
                "Addition only supported with Cl3 or float",
            ))
        }
    }

    fn __radd__(&self, other: f32) -> Self {
        Self {
            inner: self.inner + Cl3::from_scalar(other),
        }
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl3>() {
            Ok(Self {
                inner: self.inner - other_cl.inner,
            })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self {
                inner: self.inner - Cl3::from_scalar(s),
            })
        } else {
            Err(PyValueError::new_err(
                "Subtraction only supported with Cl3 or float",
            ))
        }
    }

    fn __rsub__(&self, other: f32) -> Self {
        Self {
            inner: Cl3::from_scalar(other) - self.inner,
        }
    }

    fn __neg__(&self) -> Self {
        Self { inner: -self.inner }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl3>() {
            Ok(Self {
                inner: self.inner * other_cl.inner,
            })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self {
                inner: self.inner * s,
            })
        } else {
            Err(PyValueError::new_err(
                "Multiplication only supported with Cl3 or float",
            ))
        }
    }

    fn __rmul__(&self, scalar: f32) -> Self {
        Self {
            inner: self.inner * scalar,
        }
    }

    fn __xor__(&self, other: &PyCl3) -> Self {
        Self {
            inner: self.inner.wedge(&other.inner),
        }
    }

    fn __or__(&self, other: &PyCl3) -> Self {
        Self {
            inner: self.inner.left_contraction(&other.inner),
        }
    }

    fn __invert__(&self) -> Self {
        Self {
            inner: self.inner.reverse(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Cl3({})", format_cl3(&self.inner.coords))
    }

    fn __str__(&self) -> String {
        format_cl3(&self.inner.coords)
    }

    fn grade(&self, k: usize) -> Self {
        Self {
            inner: self.inner.grade_projection(k),
        }
    }

    #[getter]
    fn scalar(&self) -> f32 {
        self.inner.scalar()
    }

    #[getter]
    fn x(&self) -> f32 {
        self.inner.coords[1]
    }

    #[getter]
    fn y(&self) -> f32 {
        self.inner.coords[2]
    }

    #[getter]
    fn z(&self) -> f32 {
        self.inner.coords[4]
    }

    fn __getitem__(&self, idx: isize) -> PyResult<f32> {
        let len = 8isize;
        let norm = if idx < 0 { len + idx } else { idx };
        if norm < 0 || norm >= len {
            return Err(PyIndexError::new_err("Cl3 index out of range"));
        }
        Ok(self.inner.coords[norm as usize])
    }

    #[getter]
    fn e12(&self) -> f32 {
        self.inner.coords[3]
    }

    #[getter]
    fn e13(&self) -> f32 {
        self.inner.coords[5]
    }

    #[getter]
    fn e23(&self) -> f32 {
        self.inner.coords[6]
    }

    #[getter]
    fn e123(&self) -> f32 {
        self.inner.coords[7]
    }

    fn reverse(&self) -> Self {
        Self {
            inner: self.inner.reverse(),
        }
    }

    fn involution(&self) -> Self {
        Self {
            inner: self.inner.grade_involution(),
        }
    }

    fn conjugation(&self) -> Self {
        Self {
            inner: self.inner.conjugation(),
        }
    }

    fn dual(&self) -> PyResult<Self> {
        self.inner
            .dual()
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("Dual not invertible"))
    }

    fn undual(&self) -> Self {
        Self {
            inner: self.inner.undual(),
        }
    }

    fn commutator(&self, other: &PyCl3) -> Self {
        Self {
            inner: self.inner.commutator(&other.inner),
        }
    }

    fn anticommutator(&self, other: &PyCl3) -> Self {
        Self {
            inner: self.inner.anticommutator(&other.inner),
        }
    }

    fn regressive(&self, other: &PyCl3) -> PyResult<Self> {
        self.inner
            .regressive_product(&other.inner)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                PyValueError::new_err("Regressive product requires invertible pseudoscalar")
            })
    }

    fn hestenes_dot(&self, other: &PyCl3) -> Self {
        Self {
            inner: self.inner.hestenes_inner_product(&other.inner),
        }
    }

    fn inverse(&self) -> PyResult<Self> {
        self.inner
            .inverse()
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("Multivector is singular (det == 0)"))
    }

    fn norm(&self) -> f32 {
        self.inner.norm()
    }

    fn norm_sq(&self) -> f32 {
        self.inner.norm_sq()
    }

    fn normalize(&self) -> Self {
        Self {
            inner: self.inner.normalize(),
        }
    }

    fn sandwich(&self, x: &PyCl3) -> Self {
        Self {
            inner: self.inner.sandwich(&x.inner),
        }
    }
}

// ============================================================================
// PYO3 WRAPPER: Cl2 (2D PLANAR GEOMETRIC ALGEBRA)
// ============================================================================

#[pyclass(name = "Cl2")]
#[derive(Clone, Copy)]
pub struct PyCl2 {
    pub inner: Cl2,
}

#[pymethods]
impl PyCl2 {
    #[new]
    #[pyo3(signature = (coords = None))]
    fn new(coords: Option<Vec<f32>>) -> PyResult<Self> {
        let mut arr = [0.0f32; 4];
        if let Some(c) = coords {
            if c.len() > 4 {
                return Err(PyValueError::new_err(format!(
                    "Cl2 expects at most 4 coordinates, got {}",
                    c.len()
                )));
            }
            for (i, &v) in c.iter().enumerate() {
                arr[i] = v;
            }
        }
        Ok(Self {
            inner: Cl2 { coords: arr },
        })
    }

    #[classmethod]
    fn from_scalar(_cls: &Bound<'_, PyType>, s: f32) -> Self {
        Self {
            inner: Cl2::from_scalar(s),
        }
    }

    #[classmethod]
    fn vector(_cls: &Bound<'_, PyType>, x: f32, y: f32) -> Self {
        Self {
            inner: Cl2::from_vector(&[x, y]),
        }
    }

    #[classmethod]
    fn from_vector(_cls: &Bound<'_, PyType>, x: f32, y: f32) -> Self {
        Self {
            inner: Cl2::from_vector(&[x, y]),
        }
    }

    #[classmethod]
    #[pyo3(name = "e1")]
    fn cls_e1(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Cl2::basis(1),
        }
    }

    #[classmethod]
    #[pyo3(name = "e2")]
    fn cls_e2(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Cl2::basis(2),
        }
    }

    #[classmethod]
    #[allow(non_snake_case)]
    fn I(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Cl2::pseudoscalar(),
        }
    }

    #[classmethod]
    fn from_buffer(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        buf: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let slice = extract_f32_1d(py, buf)?;
        if slice.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "Cl2 requires 4 coordinates, got {}",
                slice.len()
            )));
        }
        let coords: [f32; 4] = slice.try_into().unwrap();
        Ok(Self {
            inner: Cl2 { coords },
        })
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_buffer(&self) -> VersorBufferF32 {
        VersorBufferF32::from_vec_1d(self.inner.coords.to_vec())
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_list(&self) -> Vec<f32> {
        self.inner.coords.to_vec()
    }

    fn tolist(&self) -> Vec<f32> {
        self.inner.coords.to_vec()
    }

    #[getter]
    fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        build_array_interface(py, &self.inner.coords, &[4])
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let borrow = slf.borrow();
        let ptr = borrow.inner.coords.as_ptr();
        drop(borrow);
        let slice = unsafe { std::slice::from_raw_parts(ptr, 4) };
        unsafe { fill_readonly_py_buffer(slf.into_any().unbind(), slice, &[4], view, flags) }
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe { release_py_buffer(view) };
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl2>() {
            Ok(Self {
                inner: self.inner + other_cl.inner,
            })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self {
                inner: self.inner + Cl2::from_scalar(s),
            })
        } else {
            Err(PyValueError::new_err(
                "Addition only supported with Cl2 or float",
            ))
        }
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl2>() {
            Ok(Self {
                inner: self.inner - other_cl.inner,
            })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self {
                inner: self.inner - Cl2::from_scalar(s),
            })
        } else {
            Err(PyValueError::new_err(
                "Subtraction only supported with Cl2 or float",
            ))
        }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl2>() {
            Ok(Self {
                inner: self.inner * other_cl.inner,
            })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self {
                inner: self.inner * s,
            })
        } else {
            Err(PyValueError::new_err(
                "Multiplication only supported with Cl2 or float",
            ))
        }
    }

    fn __rmul__(&self, scalar: f32) -> Self {
        Self {
            inner: self.inner * scalar,
        }
    }

    fn __xor__(&self, other: &PyCl2) -> Self {
        Self {
            inner: self.inner.wedge(&other.inner),
        }
    }

    fn __or__(&self, other: &PyCl2) -> Self {
        Self {
            inner: self.inner.left_contraction(&other.inner),
        }
    }

    fn __invert__(&self) -> Self {
        Self {
            inner: self.inner.reverse(),
        }
    }

    fn __neg__(&self) -> Self {
        Self { inner: -self.inner }
    }

    fn __repr__(&self) -> String {
        format!("Cl2({})", format_cl2(&self.inner.coords))
    }

    fn __str__(&self) -> String {
        format_cl2(&self.inner.coords)
    }

    #[getter]
    fn scalar(&self) -> f32 {
        self.inner.scalar()
    }

    #[getter]
    fn x(&self) -> f32 {
        self.inner.coords[1]
    }

    #[getter]
    fn y(&self) -> f32 {
        self.inner.coords[2]
    }

    fn __getitem__(&self, idx: isize) -> PyResult<f32> {
        let len = 4isize;
        let norm = if idx < 0 { len + idx } else { idx };
        if norm < 0 || norm >= len {
            return Err(PyIndexError::new_err("Cl2 index out of range"));
        }
        Ok(self.inner.coords[norm as usize])
    }

    #[getter]
    fn e12(&self) -> f32 {
        self.inner.coords[3]
    }

    fn norm(&self) -> f32 {
        self.inner.norm()
    }

    fn norm_sq(&self) -> f32 {
        self.inner.norm_sq()
    }

    fn dual(&self) -> PyResult<Self> {
        self.inner
            .dual()
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("Dual not invertible"))
    }

    fn inverse(&self) -> PyResult<Self> {
        self.inner
            .inverse()
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("Multivector is singular"))
    }
}

// ============================================================================
// PYO3 WRAPPER: 3D PROJECTIVE GEOMETRIC ALGEBRA (PGA)
// ============================================================================

#[pyclass(name = "PgaMotor")]
#[derive(Clone)]
pub struct PyPgaMotor {
    pub inner: Box<PgaMotor>,
}

#[pymethods]
impl PyPgaMotor {
    #[new]
    #[pyo3(signature = (translation = [0.0, 0.0, 0.0], rotation_axis = [0.0, 0.0, 1.0], angle_rad = 0.0))]
    fn new(translation: [f32; 3], rotation_axis: [f32; 3], angle_rad: f32) -> Self {
        Self {
            inner: Box::new(PgaMotor::from_rotation_translation(
                &rotation_axis,
                angle_rad,
                &translation,
            )),
        }
    }

    #[getter]
    fn s(&self) -> f32 {
        self.inner.s
    }
    #[getter]
    fn b23(&self) -> f32 {
        self.inner.b23
    }
    #[getter]
    fn b31(&self) -> f32 {
        self.inner.b31
    }
    #[getter]
    fn b12(&self) -> f32 {
        self.inner.b12
    }
    #[getter]
    fn d01(&self) -> f32 {
        self.inner.d01
    }
    #[getter]
    fn d02(&self) -> f32 {
        self.inner.d02
    }
    #[getter]
    fn d03(&self) -> f32 {
        self.inner.d03
    }
    #[getter]
    fn e0123(&self) -> f32 {
        self.inner.e0123
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_list(&self) -> Vec<f32> {
        vec![
            self.inner.s,
            self.inner.b23,
            self.inner.b31,
            self.inner.b12,
            self.inner.d01,
            self.inner.d02,
            self.inner.d03,
            self.inner.e0123,
        ]
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_buffer(&self) -> VersorBufferF32 {
        VersorBufferF32::from_vec_1d(self.to_list())
    }

    fn transform_point(&self, pt: [f32; 3]) -> [f32; 3] {
        let p = PgaPoint::new(pt[0], pt[1], pt[2]);
        let transformed = self.inner.transform_point(&p);
        transformed.to_euclidean()
    }

    fn sclerp(&self, other: &PyPgaMotor, t: f32) -> Self {
        Self {
            inner: Box::new(self.inner.sclerp(&other.inner, t)),
        }
    }

    fn inverse(&self) -> Self {
        Self {
            inner: Box::new(self.inner.inverse()),
        }
    }

    /// Extracts the invariant Plücker screw axis parameters `(point_on_axis, direction, angle_rad, pitch)`
    /// of this SE(3) motor according to Chasles' Theorem.
    fn screw_parameters(&self) -> ([f32; 3], [f32; 3], f32, f32) {
        let m = self.inner.normalized();
        let half_ang = m.s.clamp(-1.0, 1.0).acos();
        let sin_half = half_ang.sin();
        if sin_half.abs() < 1e-6 {
            let tx = -2.0 * m.d01;
            let ty = -2.0 * m.d02;
            let tz = -2.0 * m.d03;
            let d = (tx * tx + ty * ty + tz * tz).sqrt();
            let dir = if d > 1e-6 {
                [tx / d, ty / d, tz / d]
            } else {
                [0.0, 0.0, 1.0]
            };
            ([0.0, 0.0, 0.0], dir, 0.0, d)
        } else {
            let inv_sin = 1.0 / sin_half;
            let vx = -m.b23 * inv_sin;
            let vy = -m.b31 * inv_sin;
            let vz = -m.b12 * inv_sin;
            let half_d = m.e0123 * inv_sin;
            let cos_half = m.s;
            let mx = (-m.d01 - half_d * cos_half * vx) * inv_sin;
            let my = (-m.d02 - half_d * cos_half * vy) * inv_sin;
            let mz = (-m.d03 - half_d * cos_half * vz) * inv_sin;
            let line = PgaLine::from_direction_and_moment([vx, vy, vz], [mx, my, mz]);
            let p0 = line.closest_point_to_origin().to_euclidean();
            let angle = 2.0 * half_ang;
            let pitch = if angle.abs() > 1e-6 {
                (2.0 * half_d) / angle
            } else {
                0.0
            };
            (p0, [vx, vy, vz], angle, pitch)
        }
    }

    fn __mul__(&self, other: &PyPgaMotor) -> Self {
        Self {
            inner: Box::new(self.inner.geometric_product(&other.inner)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PgaMotor(s={:.4}, b23={:.4}, b31={:.4}, b12={:.4}, d01={:.4}, d02={:.4}, d03={:.4}, e0123={:.4})",
            self.inner.s,
            self.inner.b23,
            self.inner.b31,
            self.inner.b12,
            self.inner.d01,
            self.inner.d02,
            self.inner.d03,
            self.inner.e0123
        )
    }
}

// ============================================================================
// PYO3 WRAPPER: 3D CONFORMAL GEOMETRIC ALGEBRA (CGA)
// ============================================================================

#[pyclass(name = "CgaPoint")]
#[derive(Clone)]
pub struct PyCgaPoint {
    pub inner: Box<CgaPoint>,
}

#[pymethods]
impl PyCgaPoint {
    #[new]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: Box::new(CgaPoint::point(x, y, z)),
        }
    }

    fn distance(&self, other: &PyCgaPoint) -> f32 {
        self.inner.distance(&other.inner)
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_euclidean(&self) -> [f32; 3] {
        self.inner.to_euclidean()
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_list(&self) -> Vec<f32> {
        vec![
            self.inner.x,
            self.inner.y,
            self.inner.z,
            self.inner.no,
            self.inner.ni,
        ]
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_buffer(&self) -> VersorBufferF32 {
        VersorBufferF32::from_vec_1d(self.to_list())
    }

    fn __repr__(&self) -> String {
        let [x, y, z] = self.inner.to_euclidean();
        format!("CgaPoint(x={:.3}, y={:.3}, z={:.3})", x, y, z)
    }
}

#[pyclass(name = "CgaSphere")]
#[derive(Clone)]
pub struct PyCgaSphere {
    pub inner: Box<CgaSphere>,
}

#[pymethods]
impl PyCgaSphere {
    #[new]
    fn new(cx: f32, cy: f32, cz: f32, radius: f32) -> PyResult<Self> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(PyValueError::new_err(format!(
                "Sphere radius must be finite and non-negative, got {}",
                radius
            )));
        }
        Ok(Self {
            inner: Box::new(CgaSphere::sphere(cx, cy, cz, radius)),
        })
    }

    #[getter]
    fn radius(&self) -> Option<f32> {
        self.inner.sphere_radius()
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_list(&self) -> Vec<f32> {
        vec![
            self.inner.x,
            self.inner.y,
            self.inner.z,
            self.inner.no,
            self.inner.ni,
        ]
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_buffer(&self) -> VersorBufferF32 {
        VersorBufferF32::from_vec_1d(self.to_list())
    }

    fn invert_in_sphere(&self, pt: &PyCgaPoint) -> PyCgaPoint {
        PyCgaPoint {
            inner: Box::new(pt.inner.invert_in_sphere(&self.inner)),
        }
    }

    fn __repr__(&self) -> String {
        let [cx, cy, cz] = self.inner.to_euclidean();
        let r = self.inner.sphere_radius().unwrap_or(0.0);
        format!(
            "CgaSphere(center=[{:.3}, {:.3}, {:.3}], radius={:.3})",
            cx, cy, cz, r
        )
    }
}

#[pyclass(name = "CgaCircle")]
#[derive(Clone)]
pub struct PyCgaCircle {
    pub inner: Box<CgaCircle>,
}

#[pymethods]
impl PyCgaCircle {
    #[new]
    fn new(center: [f32; 3], normal: [f32; 3], radius: f32) -> Self {
        Self {
            inner: Box::new(CgaCircle::new(center, normal, radius)),
        }
    }

    #[classmethod]
    fn from_sphere_plane(
        _cls: &Bound<'_, PyType>,
        sphere: &PyCgaSphere,
        plane_normal: [f32; 3],
        plane_d: f32,
    ) -> PyResult<Self> {
        let plane =
            crate::cga::CgaPlane::plane(plane_normal[0], plane_normal[1], plane_normal[2], plane_d);
        CgaCircle::from_sphere_plane_intersection(&sphere.inner, &plane)
            .map(|inner| Self {
                inner: Box::new(inner),
            })
            .ok_or_else(|| {
                PyValueError::new_err("Sphere and Plane do not intersect in a real circle")
            })
    }

    #[getter]
    fn center(&self) -> [f32; 3] {
        self.inner.center
    }
    #[getter]
    fn normal(&self) -> [f32; 3] {
        self.inner.normal
    }
    #[getter]
    fn radius(&self) -> f32 {
        self.inner.radius
    }
    #[getter]
    fn area(&self) -> f32 {
        self.inner.area()
    }
    #[getter]
    fn circumference(&self) -> f32 {
        self.inner.circumference()
    }

    fn __repr__(&self) -> String {
        format!(
            "CgaCircle(center={:?}, normal={:?}, radius={:.3})",
            self.inner.center, self.inner.normal, self.inner.radius
        )
    }
}

fn validate_conformal_axis(axis: [f32; 3]) -> PyResult<()> {
    let norm_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    if !norm_sq.is_finite() || norm_sq <= 1e-12 {
        return Err(PyValueError::new_err(format!(
            "Rotation axis must be finite and non-zero, got {:?}",
            axis
        )));
    }
    Ok(())
}

fn validate_dilation_factor(dilation_factor: f32) -> PyResult<()> {
    if !dilation_factor.is_finite() || dilation_factor <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "dilation_factor must be finite and strictly positive, got {}",
            dilation_factor
        )));
    }
    Ok(())
}

#[pyclass(name = "CgaConformalMotor")]
#[derive(Clone, Copy)]
pub struct PyCgaConformalMotor {
    pub inner: CgaConformalMotor,
}

#[pymethods]
impl PyCgaConformalMotor {
    #[new]
    #[pyo3(signature = (translation = [0.0, 0.0, 0.0], rotation_axis = [0.0, 0.0, 1.0], rotation_angle_rad = 0.0, dilation_factor = 1.0))]
    fn new(
        translation: [f32; 3],
        rotation_axis: [f32; 3],
        rotation_angle_rad: f32,
        dilation_factor: f32,
    ) -> PyResult<Self> {
        validate_conformal_axis(rotation_axis)?;
        validate_dilation_factor(dilation_factor)?;
        Ok(Self {
            inner: CgaConformalMotor::new(
                rotation_axis,
                rotation_angle_rad,
                translation,
                dilation_factor,
            ),
        })
    }

    #[classmethod]
    fn from_translation(_cls: &Bound<'_, PyType>, dx: f32, dy: f32, dz: f32) -> Self {
        Self {
            inner: CgaConformalMotor::from_translation(dx, dy, dz),
        }
    }

    #[classmethod]
    fn from_rotation(_cls: &Bound<'_, PyType>, axis: [f32; 3], angle_rad: f32) -> PyResult<Self> {
        validate_conformal_axis(axis)?;
        Ok(Self {
            inner: CgaConformalMotor::from_rotation(axis, angle_rad),
        })
    }

    #[classmethod]
    fn from_dilation(_cls: &Bound<'_, PyType>, lambda: f32) -> PyResult<Self> {
        validate_dilation_factor(lambda)?;
        Ok(Self {
            inner: CgaConformalMotor::from_dilation(lambda),
        })
    }

    fn transform_point(&self, pt: &PyCgaPoint) -> PyCgaPoint {
        PyCgaPoint {
            inner: Box::new(self.inner.transform_point(&pt.inner)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CgaConformalMotor(trans={:?}, rot_axis={:?}, theta={:.3}, dilation={:.3})",
            self.inner.translation,
            self.inner.rotation_axis,
            self.inner.rotation_angle_rad,
            self.inner.dilation_factor
        )
    }
}

// ============================================================================
// PYO3 WRAPPER: 4D SPACETIME ALGEBRA (STA) & LOUNESTO SPINORS
// ============================================================================

#[pyclass(name = "StaVector")]
#[derive(Clone, Copy)]
pub struct PyStaVector {
    pub inner: StaVector,
}

#[pymethods]
impl PyStaVector {
    #[new]
    fn new(ct: f32, x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: StaVector::new(ct, x, y, z),
        }
    }

    #[getter]
    fn ct(&self) -> f32 {
        self.inner.ct
    }
    #[getter]
    fn x(&self) -> f32 {
        self.inner.x
    }
    #[getter]
    fn y(&self) -> f32 {
        self.inner.y
    }
    #[getter]
    fn z(&self) -> f32 {
        self.inner.z
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_list(&self) -> Vec<f32> {
        vec![self.inner.ct, self.inner.x, self.inner.y, self.inner.z]
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_buffer(&self) -> VersorBufferF32 {
        VersorBufferF32::from_vec_1d(self.to_list())
    }

    fn interval_sq(&self) -> f32 {
        self.inner.interval_sq()
    }
    fn is_timelike(&self) -> bool {
        self.inner.is_timelike()
    }
    fn is_spacelike(&self) -> bool {
        self.inner.is_spacelike()
    }
    fn is_lightlike(&self) -> bool {
        self.inner.is_lightlike()
    }
    fn proper_time(&self) -> Option<f32> {
        self.inner.proper_time()
    }

    fn __repr__(&self) -> String {
        format!(
            "StaVector(ct={:.3}, x={:.3}, y={:.3}, z={:.3})",
            self.inner.ct, self.inner.x, self.inner.y, self.inner.z
        )
    }
}

#[pyclass(name = "StaSpinor")]
#[derive(Clone)]
pub struct PyStaSpinor {
    pub inner: Box<StaSpinor>,
}

#[pymethods]
impl PyStaSpinor {
    #[new]
    #[pyo3(signature = (scalar = 0.0, ex = 0.0, ey = 0.0, ez = 0.0, bx = 0.0, by = 0.0, bz = 0.0, pseudoscalar = 0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        scalar: f32,
        ex: f32,
        ey: f32,
        ez: f32,
        bx: f32,
        by: f32,
        bz: f32,
        pseudoscalar: f32,
    ) -> Self {
        Self {
            inner: Box::new(StaSpinor::new(scalar, ex, ey, ez, bx, by, bz, pseudoscalar)),
        }
    }

    #[pyo3(signature = (tol = None))]
    fn lounesto_class(&self, tol: Option<f32>) -> String {
        let t = tol.unwrap_or(1e-4);
        match self.inner.lounesto_class(t) {
            LounestoClass::Class1DiracGeneral => "Class1_DiracGeneral".to_string(),
            LounestoClass::Class2DiracElectron => "Class2_DiracElectron".to_string(),
            LounestoClass::Class3DiracPseudoscalar => "Class3_DiracPseudoscalar".to_string(),
            LounestoClass::Class4FlagDipole => "Class4_FlagDipole".to_string(),
            LounestoClass::Class5MajoranaFlagPole => "Class5_MajoranaFlagPole".to_string(),
            LounestoClass::Class6WeylDipole => "Class6_WeylDipole".to_string(),
        }
    }

    fn bilinear_covariants<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let cov = self.inner.bilinear_covariants();
        let dict = PyDict::new(py);
        dict.set_item("sigma", cov.sigma)?;
        dict.set_item("omega", cov.omega)?;
        dict.set_item("j", [cov.j.ct, cov.j.x, cov.j.y, cov.j.z])?;
        dict.set_item(
            "s",
            [cov.s.ex, cov.s.ey, cov.s.ez, cov.s.bx, cov.s.by, cov.s.bz],
        )?;
        dict.set_item("k", [cov.k.ct, cov.k.x, cov.k.y, cov.k.z])?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "StaSpinor(scalar={:.3}, ex={:.3}, ey={:.3}, ez={:.3}, bx={:.3}, by={:.3}, bz={:.3}, p={:.3})",
            self.inner.scalar,
            self.inner.ex,
            self.inner.ey,
            self.inner.ez,
            self.inner.bx,
            self.inner.by,
            self.inner.bz,
            self.inner.pseudoscalar
        )
    }
}

// ============================================================================
// PYO3 WRAPPER: VERSORS & CLIFFORD 3-BIT QUANTIZATION
// ============================================================================

/// Python wrapper for Dyadic Walsh-Hadamard Versor with Rademacher scrambling.
#[pyclass(name = "DyadicWalshVersor")]
pub struct PyDyadicWalshVersor {
    inner: DyadicWalshVersor,
}

#[pymethods]
impl PyDyadicWalshVersor {
    #[new]
    #[pyo3(signature = (dim, seed = 0x5645_5253_4F52_4331))]
    fn new(dim: usize, seed: u64) -> PyResult<Self> {
        if dim == 0 || !dim.is_multiple_of(64) {
            return Err(PyValueError::new_err(format!(
                "Dimension {} must be a positive multiple of 64",
                dim
            )));
        }
        Ok(Self {
            inner: DyadicWalshVersor::new(dim, seed),
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dimension()
    }

    /// Applies isotropic randomized Hadamard rotation to a 1D float32 buffer in `O(N log N)`.
    #[pyo3(signature = (x))]
    fn apply(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<VersorBufferF32> {
        let mut buf = extract_f32_1d(py, x)?;
        if buf.len() != self.inner.dimension() {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.inner.dimension(),
                buf.len()
            )));
        }

        self.inner.apply_in_place(&mut buf);
        Ok(VersorBufferF32::from_vec_1d(buf))
    }

    /// Inverts the randomized Hadamard rotation.
    fn invert(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<VersorBufferF32> {
        let mut buf = extract_f32_1d(py, x)?;
        if buf.len() != self.inner.dimension() {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.inner.dimension(),
                buf.len()
            )));
        }

        self.inner.invert_in_place(&mut buf);
        Ok(VersorBufferF32::from_vec_1d(buf))
    }
}

/// Python wrapper for Clifford 3-bit CacheLine64 Quantized Tensor with cached rotation versor.
#[pyclass(name = "Clifford3BitTensor")]
pub struct PyClifford3BitTensor {
    inner: Clifford3BitTensor,
    versor: DyadicWalshVersor,
    seed: u64,
}

#[pymethods]
impl PyClifford3BitTensor {
    /// Compiles a 2D float32 buffer (PEP 3118 or nested sequence) into Clifford 3-bit representation.
    #[classmethod]
    #[pyo3(signature = (weights, seed = 0x5645_5253_4F52_4331))]
    fn from_buffer(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        weights: &Bound<'_, PyAny>,
        seed: u64,
    ) -> PyResult<Self> {
        let (flat, rows, cols) = extract_f32_2d(py, weights)?;
        if cols == 0 || !cols.is_multiple_of(128) {
            return Err(PyValueError::new_err(format!(
                "Column count {} must be a positive multiple of 128 (Clifford3BitCacheLine64)",
                cols
            )));
        }

        let versor = DyadicWalshVersor::new(cols, seed);
        let tensor = Clifford3BitTensor::compile_aot(&flat, rows, cols, &versor);

        Ok(Self {
            inner: tensor,
            versor,
            seed,
        })
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }
    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    #[getter]
    fn bytes_per_row(&self) -> usize {
        (self.inner.cols / 128) * std::mem::size_of::<Clifford3BitCacheLine64>()
    }

    #[getter]
    fn effective_bpw(&self) -> f32 {
        let total_bits =
            self.inner.cache_lines.len() * std::mem::size_of::<Clifford3BitCacheLine64>() * 8;
        let total_weights = self.inner.rows * self.inner.cols;
        if total_weights > 0 {
            total_bits as f32 / total_weights as f32
        } else {
            0.0
        }
    }

    #[pyo3(signature = (x))]
    fn matvec(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<VersorBufferF32> {
        let slice = extract_f32_1d(py, x)?;
        if slice.len() != self.inner.cols {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: matrix has {} cols, vector has {}",
                self.inner.cols,
                slice.len()
            )));
        }

        let mut out = vec![0.0f32; self.inner.rows];
        self.inner.matvec(&slice, &mut out, &self.versor);
        Ok(VersorBufferF32::from_vec_1d(out))
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = Clifford3BitCacheLine64::slice_to_le_bytes(&self.inner.cache_lines);
        PyBytes::new(py, &bytes)
    }
}

/// Returns the 240 minimal roots of the E8 lattice as a `(240, 8)` `VersorBufferF32`.
#[pyfunction]
pub fn get_e8_roots() -> PyResult<VersorBufferF32> {
    let roots = E8Lattice::generate_240_roots();
    let mut data = Vec::with_capacity(240 * 8);
    for r in &roots {
        data.extend_from_slice(r);
    }
    VersorBufferF32::from_vec_2d(data, 240, 8)
}

/// Returns the 8 Lloyd-Max 3-bit centroids (`+` and `-` levels) as an `(8,)` `VersorBufferF32`.
#[pyfunction]
pub fn get_lloyd_max_3bit_centroids() -> VersorBufferF32 {
    let centroids = vec![
        LLOYD_MAX_3BIT_CENTROIDS[0],
        LLOYD_MAX_3BIT_CENTROIDS[1],
        LLOYD_MAX_3BIT_CENTROIDS[2],
        LLOYD_MAX_3BIT_CENTROIDS[3],
        -LLOYD_MAX_3BIT_CENTROIDS[0],
        -LLOYD_MAX_3BIT_CENTROIDS[1],
        -LLOYD_MAX_3BIT_CENTROIDS[2],
        -LLOYD_MAX_3BIT_CENTROIDS[3],
    ];
    VersorBufferF32::from_vec_1d(centroids)
}

/// Fast Walsh-Hadamard Transform directly on a 1D float32 buffer in `O(N log N)`.
#[pyfunction]
pub fn fwht(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<VersorBufferF32> {
    let mut data = extract_f32_1d(py, x)?;
    if !data.len().is_power_of_two() {
        return Err(PyValueError::new_err(format!(
            "FWHT requires length to be a power of 2, got {}",
            data.len()
        )));
    }
    DyadicWalshVersor::fwht_in_place(&mut data);
    Ok(VersorBufferF32::from_vec_1d(data))
}

/// Python wrapper for Conformal Hyperbolic Poincaré Ball Point (B^d).
#[pyclass(name = "PoincarePoint")]
#[derive(Clone)]
pub struct PyPoincarePoint {
    pub inner: PoincarePoint,
}

#[pymethods]
impl PyPoincarePoint {
    #[new]
    fn new(coords: Vec<f32>) -> Self {
        Self {
            inner: PoincarePoint::new(coords),
        }
    }

    #[getter]
    fn coords(&self) -> Vec<f32> {
        self.inner.coords.clone()
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_list(&self) -> Vec<f32> {
        self.inner.coords.clone()
    }

    fn tolist(&self) -> Vec<f32> {
        self.inner.coords.clone()
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_buffer(&self) -> VersorBufferF32 {
        VersorBufferF32::from_vec_1d(self.inner.coords.clone())
    }

    #[getter]
    fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        build_array_interface(py, &self.inner.coords, &[self.inner.coords.len()])
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let borrow = slf.borrow();
        let ptr = borrow.inner.coords.as_ptr();
        let len = borrow.inner.coords.len();
        drop(borrow);
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        unsafe { fill_readonly_py_buffer(slf.into_any().unbind(), slice, &[len], view, flags) }
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe { release_py_buffer(view) };
    }

    fn norm(&self) -> f32 {
        self.inner.norm()
    }

    fn lambda_factor(&self) -> f32 {
        self.inner.lambda()
    }

    fn geodesic_distance(&self, other: &PyPoincarePoint) -> PyResult<f32> {
        if self.inner.dim() != other.inner.dim() {
            return Err(PyValueError::new_err(
                "Dimension mismatch in geodesic_distance",
            ));
        }
        Ok(self.inner.geodesic_distance(&other.inner))
    }

    fn mobius_add(&self, other: &PyPoincarePoint) -> PyResult<Self> {
        if self.inner.dim() != other.inner.dim() {
            return Err(PyValueError::new_err("Dimension mismatch in mobius_add"));
        }
        Ok(Self {
            inner: self.inner.mobius_add(&other.inner),
        })
    }

    fn to_klein(&self) -> Vec<f32> {
        self.inner.to_klein()
    }

    #[classmethod]
    fn from_klein(_cls: &Bound<'_, PyType>, k: Vec<f32>) -> Self {
        Self {
            inner: PoincarePoint::from_klein(&k),
        }
    }

    #[classmethod]
    fn einstein_midpoint(
        _cls: &Bound<'_, PyType>,
        points: Vec<PyPoincarePoint>,
        weights: Vec<f32>,
    ) -> PyResult<Self> {
        if points.len() != weights.len() {
            return Err(PyValueError::new_err("Points and weights length mismatch"));
        }
        let inner_pts: Vec<PoincarePoint> = points.into_iter().map(|p| p.inner).collect();
        Ok(Self {
            inner: PoincarePoint::einstein_midpoint(&inner_pts, &weights),
        })
    }

    /// Samples `num_steps` points along the exact hyperbolic geodesic connecting `self` and `other`
    /// via projective interpolation in the Beltrami-Klein model mapped back to the conformal Poincaré ball.
    fn geodesic_arc(&self, other: &PyPoincarePoint, num_steps: usize) -> PyResult<Vec<Vec<f32>>> {
        if self.inner.dim() != other.inner.dim() {
            return Err(PyValueError::new_err("Dimension mismatch in geodesic_arc"));
        }
        let k0 = self.inner.to_klein();
        let k1 = other.inner.to_klein();
        let steps = num_steps.max(2);
        let mut arc = Vec::with_capacity(steps);
        for i in 0..steps {
            let t = (i as f32) / ((steps - 1) as f32);
            let kt: Vec<f32> = k0
                .iter()
                .zip(k1.iter())
                .map(|(&a, &b)| a * (1.0 - t) + b * t)
                .collect();
            let pt = PoincarePoint::from_klein(&kt);
            arc.push(pt.coords);
        }
        Ok(arc)
    }

    fn __repr__(&self) -> String {
        format!("PoincarePoint(coords={:?})", self.inner.coords)
    }
}

// ============================================================================
// PYO3 WRAPPERS: PGA GEOMETRIC ENTITIES (POINT, PLANE, LINE)
// ============================================================================

/// Python wrapper for 3D PGA point (tri-vector).
#[pyclass(name = "PgaPoint")]
#[derive(Clone, Copy)]
pub struct PyPgaPoint {
    pub inner: PgaPoint,
    coords_xyz: [f32; 3],
}

impl PyPgaPoint {
    fn from_inner(inner: PgaPoint) -> Self {
        let [x, y, z] = inner.to_euclidean();
        Self {
            inner,
            coords_xyz: [x, y, z],
        }
    }
}

#[pymethods]
impl PyPgaPoint {
    #[new]
    #[pyo3(signature = (x = 0.0, y = 0.0, z = 0.0))]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self::from_inner(PgaPoint::new(x, y, z))
    }

    #[classmethod]
    fn origin(_cls: &Bound<'_, PyType>) -> Self {
        Self::from_inner(PgaPoint::origin())
    }

    #[getter]
    fn x(&self) -> f32 {
        self.inner.x
    }
    #[getter]
    fn y(&self) -> f32 {
        self.inner.y
    }
    #[getter]
    fn z(&self) -> f32 {
        self.inner.z
    }
    #[getter]
    fn w(&self) -> f32 {
        self.inner.w
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_tuple(&self) -> (f32, f32, f32) {
        (self.inner.x, self.inner.y, self.inner.z)
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_list(&self) -> Vec<f32> {
        self.coords_xyz.to_vec()
    }

    fn tolist(&self) -> Vec<f32> {
        self.coords_xyz.to_vec()
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_buffer(&self) -> VersorBufferF32 {
        VersorBufferF32::from_vec_1d(self.coords_xyz.to_vec())
    }

    #[getter]
    fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        build_array_interface(py, &self.coords_xyz, &[3])
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let borrow = slf.borrow();
        let ptr = borrow.coords_xyz.as_ptr();
        drop(borrow);
        let slice = unsafe { std::slice::from_raw_parts(ptr, 3) };
        unsafe { fill_readonly_py_buffer(slf.into_any().unbind(), slice, &[3], view, flags) }
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe { release_py_buffer(view) };
    }

    fn __repr__(&self) -> String {
        format!(
            "PgaPoint(x={:.3}, y={:.3}, z={:.3})",
            self.inner.x, self.inner.y, self.inner.z
        )
    }
}

/// Python wrapper for 3D PGA plane (1-vector: `a*e1 + b*e2 + c*e3 + d*e0`).
#[pyclass(name = "PgaPlane")]
#[derive(Clone, Copy)]
pub struct PyPgaPlane {
    pub inner: PgaPlane,
}

#[pymethods]
impl PyPgaPlane {
    #[new]
    fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self {
            inner: PgaPlane::new(a, b, c, d),
        }
    }

    #[getter]
    fn a(&self) -> f32 {
        self.inner.a
    }
    #[getter]
    fn b(&self) -> f32 {
        self.inner.b
    }
    #[getter]
    fn c(&self) -> f32 {
        self.inner.c
    }
    #[getter]
    fn d(&self) -> f32 {
        self.inner.d
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_list(&self) -> Vec<f32> {
        vec![self.inner.a, self.inner.b, self.inner.c, self.inner.d]
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_buffer(&self) -> VersorBufferF32 {
        VersorBufferF32::from_vec_1d(self.to_list())
    }

    fn distance_to_point(&self, p: &PyPgaPoint) -> f32 {
        self.inner.distance_to_point(&p.inner)
    }

    fn reflect_point(&self, p: &PyPgaPoint) -> PyPgaPoint {
        PyPgaPoint::from_inner(self.inner.reflect_point(&p.inner))
    }

    fn __repr__(&self) -> String {
        format!(
            "PgaPlane({:.3}*x + {:.3}*y + {:.3}*z + {:.3} = 0)",
            self.inner.a, self.inner.b, self.inner.c, self.inner.d
        )
    }
}

/// Python wrapper for 3D PGA line (bi-vector / Plücker coordinates).
#[pyclass(name = "PgaLine")]
#[derive(Clone)]
pub struct PyPgaLine {
    pub inner: Box<PgaLine>,
}

#[pymethods]
impl PyPgaLine {
    #[new]
    fn new(vx: f32, vy: f32, vz: f32, mx: f32, my: f32, mz: f32) -> Self {
        Self {
            inner: Box::new(PgaLine::new(vx, vy, vz, mx, my, mz)),
        }
    }

    #[classmethod]
    fn from_points(_cls: &Bound<'_, PyType>, p1: &PyPgaPoint, p2: &PyPgaPoint) -> Self {
        Self {
            inner: Box::new(p1.inner.join(&p2.inner)),
        }
    }

    fn distance_to_point(&self, p: &PyPgaPoint) -> f32 {
        self.inner.distance_to_point(&p.inner)
    }

    fn project_point(&self, p: &PyPgaPoint) -> PyPgaPoint {
        PyPgaPoint::from_inner(self.inner.project_point(&p.inner))
    }

    fn __repr__(&self) -> String {
        format!(
            "PgaLine(dir=[{:.3}, {:.3}, {:.3}], mom=[{:.3}, {:.3}, {:.3}])",
            self.inner.vx,
            self.inner.vy,
            self.inner.vz,
            self.inner.mx,
            self.inner.my,
            self.inner.mz
        )
    }
}

// ============================================================================
// PYO3 WRAPPERS: QUATERNIONS & DUAL QUATERNIONS
// ============================================================================

/// Python wrapper for Hamilton Quaternions (H).
#[pyclass(name = "Quaternion")]
#[derive(Clone, Copy)]
pub struct PyQuaternion {
    pub inner: Quaternion,
}

#[pymethods]
impl PyQuaternion {
    #[new]
    #[pyo3(signature = (w = 1.0, x = 0.0, y = 0.0, z = 0.0))]
    fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: Quaternion::new(w, x, y, z),
        }
    }

    #[classmethod]
    fn from_axis_angle(_cls: &Bound<'_, PyType>, axis: [f32; 3], angle_rad: f32) -> Self {
        Self {
            inner: Quaternion::from_axis_angle(&axis, angle_rad),
        }
    }

    #[getter]
    fn w(&self) -> f32 {
        self.inner.w
    }
    #[getter]
    fn x(&self) -> f32 {
        self.inner.x
    }
    #[getter]
    fn y(&self) -> f32 {
        self.inner.y
    }
    #[getter]
    fn z(&self) -> f32 {
        self.inner.z
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_list(&self) -> Vec<f32> {
        vec![self.inner.w, self.inner.x, self.inner.y, self.inner.z]
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_buffer(&self) -> VersorBufferF32 {
        VersorBufferF32::from_vec_1d(self.to_list())
    }

    fn rotate_vector(&self, v: [f32; 3]) -> [f32; 3] {
        self.inner.rotate_vector3(&v)
    }

    fn slerp(&self, other: &PyQuaternion, t: f32) -> Self {
        Self {
            inner: Quaternion::slerp(self.inner, other.inner, t),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Quaternion(w={:.4}, x={:.4}, y={:.4}, z={:.4})",
            self.inner.w, self.inner.x, self.inner.y, self.inner.z
        )
    }
}

/// Python wrapper for Dual Quaternions (`H (x) D ~= Cl+(3, 0, 1)`).
#[pyclass(name = "DualQuaternion")]
#[derive(Clone)]
pub struct PyDualQuaternion {
    pub inner: Box<DualQuaternion>,
}

#[pymethods]
impl PyDualQuaternion {
    #[new]
    fn new(real: &PyQuaternion, dual: &PyQuaternion) -> Self {
        Self {
            inner: Box::new(DualQuaternion::new(real.inner, dual.inner)),
        }
    }

    #[classmethod]
    fn from_rotation_translation(
        _cls: &Bound<'_, PyType>,
        rot: &PyQuaternion,
        trans: [f32; 3],
    ) -> Self {
        Self {
            inner: Box::new(DualQuaternion::from_rotation_translation(
                &rot.inner, &trans,
            )),
        }
    }

    #[getter]
    fn real(&self) -> PyQuaternion {
        PyQuaternion {
            inner: self.inner.real,
        }
    }
    #[getter]
    fn dual(&self) -> PyQuaternion {
        PyQuaternion {
            inner: self.inner.dual,
        }
    }

    fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        self.inner.transform_point(&p)
    }

    fn sclerp(&self, other: &PyDualQuaternion, t: f32) -> Self {
        Self {
            inner: Box::new(self.inner.sclerp(&other.inner, t)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DualQuaternion(real=({:.3}, {:.3}, {:.3}, {:.3}), dual=({:.3}, {:.3}, {:.3}, {:.3}))",
            self.inner.real.w,
            self.inner.real.x,
            self.inner.real.y,
            self.inner.real.z,
            self.inner.dual.w,
            self.inner.dual.x,
            self.inner.dual.y,
            self.inner.dual.z
        )
    }
}

// ============================================================================
// PYO3 WRAPPER: CAYLEY-KLEIN UNIFIED METRIC GEOMETRY
// ============================================================================

/// Python wrapper for continuous Cayley-Klein space parametrized by `tau`.
#[pyclass(name = "CayleyKleinSpace")]
#[derive(Clone, Copy)]
pub struct PyCayleyKleinSpace {
    pub inner: CayleyKleinSpace,
}

#[pymethods]
impl PyCayleyKleinSpace {
    #[new]
    fn new(tau: f32) -> Self {
        Self {
            inner: CayleyKleinSpace::new(tau),
        }
    }

    #[classmethod]
    fn hyperbolic(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: CayleyKleinSpace::hyperbolic(),
        }
    }
    #[classmethod]
    fn euclidean(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: CayleyKleinSpace::euclidean(),
        }
    }
    #[classmethod]
    fn elliptic(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: CayleyKleinSpace::elliptic(),
        }
    }

    #[getter]
    fn tau(&self) -> f32 {
        self.inner.tau
    }

    fn distance(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<f32> {
        let sx = extract_f32_1d(py, x)?;
        let sy = extract_f32_1d(py, y)?;
        if sx.len() != sy.len() {
            return Err(PyValueError::new_err(
                "Dimension mismatch in Cayley-Klein distance",
            ));
        }
        Ok(self.inner.geodesic_distance(&sx, &sy))
    }

    fn conformal_factor(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<f32> {
        let sx = extract_f32_1d(py, x)?;
        Ok(self.inner.conformal_factor(&sx))
    }

    fn __repr__(&self) -> String {
        format!("CayleyKleinSpace(tau={:.4})", self.inner.tau)
    }
}

// ============================================================================
// PYO3 FUNCTIONS: CONWAY-SLOANE E8 LATTICE PROJECTION
// ============================================================================

/// Projects an 8D continuous vector onto the closest point in the Gosset E8 lattice.
#[pyfunction]
pub fn project_to_e8(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<VersorBufferF32> {
    let slice = extract_f32_1d(py, x)?;
    if slice.len() != 8 {
        return Err(PyValueError::new_err(
            "E8 lattice projection requires an 8-dimensional vector",
        ));
    }
    let arr: [f32; 8] = slice.try_into().unwrap();
    let projected = E8Lattice::project_to_e8(&arr);
    Ok(VersorBufferF32::from_vec_1d(projected.to_vec()))
}

/// Projects an 8D continuous vector onto the closest point in the D8 checkerboard lattice.
#[pyfunction]
pub fn project_to_d8(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<VersorBufferI32> {
    let slice = extract_f32_1d(py, x)?;
    if slice.len() != 8 {
        return Err(PyValueError::new_err(
            "D8 lattice projection requires an 8-dimensional vector",
        ));
    }
    let arr: [f32; 8] = slice.try_into().unwrap();
    let projected = E8Lattice::project_to_d8(&arr);
    Ok(VersorBufferI32::from_vec_1d(projected.to_vec()))
}

// ============================================================================
// PYO3 WRAPPER: VULKAN GPU COMPUTE CONTEXT
// ============================================================================

#[cfg(feature = "vulkan")]
fn extract_points_3d(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<[f32; 3]>> {
    let (flat, rows, cols) = extract_f32_2d(py, obj)?;
    if cols != 3 {
        return Err(PyValueError::new_err(format!(
            "Expected [N, 3] float32 buffer, got shape [{}, {}]",
            rows, cols
        )));
    }
    Ok(flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
}

#[cfg(feature = "vulkan")]
fn extract_cl3_batch(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<[f32; 8]>> {
    let (flat, rows, cols) = extract_f32_2d(py, obj)?;
    if cols != 8 {
        return Err(PyValueError::new_err(format!(
            "Expected [N, 8] Cl3 batch buffer, got shape [{}, {}]",
            rows, cols
        )));
    }
    Ok(flat
        .chunks_exact(8)
        .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
        .collect())
}

#[cfg(feature = "vulkan")]
#[pyclass(name = "GpuComputeContext")]
pub struct PyGpuComputeContext {
    inner: std::sync::Arc<GpuComputeContext>,
}

#[cfg(feature = "vulkan")]
#[pymethods]
impl PyGpuComputeContext {
    #[classmethod]
    fn probe_devices(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let gpus = GpuComputeContext::probe_devices()
            .map_err(|e| PyValueError::new_err(format!("Vulkan probe failed: {}", e)))?;
        let mut out = Vec::with_capacity(gpus.len());
        for gpu in gpus {
            let dict = PyDict::new(py);
            dict.set_item("name", gpu.name)?;
            dict.set_item("is_discrete", gpu.is_discrete)?;
            dict.set_item("driver_version", gpu.driver_version)?;
            dict.set_item("api_version", gpu.api_version)?;
            out.push(dict.unbind());
        }
        Ok(out)
    }

    #[classmethod]
    fn init(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        let ctx = GpuComputeContext::init_compute()
            .map_err(|e| PyValueError::new_err(format!("Vulkan initialization failed: {}", e)))?;
        Ok(Self {
            inner: std::sync::Arc::new(ctx),
        })
    }

    fn dispatch_vec_add(
        &self,
        py: Python<'_>,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
    ) -> PyResult<VersorBufferF32> {
        let sa = extract_f32_1d(py, a)?;
        let sb = extract_f32_1d(py, b)?;
        let res = self
            .inner
            .dispatch_vec_add(&sa, &sb)
            .map_err(|e| PyValueError::new_err(format!("GPU vec_add failed: {}", e)))?;
        Ok(VersorBufferF32::from_vec_1d(res))
    }

    fn dispatch_silu_mul(
        &self,
        py: Python<'_>,
        gate: &Bound<'_, PyAny>,
        up: &Bound<'_, PyAny>,
    ) -> PyResult<VersorBufferF32> {
        let sg = extract_f32_1d(py, gate)?;
        let su = extract_f32_1d(py, up)?;
        let res = self
            .inner
            .dispatch_silu_mul(&sg, &su)
            .map_err(|e| PyValueError::new_err(format!("GPU silu_mul failed: {}", e)))?;
        Ok(VersorBufferF32::from_vec_1d(res))
    }

    fn dispatch_rmsnorm(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        gamma: &Bound<'_, PyAny>,
        eps: f32,
    ) -> PyResult<VersorBufferF32> {
        let sx = extract_f32_1d(py, x)?;
        let sg = extract_f32_1d(py, gamma)?;
        let res = self
            .inner
            .dispatch_rmsnorm(&sx, &sg, eps)
            .map_err(|e| PyValueError::new_err(format!("GPU rmsnorm failed: {}", e)))?;
        Ok(VersorBufferF32::from_vec_1d(res))
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_rope(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        pos: u32,
        n_heads: u32,
        head_dim: u32,
        rope_theta: f32,
        is_neox: bool,
    ) -> PyResult<VersorBufferF32> {
        let mut data = extract_f32_1d(py, q)?;
        self.inner
            .dispatch_rope(&mut data, pos, n_heads, head_dim, rope_theta, is_neox)
            .map_err(|e| PyValueError::new_err(format!("GPU rope failed: {}", e)))?;
        Ok(VersorBufferF32::from_vec_1d(data))
    }

    /// Dispatches a batch of `Cl(3, 0)` multivector operations (`op_code` `0..=12`) on GPU VRAM.
    #[pyo3(signature = (a, b = None, op_code = 0))]
    fn dispatch_cl3_batch(
        &self,
        py: Python<'_>,
        a: &Bound<'_, PyAny>,
        b: Option<&Bound<'_, PyAny>>,
        op_code: u32,
    ) -> PyResult<VersorBufferF32> {
        let a_pts = extract_cl3_batch(py, a)?;
        let b_pts = match b {
            Some(b_obj) => extract_cl3_batch(py, b_obj)?,
            None => Vec::new(),
        };
        let res = self
            .inner
            .dispatch_cl3_batch(&a_pts, &b_pts, op_code)
            .map_err(|e| PyValueError::new_err(format!("GPU cl3_batch failed: {}", e)))?;
        let rows = res.len();
        let mut flat = Vec::with_capacity(rows * 8);
        for mv in res {
            flat.extend_from_slice(&mv);
        }
        VersorBufferF32::from_vec_2d(flat, rows, 8)
    }

    /// Dispatches Poincare/Hyperbolic batch operations (`0=Mobius Add`, `1=Geodesic Distance`,
    /// `2=Spherical Inversion`, `3=Poincare to Klein`, `4=Klein to Poincare`, `5=Einstein Midpoint`).
    #[pyo3(signature = (u, v = None, op_code = 0, sphere_center = [0.0, 0.0, 0.0], sphere_radius = 1.0))]
    fn dispatch_poincare_batch(
        &self,
        py: Python<'_>,
        u: &Bound<'_, PyAny>,
        v: Option<&Bound<'_, PyAny>>,
        op_code: u32,
        sphere_center: [f32; 3],
        sphere_radius: f32,
    ) -> PyResult<VersorBufferF32> {
        let u_pts = extract_points_3d(py, u)?;
        let v_pts = match v {
            Some(v_obj) => extract_points_3d(py, v_obj)?,
            None => Vec::new(),
        };
        let res = self
            .inner
            .dispatch_poincare_batch(&u_pts, &v_pts, op_code, sphere_center, sphere_radius)
            .map_err(|e| PyValueError::new_err(format!("GPU poincare_batch failed: {}", e)))?;
        let rows = res.len();
        let mut flat = Vec::with_capacity(rows * 3);
        for pt in res {
            flat.extend_from_slice(&pt);
        }
        VersorBufferF32::from_vec_2d(flat, rows, 3)
    }

    /// Dispatches Poincare geodesic distances over `[N, 3]` batches and returns `[N]` `VersorBufferF32`.
    fn dispatch_poincare_distance(
        &self,
        py: Python<'_>,
        u: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<VersorBufferF32> {
        let u_pts = extract_points_3d(py, u)?;
        let v_pts = extract_points_3d(py, v)?;
        let res = self
            .inner
            .dispatch_poincare_distance(&u_pts, &v_pts)
            .map_err(|e| PyValueError::new_err(format!("GPU poincare_distance failed: {}", e)))?;
        Ok(VersorBufferF32::from_vec_1d(res))
    }

    /// Dispatches 3D PGA rigid motor transformation over an `[N, 3]` point batch in GPU VRAM.
    fn dispatch_pga_transform(
        &self,
        py: Python<'_>,
        points: &Bound<'_, PyAny>,
        motor: &PyPgaMotor,
    ) -> PyResult<VersorBufferF32> {
        let pts = extract_points_3d(py, points)?;
        let res = self
            .inner
            .dispatch_pga_transform(&pts, &motor.inner)
            .map_err(|e| PyValueError::new_err(format!("GPU pga_transform failed: {}", e)))?;
        let rows = res.len();
        let mut flat = Vec::with_capacity(rows * 3);
        for pt in res {
            flat.extend_from_slice(&pt);
        }
        VersorBufferF32::from_vec_2d(flat, rows, 3)
    }

    /// Dispatches 3D CGA conformal transformations over an `[N, 3]` batch in GPU VRAM.
    /// `op_type`: `0` = spherical inversion, `1` = dilation (`params[0] = lambda`), `2` = translation (`params = [dx, dy, dz]`).
    #[pyo3(signature = (points, op_type = 0, sphere_center = [0.0, 0.0, 0.0], sphere_radius = 1.0, params = [1.0, 0.0, 0.0]))]
    fn dispatch_cga_transform(
        &self,
        py: Python<'_>,
        points: &Bound<'_, PyAny>,
        op_type: u32,
        sphere_center: [f32; 3],
        sphere_radius: f32,
        params: [f32; 3],
    ) -> PyResult<VersorBufferF32> {
        let pts = extract_points_3d(py, points)?;
        let n_u32 = u32::try_from(pts.len())
            .map_err(|_| PyValueError::new_err("Point batch exceeds u32 dispatch limit"))?;
        let pc = match op_type {
            0 => {
                CgaTransformPushConstants::spherical_inversion(n_u32, sphere_center, sphere_radius)
            }
            1 => CgaTransformPushConstants::dilation(n_u32, params[0]),
            2 => CgaTransformPushConstants::translation(n_u32, params),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid CGA op_type {}: expected 0 (inversion), 1 (dilation), or 2 (translation)",
                    op_type
                )));
            }
        };
        let res = self
            .inner
            .dispatch_cga_transform(&pts, &pc)
            .map_err(|e| PyValueError::new_err(format!("GPU cga_transform failed: {}", e)))?;
        let rows = res.len();
        let mut flat = Vec::with_capacity(rows * 3);
        for pt in res {
            flat.extend_from_slice(&pt);
        }
        VersorBufferF32::from_vec_2d(flat, rows, 3)
    }
}

/// Native PyO3 module definition for `versorc_math`.
#[pymodule]
pub fn _versorc_math(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VersorBufferF32>()?;
    m.add_class::<VersorBufferI32>()?;
    m.add_class::<PyCl3>()?;
    m.add_class::<PyCl2>()?;
    m.add_class::<PyPgaMotor>()?;
    m.add_class::<PyPgaPoint>()?;
    m.add_class::<PyPgaPlane>()?;
    m.add_class::<PyPgaLine>()?;
    m.add_class::<PyQuaternion>()?;
    m.add_class::<PyDualQuaternion>()?;
    m.add_class::<PyCgaPoint>()?;
    m.add_class::<PyCgaSphere>()?;
    m.add_class::<PyCgaCircle>()?;
    m.add_class::<PyCgaConformalMotor>()?;
    m.add_class::<PyStaVector>()?;
    m.add_class::<PyStaSpinor>()?;
    m.add_class::<PyDyadicWalshVersor>()?;
    m.add_class::<PyClifford3BitTensor>()?;
    m.add_class::<PyPoincarePoint>()?;
    m.add_class::<PyCayleyKleinSpace>()?;
    m.add_function(wrap_pyfunction!(get_e8_roots, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_e8, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_d8, m)?)?;
    m.add_function(wrap_pyfunction!(get_lloyd_max_3bit_centroids, m)?)?;
    m.add_function(wrap_pyfunction!(fwht, m)?)?;

    #[cfg(feature = "vulkan")]
    m.add_class::<PyGpuComputeContext>()?;

    Ok(())
}
