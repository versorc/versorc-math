// ============================================================================
// VERSORC-MATH: PYO3 NATIVE BINDINGS & NUMPY BRIDGE
// License: Apache-2.0
// ============================================================================

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyType};
use numpy::{
    ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};

use crate::multivector::{Cl2, Cl3};
use crate::pga::{PgaMotor, PgaPoint};
use crate::cga::{CgaPoint, CgaSphere, CgaCircle, CgaConformalMotor};
use crate::sta::{StaVector, StaSpinor, LounestoClass};
use crate::versor::DyadicWalshVersor;
use crate::quantization::clifford_3bit::{
    Clifford3BitCacheLine64, Clifford3BitTensor, LLOYD_MAX_3BIT_CENTROIDS,
};
use crate::lattice::E8Lattice;
use crate::poincare::PoincarePoint;

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
                return Err(PyValueError::new_err(format!("Cl3 expects at most 8 coordinates, got {}", c.len())));
            }
            for (i, &v) in c.iter().enumerate() {
                arr[i] = v;
            }
        }
        Ok(Self { inner: Cl3 { coords: arr } })
    }

    #[classmethod]
    fn from_scalar(_cls: &Bound<'_, PyType>, s: f32) -> Self {
        Self { inner: Cl3::from_scalar(s) }
    }

    #[classmethod]
    fn vector(_cls: &Bound<'_, PyType>, x: f32, y: f32, z: f32) -> Self {
        Self { inner: Cl3::from_vector(&[x, y, z]) }
    }

    #[classmethod]
    fn from_vector(_cls: &Bound<'_, PyType>, x: f32, y: f32, z: f32) -> Self {
        Self { inner: Cl3::from_vector(&[x, y, z]) }
    }

    #[classmethod]
    fn bivector(_cls: &Bound<'_, PyType>, e12: f32, e13: f32, e23: f32) -> Self {
        let mut arr = [0.0f32; 8];
        arr[3] = e12;
        arr[5] = e13;
        arr[6] = e23;
        Self { inner: Cl3 { coords: arr } }
    }

    #[classmethod]
    fn e1(_cls: &Bound<'_, PyType>) -> Self { Self { inner: Cl3::basis(1) } }
    #[classmethod]
    fn e2(_cls: &Bound<'_, PyType>) -> Self { Self { inner: Cl3::basis(2) } }
    #[classmethod]
    fn e3(_cls: &Bound<'_, PyType>) -> Self { Self { inner: Cl3::basis(3) } }
    #[classmethod]
    #[allow(non_snake_case)]
    fn I(_cls: &Bound<'_, PyType>) -> Self { Self { inner: Cl3::pseudoscalar() } }

    #[classmethod]
    fn from_numpy(_cls: &Bound<'_, PyType>, arr: PyReadonlyArray1<'_, f32>) -> PyResult<Self> {
        let slice = arr.as_slice()?;
        if slice.len() != 8 {
            return Err(PyValueError::new_err(format!("Cl3 requires 8 coordinates, got {}", slice.len())));
        }
        let mut coords = [0.0f32; 8];
        coords.copy_from_slice(slice);
        Ok(Self { inner: Cl3 { coords } })
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.coords)
    }

    // Dunder operators
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl3>() {
            Ok(Self { inner: self.inner + other_cl.inner })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self { inner: self.inner + Cl3::from_scalar(s) })
        } else {
            Err(PyValueError::new_err("Addition only supported with Cl3 or float"))
        }
    }

    fn __radd__(&self, other: f32) -> Self {
        Self { inner: self.inner + Cl3::from_scalar(other) }
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl3>() {
            Ok(Self { inner: self.inner - other_cl.inner })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self { inner: self.inner - Cl3::from_scalar(s) })
        } else {
            Err(PyValueError::new_err("Subtraction only supported with Cl3 or float"))
        }
    }

    fn __rsub__(&self, other: f32) -> Self {
        Self { inner: Cl3::from_scalar(other) - self.inner }
    }

    fn __neg__(&self) -> Self {
        Self { inner: -self.inner }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl3>() {
            Ok(Self { inner: self.inner * other_cl.inner })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self { inner: self.inner * s })
        } else {
            Err(PyValueError::new_err("Multiplication only supported with Cl3 or float"))
        }
    }

    fn __rmul__(&self, scalar: f32) -> Self {
        Self { inner: self.inner * scalar }
    }

    fn __xor__(&self, other: &PyCl3) -> Self {
        Self { inner: self.inner.wedge(&other.inner) }
    }

    fn __or__(&self, other: &PyCl3) -> Self {
        Self { inner: self.inner.left_contraction(&other.inner) }
    }

    fn __invert__(&self) -> Self {
        Self { inner: self.inner.reverse() }
    }

    fn __repr__(&self) -> String {
        format!("Cl3({})", format_cl3(&self.inner.coords))
    }

    fn __str__(&self) -> String {
        format_cl3(&self.inner.coords)
    }

    // Methods
    fn grade(&self, k: usize) -> Self {
        Self { inner: self.inner.grade_projection(k) }
    }

    #[getter]
    fn scalar(&self) -> f32 {
        self.inner.scalar()
    }

    fn reverse(&self) -> Self { Self { inner: self.inner.reverse() } }
    fn involution(&self) -> Self { Self { inner: self.inner.grade_involution() } }
    fn conjugation(&self) -> Self { Self { inner: self.inner.conjugation() } }

    fn dual(&self) -> PyResult<Self> {
        self.inner.dual().map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("Dual not invertible"))
    }

    fn undual(&self) -> Self {
        Self { inner: self.inner.undual() }
    }

    fn commutator(&self, other: &PyCl3) -> Self {
        Self { inner: self.inner.commutator(&other.inner) }
    }

    fn anticommutator(&self, other: &PyCl3) -> Self {
        Self { inner: self.inner.anticommutator(&other.inner) }
    }

    fn regressive(&self, other: &PyCl3) -> PyResult<Self> {
        self.inner.regressive_product(&other.inner)
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("Regressive product requires invertible pseudoscalar"))
    }

    fn hestenes_dot(&self, other: &PyCl3) -> Self {
        Self { inner: self.inner.hestenes_inner_product(&other.inner) }
    }

    fn inverse(&self) -> PyResult<Self> {
        self.inner.inverse().map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("Multivector is singular (det == 0)"))
    }

    fn norm(&self) -> f32 { self.inner.norm() }
    fn norm_sq(&self) -> f32 { self.inner.norm_sq() }
    fn normalize(&self) -> Self { Self { inner: self.inner.normalize() } }

    fn sandwich(&self, x: &PyCl3) -> Self {
        Self { inner: self.inner.sandwich(&x.inner) }
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
                return Err(PyValueError::new_err(format!("Cl2 expects at most 4 coordinates, got {}", c.len())));
            }
            for (i, &v) in c.iter().enumerate() {
                arr[i] = v;
            }
        }
        Ok(Self { inner: Cl2 { coords: arr } })
    }

    #[classmethod]
    fn from_scalar(_cls: &Bound<'_, PyType>, s: f32) -> Self { Self { inner: Cl2::from_scalar(s) } }
    #[classmethod]
    fn vector(_cls: &Bound<'_, PyType>, x: f32, y: f32) -> Self { Self { inner: Cl2::from_vector(&[x, y]) } }
    #[classmethod]
    fn from_vector(_cls: &Bound<'_, PyType>, x: f32, y: f32) -> Self { Self { inner: Cl2::from_vector(&[x, y]) } }
    #[classmethod]
    fn e1(_cls: &Bound<'_, PyType>) -> Self { Self { inner: Cl2::basis(1) } }
    #[classmethod]
    fn e2(_cls: &Bound<'_, PyType>) -> Self { Self { inner: Cl2::basis(2) } }
    #[classmethod]
    #[allow(non_snake_case)]
    fn I(_cls: &Bound<'_, PyType>) -> Self { Self { inner: Cl2::pseudoscalar() } }

    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.coords)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl2>() {
            Ok(Self { inner: self.inner + other_cl.inner })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self { inner: self.inner + Cl2::from_scalar(s) })
        } else {
            Err(PyValueError::new_err("Addition only supported with Cl2 or float"))
        }
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl2>() {
            Ok(Self { inner: self.inner - other_cl.inner })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self { inner: self.inner - Cl2::from_scalar(s) })
        } else {
            Err(PyValueError::new_err("Subtraction only supported with Cl2 or float"))
        }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_cl) = other.extract::<PyCl2>() {
            Ok(Self { inner: self.inner * other_cl.inner })
        } else if let Ok(s) = other.extract::<f32>() {
            Ok(Self { inner: self.inner * s })
        } else {
            Err(PyValueError::new_err("Multiplication only supported with Cl2 or float"))
        }
    }

    fn __rmul__(&self, scalar: f32) -> Self { Self { inner: self.inner * scalar } }
    fn __xor__(&self, other: &PyCl2) -> Self { Self { inner: self.inner.wedge(&other.inner) } }
    fn __or__(&self, other: &PyCl2) -> Self { Self { inner: self.inner.left_contraction(&other.inner) } }
    fn __invert__(&self) -> Self { Self { inner: self.inner.reverse() } }
    fn __neg__(&self) -> Self { Self { inner: -self.inner } }

    fn __repr__(&self) -> String { format!("Cl2({})", format_cl2(&self.inner.coords)) }
    fn __str__(&self) -> String { format_cl2(&self.inner.coords) }

    #[getter]
    fn scalar(&self) -> f32 { self.inner.scalar() }
    fn norm(&self) -> f32 { self.inner.norm() }
    fn norm_sq(&self) -> f32 { self.inner.norm_sq() }

    fn dual(&self) -> PyResult<Self> {
        self.inner.dual().map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("Dual not invertible"))
    }

    fn inverse(&self) -> PyResult<Self> {
        self.inner.inverse().map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("Multivector is singular"))
    }
}

// ============================================================================
// PYO3 WRAPPER: 3D PROJECTIVE GEOMETRIC ALGEBRA (PGA)
// ============================================================================

#[pyclass(name = "PgaMotor")]
#[derive(Clone, Copy)]
pub struct PyPgaMotor {
    pub inner: PgaMotor,
}

#[pymethods]
impl PyPgaMotor {
    #[new]
    #[pyo3(signature = (translation = [0.0, 0.0, 0.0], rotation_axis = [0.0, 0.0, 1.0], angle_rad = 0.0))]
    fn new(translation: [f32; 3], rotation_axis: [f32; 3], angle_rad: f32) -> Self {
        Self {
            inner: PgaMotor::from_rotation_translation(&rotation_axis, angle_rad, &translation),
        }
    }

    #[getter] fn s(&self) -> f32 { self.inner.s }
    #[getter] fn b23(&self) -> f32 { self.inner.b23 }
    #[getter] fn b31(&self) -> f32 { self.inner.b31 }
    #[getter] fn b12(&self) -> f32 { self.inner.b12 }
    #[getter] fn d01(&self) -> f32 { self.inner.d01 }
    #[getter] fn d02(&self) -> f32 { self.inner.d02 }
    #[getter] fn d03(&self) -> f32 { self.inner.d03 }
    #[getter] fn e0123(&self) -> f32 { self.inner.e0123 }

    fn transform_point(&self, pt: [f32; 3]) -> [f32; 3] {
        let p = PgaPoint::new(pt[0], pt[1], pt[2]);
        let transformed = self.inner.transform_point(&p);
        transformed.to_euclidean()
    }

    fn sclerp(&self, other: &PyPgaMotor, t: f32) -> Self {
        Self { inner: self.inner.sclerp(&other.inner, t) }
    }

    fn inverse(&self) -> Self {
        Self { inner: self.inner.inverse() }
    }

    /// Extracts the invariant Plucker screw axis parameters (point_on_axis, direction, angle_rad, pitch)
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
            let dir = if d > 1e-6 { [tx / d, ty / d, tz / d] } else { [0.0, 0.0, 1.0] };
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
            let line = crate::pga::PgaLine::from_direction_and_moment([vx, vy, vz], [mx, my, mz]);
            let p0 = line.closest_point_to_origin().to_euclidean();
            let angle = 2.0 * half_ang;
            let pitch = if angle.abs() > 1e-6 { (2.0 * half_d) / angle } else { 0.0 };
            (p0, [vx, vy, vz], angle, pitch)
        }
    }

    fn __mul__(&self, other: &PyPgaMotor) -> Self {
        Self { inner: self.inner.geometric_product(&other.inner) }
    }

    fn __repr__(&self) -> String {
        format!(
            "PgaMotor(s={:.4}, b23={:.4}, b31={:.4}, b12={:.4}, d01={:.4}, d02={:.4}, d03={:.4}, e0123={:.4})",
            self.inner.s, self.inner.b23, self.inner.b31, self.inner.b12,
            self.inner.d01, self.inner.d02, self.inner.d03, self.inner.e0123
        )
    }
}

// ============================================================================
// PYO3 WRAPPER: 3D CONFORMAL GEOMETRIC ALGEBRA (CGA)
// ============================================================================

#[pyclass(name = "CgaPoint")]
#[derive(Clone, Copy)]
pub struct PyCgaPoint {
    pub inner: CgaPoint,
}

#[pymethods]
impl PyCgaPoint {
    #[new]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { inner: CgaPoint::point(x, y, z) }
    }

    fn distance(&self, other: &PyCgaPoint) -> f32 {
        self.inner.distance(&other.inner)
    }

    fn to_euclidean(&self) -> [f32; 3] {
        self.inner.to_euclidean()
    }

    fn __repr__(&self) -> String {
        let [x, y, z] = self.inner.to_euclidean();
        format!("CgaPoint(x={:.3}, y={:.3}, z={:.3})", x, y, z)
    }
}

#[pyclass(name = "CgaSphere")]
#[derive(Clone, Copy)]
pub struct PyCgaSphere {
    pub inner: CgaSphere,
}

#[pymethods]
impl PyCgaSphere {
    #[new]
    fn new(cx: f32, cy: f32, cz: f32, radius: f32) -> Self {
        Self { inner: CgaSphere::sphere(cx, cy, cz, radius) }
    }

    #[getter]
    fn radius(&self) -> Option<f32> {
        self.inner.sphere_radius()
    }

    fn invert_in_sphere(&self, pt: &PyCgaPoint) -> PyCgaPoint {
        PyCgaPoint { inner: pt.inner.invert_in_sphere(&self.inner) }
    }

    fn __repr__(&self) -> String {
        let [cx, cy, cz] = self.inner.to_euclidean();
        let r = self.inner.sphere_radius().unwrap_or(0.0);
        format!("CgaSphere(center=[{:.3}, {:.3}, {:.3}], radius={:.3})", cx, cy, cz, r)
    }
}

#[pyclass(name = "CgaCircle")]
#[derive(Clone, Copy)]
pub struct PyCgaCircle {
    pub inner: CgaCircle,
}

#[pymethods]
impl PyCgaCircle {
    #[new]
    fn new(center: [f32; 3], normal: [f32; 3], radius: f32) -> Self {
        Self { inner: CgaCircle::new(center, normal, radius) }
    }

    #[classmethod]
    fn from_sphere_plane(
        _cls: &Bound<'_, PyType>,
        sphere: &PyCgaSphere,
        plane_normal: [f32; 3],
        plane_d: f32,
    ) -> PyResult<Self> {
        let plane = crate::cga::CgaPlane::plane(plane_normal[0], plane_normal[1], plane_normal[2], plane_d);
        CgaCircle::from_sphere_plane_intersection(&sphere.inner, &plane)
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("Sphere and Plane do not intersect in a real circle"))
    }

    #[getter] fn center(&self) -> [f32; 3] { self.inner.center }
    #[getter] fn normal(&self) -> [f32; 3] { self.inner.normal }
    #[getter] fn radius(&self) -> f32 { self.inner.radius }
    #[getter] fn area(&self) -> f32 { self.inner.area() }
    #[getter] fn circumference(&self) -> f32 { self.inner.circumference() }

    fn __repr__(&self) -> String {
        format!("CgaCircle(center={:?}, normal={:?}, radius={:.3})", self.inner.center, self.inner.normal, self.inner.radius)
    }
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
    ) -> Self {
        Self {
            inner: CgaConformalMotor {
                translation,
                rotation_axis,
                rotation_angle_rad,
                dilation_factor,
            },
        }
    }

    fn transform_point(&self, pt: &PyCgaPoint) -> PyCgaPoint {
        PyCgaPoint { inner: self.inner.transform_point(&pt.inner) }
    }

    fn __repr__(&self) -> String {
        format!("CgaConformalMotor(trans={:?}, rot_axis={:?}, theta={:.3}, dilation={:.3})",
            self.inner.translation, self.inner.rotation_axis, self.inner.rotation_angle_rad, self.inner.dilation_factor
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
        Self { inner: StaVector::new(ct, x, y, z) }
    }

    #[getter] fn ct(&self) -> f32 { self.inner.ct }
    #[getter] fn x(&self) -> f32 { self.inner.x }
    #[getter] fn y(&self) -> f32 { self.inner.y }
    #[getter] fn z(&self) -> f32 { self.inner.z }

    fn interval_sq(&self) -> f32 { self.inner.interval_sq() }
    fn is_timelike(&self) -> bool { self.inner.is_timelike() }
    fn is_spacelike(&self) -> bool { self.inner.is_spacelike() }
    fn is_lightlike(&self) -> bool { self.inner.is_lightlike() }
    fn proper_time(&self) -> Option<f32> { self.inner.proper_time() }

    fn __repr__(&self) -> String {
        format!("StaVector(ct={:.3}, x={:.3}, y={:.3}, z={:.3})", self.inner.ct, self.inner.x, self.inner.y, self.inner.z)
    }
}

#[pyclass(name = "StaSpinor")]
#[derive(Clone, Copy)]
pub struct PyStaSpinor {
    pub inner: StaSpinor,
}

#[pymethods]
impl PyStaSpinor {
    #[new]
    #[pyo3(signature = (scalar = 0.0, ex = 0.0, ey = 0.0, ez = 0.0, bx = 0.0, by = 0.0, bz = 0.0, pseudoscalar = 0.0))]
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
            inner: StaSpinor::new(scalar, ex, ey, ez, bx, by, bz, pseudoscalar),
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

    fn bilinear_covariants<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let cov = self.inner.bilinear_covariants();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("sigma", cov.sigma)?;
        dict.set_item("omega", cov.omega)?;
        dict.set_item("j", [cov.j.ct, cov.j.x, cov.j.y, cov.j.z])?;
        dict.set_item("s", [cov.s.ex, cov.s.ey, cov.s.ez, cov.s.bx, cov.s.by, cov.s.bz])?;
        dict.set_item("k", [cov.k.ct, cov.k.x, cov.k.y, cov.k.z])?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("StaSpinor(scalar={:.3}, ex={:.3}, ey={:.3}, ez={:.3}, bx={:.3}, by={:.3}, bz={:.3}, p={:.3})",
            self.inner.scalar, self.inner.ex, self.inner.ey, self.inner.ez,
            self.inner.bx, self.inner.by, self.inner.bz, self.inner.pseudoscalar
        )
    }
}

// ============================================================================
// PYO3 WRAPPER: VERSORS & TURBOQUANT QUANTIZATION
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
        if dim % 64 != 0 {
            return Err(PyValueError::new_err(format!(
                "Dimension {} must be a multiple of at least 64",
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

    /// Applies isotropic randomized Hadamard rotation to 1D NumPy float32 array in O(N log N).
    #[pyo3(signature = (x))]
    fn apply<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<'py, f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let slice = x.as_slice()?;
        if slice.len() != self.inner.dimension() {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.inner.dimension(),
                slice.len()
            )));
        }

        let mut buf = slice.to_vec();
        self.inner.apply_in_place(&mut buf);
        Ok(PyArray1::from_vec(py, buf))
    }

    /// Inverts the randomized Hadamard rotation.
    fn invert<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<'py, f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let slice = x.as_slice()?;
        if slice.len() != self.inner.dimension() {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.inner.dimension(),
                slice.len()
            )));
        }

        let mut buf = slice.to_vec();
        self.inner.invert_in_place(&mut buf);
        Ok(PyArray1::from_vec(py, buf))
    }
}

/// Python wrapper for Clifford 3-bit CacheLine64 Quantized Tensor.
#[pyclass(name = "Clifford3BitTensor")]
pub struct PyClifford3BitTensor {
    inner: Clifford3BitTensor,
    seed: u64,
}

#[pymethods]
impl PyClifford3BitTensor {
    /// Compiles a 2D float32 NumPy weight matrix into Clifford 3-bit representation.
    #[classmethod]
    #[pyo3(signature = (weights, seed = 0x5645_5253_4F52_4331))]
    fn from_numpy(
        _cls: &Bound<'_, PyType>,
        weights: PyReadonlyArray2<'_, f32>,
        seed: u64,
    ) -> PyResult<Self> {
        let shape = weights.shape();
        let rows = shape[0];
        let cols = shape[1];

        if cols % 128 != 0 {
            return Err(PyValueError::new_err(format!(
                "Column count {} must be a multiple of 128 (Clifford3BitCacheLine64)",
                cols
            )));
        }

        let slice = weights.as_slice()?;
        let versor = DyadicWalshVersor::new(cols, seed);
        let tensor = Clifford3BitTensor::compile_aot(slice, rows, cols, &versor);

        Ok(Self { inner: tensor, seed })
    }

    #[getter] fn rows(&self) -> usize { self.inner.rows }
    #[getter] fn cols(&self) -> usize { self.inner.cols }
    #[getter] fn seed(&self) -> u64 { self.seed }

    #[getter]
    fn bytes_per_row(&self) -> usize {
        (self.inner.cols / 128) * std::mem::size_of::<Clifford3BitCacheLine64>()
    }

    #[getter]
    fn effective_bpw(&self) -> f32 {
        4.0
    }

    #[pyo3(signature = (x))]
    fn matvec<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<'py, f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let slice = x.as_slice()?;
        if slice.len() != self.inner.cols {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: matrix has {} cols, vector has {}",
                self.inner.cols,
                slice.len()
            )));
        }

        let versor = DyadicWalshVersor::new(self.inner.cols, self.seed);
        let mut out = vec![0.0f32; self.inner.rows];
        self.inner.matvec(slice, &mut out, &versor);

        Ok(PyArray1::from_vec(py, out))
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.inner.cache_lines.as_ptr() as *const u8,
                self.inner.cache_lines.len() * std::mem::size_of::<Clifford3BitCacheLine64>(),
            )
        };
        PyBytes::new(py, bytes_slice)
    }
}

/// Returns the 240 minimal roots of the E8 lattice as a (240, 8) float32 NumPy array.
#[pyfunction]
pub fn get_e8_roots<'py>(py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
    let roots = E8Lattice::generate_240_roots();
    let mut data = Vec::with_capacity(240 * 8);
    for r in &roots {
        data.extend_from_slice(r);
    }
    let arr2 = Array2::from_shape_vec((240, 8), data).unwrap();
    arr2.into_pyarray(py)
}

/// Returns the 8 Lloyd-Max 3-bit centroids as a (8,) float32 NumPy array.
#[pyfunction]
pub fn get_lloyd_max_3bit_centroids<'py>(py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
    let centroids = [
        LLOYD_MAX_3BIT_CENTROIDS[0],
        LLOYD_MAX_3BIT_CENTROIDS[1],
        LLOYD_MAX_3BIT_CENTROIDS[2],
        LLOYD_MAX_3BIT_CENTROIDS[3],
        -LLOYD_MAX_3BIT_CENTROIDS[0],
        -LLOYD_MAX_3BIT_CENTROIDS[1],
        -LLOYD_MAX_3BIT_CENTROIDS[2],
        -LLOYD_MAX_3BIT_CENTROIDS[3],
    ];
    PyArray1::from_slice(py, &centroids)
}

/// Fast Walsh-Hadamard Transform directly on 1D NumPy float32 array in O(N log N).
#[pyfunction]
pub fn fwht<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let slice = x.as_slice()?;
    if !slice.len().is_power_of_two() {
        return Err(PyValueError::new_err(format!(
            "FWHT requires length to be a power of 2, got {}",
            slice.len()
        )));
    }
    let mut data = slice.to_vec();
    DyadicWalshVersor::fwht_in_place(&mut data);
    Ok(PyArray1::from_vec(py, data))
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

    fn norm(&self) -> f32 {
        self.inner.norm()
    }

    fn lambda_factor(&self) -> f32 {
        self.inner.lambda()
    }

    fn geodesic_distance(&self, other: &PyPoincarePoint) -> PyResult<f32> {
        if self.inner.dim() != other.inner.dim() {
            return Err(PyValueError::new_err("Dimension mismatch in geodesic_distance"));
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
    fn einstein_midpoint(_cls: &Bound<'_, PyType>, points: Vec<PyPoincarePoint>, weights: Vec<f32>) -> PyResult<Self> {
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
            let kt: Vec<f32> = k0.iter().zip(k1.iter()).map(|(&a, &b)| a * (1.0 - t) + b * t).collect();
            let pt = PoincarePoint::from_klein(&kt);
            arc.push(pt.coords);
        }
        Ok(arc)
    }

    fn __repr__(&self) -> String {
        format!("PoincarePoint(coords={:?})", self.inner.coords)
    }
}

/// Native PyO3 module definition for versorc_math.
#[pymodule]
pub fn _versorc_math(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCl3>()?;
    m.add_class::<PyCl2>()?;
    m.add_class::<PyPgaMotor>()?;
    m.add_class::<PyCgaPoint>()?;
    m.add_class::<PyCgaSphere>()?;
    m.add_class::<PyCgaCircle>()?;
    m.add_class::<PyCgaConformalMotor>()?;
    m.add_class::<PyStaVector>()?;
    m.add_class::<PyStaSpinor>()?;
    m.add_class::<PyDyadicWalshVersor>()?;
    m.add_class::<PyClifford3BitTensor>()?;
    m.add_class::<PyPoincarePoint>()?;
    m.add_function(wrap_pyfunction!(get_e8_roots, m)?)?;
    m.add_function(wrap_pyfunction!(get_lloyd_max_3bit_centroids, m)?)?;
    m.add_function(wrap_pyfunction!(fwht, m)?)?;
    Ok(())
}
