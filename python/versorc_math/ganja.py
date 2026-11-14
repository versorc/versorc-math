# ============================================================================
# VERSORC-MATH: GANJA.JS FORMAT COMPATIBILITY ADAPTER (PURE PYTHON STDLIB)
# License: Apache-2.0
# ============================================================================

"""
Pure format-interchange adapter mapping `versorc_math` objects to `ganja.js`
compatible JSON schemas `(p, q, r)`, basis names, and ordered coefficient arrays.
Does not load `ganja.js`, emit HTML, open browsers, or fetch external resources.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


CL2_BASIS: Tuple[str, ...] = ("1", "e1", "e2", "e12")
CL3_BASIS: Tuple[str, ...] = ("1", "e1", "e2", "e12", "e3", "e13", "e23", "e123")
PGA3D_MOTOR_BASIS: Tuple[str, ...] = ("1", "e23", "e31", "e12", "e01", "e02", "e03", "e0123")
PGA3D_POINT_BASIS: Tuple[str, ...] = ("e032", "e013", "e021", "e123")
PGA3D_PLANE_BASIS: Tuple[str, ...] = ("e1", "e2", "e3", "e0")
CGA3D_NULL_VECTOR_BASIS: Tuple[str, ...] = ("no", "e1", "e2", "e3", "ni")


def _extract_sequence(obj: Any) -> List[float]:
    """Extracts float coefficients from any sequence, buffer, or object with `tolist`."""
    if hasattr(obj, "tolist"):
        return [float(v) for v in obj.tolist()]
    if hasattr(obj, "to_list"):
        return [float(v) for v in obj.to_list()]
    return [float(v) for v in list(memoryview(obj))]


def to_ganja_payload(obj: Any) -> Dict[str, Any]:
    """
    Converts a `versorc_math` geometric entity into a `ganja.js`-compatible
    format dictionary specifying metric signature `(p, q, r)`, basis labels,
    and ordered float coefficients.
    """
    type_name = type(obj).__name__

    if type_name == "Cl2":
        return {
            "signature": [2, 0, 0],
            "algebra": "Cl(2,0)",
            "type": "multivector",
            "basis": list(CL2_BASIS),
            "coefficients": _extract_sequence(obj),
        }

    if type_name == "Cl3":
        return {
            "signature": [3, 0, 0],
            "algebra": "Cl(3,0)",
            "type": "multivector",
            "basis": list(CL3_BASIS),
            "coefficients": _extract_sequence(obj),
        }

    if type_name == "PgaMotor":
        coeffs = [
            float(obj.s),
            float(obj.b23),
            float(obj.b31),
            float(obj.b12),
            float(obj.d01),
            float(obj.d02),
            float(obj.d03),
            float(obj.e0123),
        ]
        return {
            "signature": [3, 0, 1],
            "algebra": "Cl*(3,0,1)",
            "type": "motor",
            "basis": list(PGA3D_MOTOR_BASIS),
            "coefficients": coeffs,
        }

    if type_name == "PgaPoint":
        w = float(getattr(obj, "w", 1.0))
        return {
            "signature": [3, 0, 1],
            "algebra": "Cl*(3,0,1)",
            "type": "dual_point",
            "basis": list(PGA3D_POINT_BASIS),
            "coefficients": [float(obj.x), float(obj.y), float(obj.z), w],
        }

    if type_name == "PgaPlane":
        return {
            "signature": [3, 0, 1],
            "algebra": "Cl*(3,0,1)",
            "type": "plane",
            "basis": list(PGA3D_PLANE_BASIS),
            "coefficients": [float(obj.a), float(obj.b), float(obj.c), float(obj.d)],
        }

    if type_name == "CgaPoint":
        x, y, z = [float(v) for v in obj.to_euclidean()]
        return {
            "signature": [4, 1, 0],
            "algebra": "Cl(4,1)",
            "type": "conformal_point",
            "basis": list(CGA3D_NULL_VECTOR_BASIS),
            "coefficients": [1.0, x, y, z, 0.5 * (x * x + y * y + z * z)],
        }

    if type_name == "CgaSphere":
        raw = _extract_sequence(obj) if hasattr(obj, "to_list") else None
        radius = float(obj.radius) if obj.radius is not None else 0.0
        payload: Dict[str, Any] = {
            "signature": [4, 1, 0],
            "algebra": "Cl(4,1)",
            "type": "conformal_sphere",
            "radius": radius,
        }
        if raw is not None:
            payload["basis"] = list(CGA3D_NULL_VECTOR_BASIS)
            payload["coefficients"] = raw
        return payload

    if type_name == "CgaCircle":
        return {
            "signature": [4, 1, 0],
            "algebra": "Cl(4,1)",
            "type": "conformal_circle",
            "center": [float(v) for v in obj.center],
            "normal": [float(v) for v in obj.normal],
            "radius": float(obj.radius),
        }

    raise TypeError(f"Unsupported type for Ganja format serialization: {type_name}")


def to_ganja_json(obj: Any, indent: Optional[int] = None) -> str:
    """Serializes a `versorc_math` entity to a JSON string in Ganja-compatible format."""
    return json.dumps(to_ganja_payload(obj), indent=indent)


class GanjaScene:
    """
    Data-only container collecting geometric entities and signatures in a format
    directly consumable by external `ganja.js` frontends via JSON.
    """

    def __init__(
        self,
        p: int = 3,
        q: int = 0,
        r: int = 1,
        title: str = "VersorC Math Ganja Scene",
        camera: Optional[List[float]] = None,
    ):
        self.p = int(p)
        self.q = int(q)
        self.r = int(r)
        self.title = str(title)
        self.camera = [float(v) for v in (camera if camera is not None else [0.0, 0.0, -3.5])]
        self.elements: List[Dict[str, Any]] = []

    def add_element(
        self,
        obj: Any,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serializes a `versorc_math` entity and appends it to the scene payload."""
        entry: Dict[str, Any] = {"data": to_ganja_payload(obj)}
        if label is not None:
            entry["label"] = str(label)
        if color is not None:
            entry["color"] = str(color)
        self.elements.append(entry)
        return entry

    def to_dict(self) -> Dict[str, Any]:
        """Returns the full scene schema as a plain Python dictionary."""
        return {
            "signature": [self.p, self.q, self.r],
            "title": self.title,
            "camera": list(self.camera),
            "elements": list(self.elements),
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serializes the scene schema to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, filepath: str) -> None:
        """Writes the JSON scene payload to a local file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())
