# ============================================================================
# VERSORC-MATH: INTERACTIVE GANJA.JS GEOMETRIC ALGEBRA VISUALIZATION SUITE
# License: Apache-2.0
# ============================================================================

import json
import os
import tempfile
import webbrowser
from typing import List, Optional, Tuple, Union

try:
    from ._versorc_math import (
        Cl3,
        PgaMotor,
        CgaPoint,
        CgaSphere,
        CgaCircle,
        CgaConformalMotor,
    )
except ImportError:
    pass


class GanjaScene:
    """
    Interactive WebGL/SVG Geometric Algebra scene rendered with GanJa.js.
    Generates standalone, zero-dependency HTML files with real-time 3D camera
    controls, animation loops, and mouse drag interaction.
    """

    def __init__(
        self,
        p: int = 3,
        q: int = 0,
        r: int = 1,
        title: str = "VersorC Math · GanJa.js Visualization",
        width: str = "100%",
        height: str = "600px",
        animate: bool = True,
        camera: Optional[List[float]] = None,
    ):
        self.p = p
        self.q = q
        self.r = r
        self.title = title
        self.width = width
        self.height = height
        self.animate = animate
        self.camera = camera or [0.0, 0.0, -3.5]
        self.elements_js = []
        self.custom_js_setup = []

    def add_raw_js(self, js_code: str):
        """Adds arbitrary JavaScript / GanJa code inside the Algebra block."""
        self.elements_js.append(js_code)

    def add_color(self, hex_color: Union[int, str]):
        """Sets active drawing color (e.g. 0xff0000 or '#ff0000')."""
        if isinstance(hex_color, str):
            c = int(hex_color.replace("#", "0x"), 16)
            self.elements_js.append(f"0x{c:08x}")
        else:
            self.elements_js.append(f"0x{hex_color:08x}")

    def add_label(self, label: str):
        """Adds a text label at the next element position."""
        self.elements_js.append(json.dumps(label))

    def to_html(self) -> str:
        """Generates self-contained, responsive HTML embedding GanJa.js via CDN."""
        elements_str = ",\n        ".join(self.elements_js)
        setup_str = "\n    ".join(self.custom_js_setup)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self.title}</title>
  <script src="https://cdn.jsdelivr.net/npm/ganja.js"></script>
  <style>
    body, html {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background-color: #0b0d13;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: #e0e6ed;
    }}
    #header {{
      position: absolute;
      top: 14px;
      left: 18px;
      z-index: 100;
      pointer-events: none;
      background: rgba(16, 20, 31, 0.75);
      backdrop-filter: blur(8px);
      padding: 10px 18px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    #header h1 {{
      margin: 0;
      font-size: 15px;
      font-weight: 600;
      letter-spacing: 0.5px;
      color: #ffffff;
    }}
    #header p {{
      margin: 4px 0 0 0;
      font-size: 12px;
      color: #8da2be;
    }}
    #canvas-container {{
      width: {self.width};
      height: {self.height};
      display: flex;
      justify-content: center;
      align-items: center;
    }}
    canvas {{
      width: 100% !important;
      height: 100% !important;
    }}
  </style>
</head>
<body>
  <div id="header">
    <h1>{self.title}</h1>
    <p>Clifford Algebra Cl({self.p}, {self.q}, {self.r}) · Drag with left/right mouse button to orbit/pan</p>
  </div>
  <div id="canvas-container"></div>

  <script>
    Algebra({self.p}, {self.q}, {self.r}, function() {{
      {setup_str}
      
      var scene = [
        {elements_str}
      ];

      var graphElement = this.graph(scene, {{
        animate: {"true" if self.animate else "false"},
        camera: {json.dumps(self.camera)},
        gl: true
      }});

      document.getElementById('canvas-container').appendChild(graphElement);
    }});
  </script>
</body>
</html>
"""

    def save(self, filepath: str):
        """Saves the interactive HTML scene to a local file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_html())

    def show(self):
        """Opens the interactive scene in the default web browser."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as tmp:
            tmp.write(self.to_html())
            tmp_path = tmp.name
        webbrowser.open(f"file://{tmp_path}")


# ============================================================================
# CANONICAL VISUALIZATION GENERATORS
# ============================================================================

def plot_pga_screw_motion(
    translation: List[float] = [2.0, 1.0, 1.5],
    rotation_axis: List[float] = [0.0, 0.0, 1.0],
    angle_rad: float = 3.14159,
    num_frames: int = 60,
    save_path: Optional[str] = None,
) -> GanjaScene:
    """
    Renders an interactive 3D PGA (3,0,1) scene demonstrating SE(3) screw kinematics,
    coordinate frame transformation, and smooth motor interpolation (Sclerp).
    """
    scene = GanjaScene(
        p=3,
        q=0,
        r=1,
        title="VersorC-Math · 3D PGA SE(3) Screw Kinematics & Motor Action",
        camera=[0.0, 0.0, -5.0],
        animate=True,
    )

    tx, ty, tz = translation
    rx, ry, rz = rotation_axis

    scene.custom_js_setup.append(f"""
      // Base Euclidean Points
      var origin = !(1e0);
      var pX = !(1e0 + 1.2*1e1);
      var pY = !(1e0 + 1.2*1e2);
      var pZ = !(1e0 + 1.2*1e3);

      // Target motor parameters
      var tx = {tx}, ty = {ty}, tz = {tz};
      var axis = [ {rx}, {ry}, {rz} ];
      var totalAngle = {angle_rad};

      // Continuous animated motor M(t)
      var motor = ()=>{{
        var time = (Date.now() * 0.001) % 4.0;
        var t = 0.5 * (1.0 - Math.cos(time * Math.PI * 0.5)); // smooth oscillation
        var ang = t * totalAngle * 0.5;
        var c = Math.cos(ang), s = Math.sin(ang);
        
        // Rotor around axis
        var R = c - s * (axis[0]*1e23 + axis[1]*1e31 + axis[2]*1e12);
        // Translator along vector
        var T = 1 - 0.5 * t * (tx*1e01 + ty*1e02 + tz*1e03);
        // Rigid motor M = T * R
        return T * R;
      }};
    """)

    # Grid / World ground plane
    scene.add_color(0x334455aa)
    scene.add_raw_js("1e2")  # ground plane y = 0

    # Fixed base coordinate axes
    scene.add_color(0xff3333)
    scene.add_label("World X")
    scene.add_raw_js("origin & pX")
    scene.add_color(0x33cc33)
    scene.add_label("World Y")
    scene.add_raw_js("origin & pY")
    scene.add_color(0x3388ff)
    scene.add_label("World Z")
    scene.add_raw_js("origin & pZ")

    # Animated transformed body frame
    scene.add_color(0xffaa00)
    scene.add_label("Rigid Body")
    scene.add_raw_js("()=> motor() * origin * ~motor()")
    scene.add_color(0xff4444)
    scene.add_raw_js("()=> motor() * (origin & pX) * ~motor()")
    scene.add_color(0x44ff44)
    scene.add_raw_js("()=> motor() * (origin & pY) * ~motor()")
    scene.add_color(0x44aaff)
    scene.add_raw_js("()=> motor() * (origin & pZ) * ~motor()")

    # Kinematic screw axis line
    scene.add_color(0xaa44ff)
    scene.add_label("Screw Axis")
    scene.add_raw_js(f"!(1e0) & !(1e0 + {rx}*1e1 + {ry}*1e2 + {rz}*1e3)")

    if save_path:
        scene.save(save_path)
    return scene


def plot_cga_spheres_and_circles(
    sphere_center: List[float] = [0.0, 0.0, 0.0],
    sphere_radius: float = 2.0,
    plane_normal: List[float] = [0.0, 1.0, 0.0],
    plane_dist: float = 0.5,
    save_path: Optional[str] = None,
) -> GanjaScene:
    """
    Renders an interactive 3D Conformal Geometric Algebra (CGA 4,1) scene,
    demonstrating sphere-plane circle intersections and conformal sphere inversions.
    """
    scene = GanjaScene(
        p=4,
        q=1,
        r=0,
        title="VersorC-Math · 3D CGA (4,1) Conformal Inversions & Circles",
        camera=[0.0, 0.0, -6.0],
        animate=True,
    )

    cx, cy, cz = sphere_center
    nx, ny, nz = plane_normal

    scene.custom_js_setup.append(f"""
      // CGA Basis: e1, e2, e3 (Euclidean), 1e4 (e+), 1e5 (e-)
      // Null basis: no = 0.5*(1e4 - 1e5), ni = 1e4 + 1e5
      var no = 0.5*(1e4 - 1e5);
      var ni = 1e4 + 1e5;

      // CGA Point Constructor
      var up = (x, y, z) => {{
        return no + x*1e1 + y*1e2 + z*1e3 + 0.5*(x*x + y*y + z*z)*ni;
      }};

      // Dual Sphere: S = P - 0.5*r^2*ni
      var R = {sphere_radius};
      var sphereCenter = up({cx}, {cy}, {cz});
      var dualSphere = sphereCenter - 0.5*R*R*ni;

      // Dual Plane: p = nx*e1 + ny*e2 + nz*e3 + d*ni
      var dualPlane = {nx}*1e1 + {ny}*1e2 + {nz}*1e3 + {plane_dist}*ni;

      // Circle = DualSphere ^ DualPlane
      var circle = dualSphere ^ dualPlane;

      // Dynamic orbiting probe point
      var probePoint = ()=>{{
        var t = Date.now() * 0.0015;
        var px = 3.5 * Math.cos(t);
        var py = 1.0 * Math.sin(t*0.5);
        var pz = 3.5 * Math.sin(t);
        return up(px, py, pz);
      }};

      // Conformal Inversion across Sphere: P' = S * P * S
      var invertedPoint = ()=>{{
        var P = probePoint();
        return dualSphere * P * dualSphere;
      }};
    """)

    # Sphere (translucent cyan)
    scene.add_color(0x3300ccff)
    scene.add_label("Inversion Sphere S")
    scene.add_raw_js("dualSphere")

    # Intersecting plane (translucent gold)
    scene.add_color(0x22ffaa00)
    scene.add_label("Cutting Plane")
    scene.add_raw_js("dualPlane")

    # Real circle intersection (glowing bright gold line)
    scene.add_color(0xffd700)
    scene.add_label("Intersection Circle C = S ∧ P")
    scene.add_raw_js("circle")

    # Exterior probe point (red)
    scene.add_color(0xff3344)
    scene.add_label("External Point P")
    scene.add_raw_js("probePoint")

    # Inverted interior point (emerald green)
    scene.add_color(0x00ff88)
    scene.add_label("Inverted Point P' = S P S")
    scene.add_raw_js("invertedPoint")

    if save_path:
        scene.save(save_path)
    return scene


def plot_cl3_bivector_orbit(
    vector_u: List[float] = [1.5, 0.0, 0.0],
    vector_v: List[float] = [0.0, 1.5, 0.0],
    save_path: Optional[str] = None,
) -> GanjaScene:
    """
    Renders an interactive 3D Euclidean Geometric Algebra (Cl 3,0) scene,
    demonstrating the 2-blade (bivector u ∧ v), rotor exponential action,
    and vector precession.
    """
    scene = GanjaScene(
        p=3,
        q=0,
        r=0,
        title="VersorC-Math · 3D Euclidean GA Cl(3,0) Bivector Blade & Rotor Orbit",
        camera=[0.0, 0.0, -4.5],
        animate=True,
    )

    ux, uy, uz = vector_u
    vx, vy, vz = vector_v

    scene.custom_js_setup.append(f"""
      var u = {ux}*1e1 + {uy}*1e2 + {uz}*1e3;
      var v = {vx}*1e1 + {vy}*1e2 + {vz}*1e3;
      var B = u ^ v; // 2-blade bivector

      // Unit bivector plane normal
      var B_norm = Math.sqrt(Math.abs((B * B).s || 1.0));
      var B_unit = B * (1.0 / B_norm);

      // Rotating vector v(t) = R(t) * u * ~R(t)
      var rotatingVector = ()=>{{
        var t = Date.now() * 0.002;
        var c = Math.cos(t * 0.5);
        var s = Math.sin(t * 0.5);
        var R = c - s * B_unit; // Clifford rotor exp(-B*t/2)
        return R * u * ~R;
      }};
    """)

    # Coordinate Origin
    scene.add_color(0x888888)
    scene.add_raw_js("[0, 0, 0]")

    # 1-Blades u and v
    scene.add_color(0xff4444)
    scene.add_label("Vector u")
    scene.add_raw_js("[ [0,0,0], u ]")

    scene.add_color(0x44ff44)
    scene.add_label("Vector v")
    scene.add_raw_js("[ [0,0,0], v ]")

    # 2-Blade B = u ^ v (planar patch)
    scene.add_color(0x442288ff)
    scene.add_label("Bivector 2-Blade u ∧ v")
    scene.add_raw_js("B")

    # Rotating vector orbit
    scene.add_color(0xffaa00)
    scene.add_label("Rotor Action v(t) = R u R~")
    scene.add_raw_js("()=> [ [0,0,0], rotatingVector() ]")

    if save_path:
        scene.save(save_path)
    return scene
