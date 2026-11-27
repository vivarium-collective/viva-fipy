"""A general, config-driven single-field reaction-advection-diffusion process."""
from __future__ import annotations

from process_bigraph import Process

from .engine import fipy_classes, grid2d, lu_solver, solve_sweep

_OW = {"_type": "overwrite[float]"}


class FipyTransportProcess(Process):
    """Generic 2-D reaction-advection-diffusion ``dc/dt = D∇²c − u·∇c + source`` solved
    with the genuine FiPy FVM engine. Config sets the grid, diffusivity, a constant
    background velocity ``(vx, vy)`` and optional Dirichlet values on the left/right
    faces; a ``source`` input port is a uniform volumetric source each step. Emits the
    field's mean / max / right-outlet value.

    Coupled or buoyancy-driven systems (e.g. a Darcy-Rayleigh vent reactor) keep their
    domain logic and reuse the engine primitives (``grid2d`` / ``lu_solver`` /
    ``solve_sweep`` / ``fipy_classes``) instead.
    """

    config_schema = {
        "nx": {"_type": "integer", "_default": 40}, "ny": {"_type": "integer", "_default": 40},
        "length_m": {"_type": "float", "_default": 1.0}, "height_m": {"_type": "float", "_default": 1.0},
        "diffusivity": {"_type": "float", "_default": 1.0e-9, "_units": "meter**2/second"},
        "vx": {"_type": "float", "_default": 0.0, "_units": "meter/second"},
        "vy": {"_type": "float", "_default": 1.0e-3, "_units": "meter/second"},
        "initial": {"_type": "float", "_default": 0.0},
        "left": {"_type": "float", "_default": 0.0},    # Dirichlet on facesLeft
        "right": {"_type": "float", "_default": 0.0},   # Dirichlet on facesRight
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._f = None

    def inputs(self):
        return {"source": {"_type": "float", "_units": "mole/second"}}

    def outputs(self):
        return {"mean": _OW, "max": _OW, "outlet": _OW}

    def _build(self):
        F = fipy_classes()
        nx, ny = int(self.config["nx"]), int(self.config["ny"])
        dx = float(self.config["length_m"]) / nx
        dy = float(self.config["height_m"]) / ny
        self._mesh = grid2d(nx, ny, dx, dy)
        self._c = F["CellVariable"](name="c", mesh=self._mesh, value=float(self.config["initial"]), hasOld=True)
        self._c.constrain(float(self.config["left"]), self._mesh.facesLeft)
        self._c.constrain(float(self.config["right"]), self._mesh.facesRight)
        self._c.faceGrad.constrain(0.0, self._mesh.facesTop | self._mesh.facesBottom)
        u = F["FaceVariable"](mesh=self._mesh, rank=1,
                              value=((float(self.config["vx"]),), (float(self.config["vy"]),)))
        self._source = F["CellVariable"](name="source", mesh=self._mesh, value=0.0)
        self._eq = (F["TransientTerm"]() == F["DiffusionTerm"](coeff=float(self.config["diffusivity"]))
                    - F["PowerLawConvectionTerm"](coeff=u) + self._source)
        self._solver = lu_solver()
        self._f = True

    def update(self, state, interval):
        import numpy as np
        if self._f is None:
            self._build()
        self._source.setValue(float(state.get("source", 0.0)))
        solve_sweep(self._eq, self._c, float(interval), solver=self._solver)
        v = np.asarray(self._c.value, dtype=float)
        # right-outlet = mean over the right-most column of cells
        nx, ny = int(self.config["nx"]), int(self.config["ny"])
        outlet = float(v.reshape((ny, nx))[:, -1].mean())
        return {"mean": float(v.mean()), "max": float(v.max()), "outlet": outlet}
