"""Q4 local default: sparse optimized patrol backbone + 25 m certificate cells.

Two differences from q4_dispatch_policy:

1. The continuous directional certificate (q4_coverage.certified_cells) is
   evaluated on 25 m squares instead of 100 m squares. Both are sufficient
   conditions for the same continuous statement (every corner strictly inside
   the hull of stations that cover the WHOLE square); the finer grid is merely
   less conservative near the disk boundary, so outer stations may sit nearer.
2. The initial patrol route comes from an offline local search
   (optimize_q4_backbone.py) that minimises open-route length subject to the
   fine-grid certificate, instead of the pruned 875 m triangular mesh.
   Result stored in q4_sparse_backbone.json (29 stations + origin, ~17.6 km, vs
   37 stations, ~31.5 km).

Local paired evidence: run_q4_sparse.py. Not an official-simulator result.
"""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from q4_coverage import certified_cells as _certified_cells, domain_cells
from q4_dispatch_policy import Policy as DispatchPolicy

HALF = 12.5
CELLS = domain_cells(2*HALF)
BACKBONE_FILE = Path(__file__).resolve().parent/'q4_sparse_backbone.json'


def certified(stations, cells=None):
    """Fine-grid continuous certificate (same predicate as q4_coverage)."""
    return _certified_cells(stations, CELLS if cells is None else cells, HALF)


def independent_cover_check(positions, half=HALF):
    """Qhull supporting-halfplane audit on the same fine grid."""
    stations = np.asarray(positions, float).reshape(-1, 2)
    cells = domain_cells(2*half)
    corners = np.array([[-half, -half], [-half, half], [half, -half], [half, half]])
    hulls = {}
    for c in cells:
        eligible = np.linalg.norm(np.abs(stations-c)+half, axis=1) <= 999.9
        p = stations[eligible]
        if len(p) < 3:
            return False
        # Adjacent cells often use exactly the same stations. Reuse only the
        # hull equations; every cell's four corners are still checked.
        key = eligible.tobytes()
        if key not in hulls:
            try:
                hulls[key] = ConvexHull(p).equations
            except Exception:
                return False
        eq = hulls[key]
        if np.max((c+corners)@eq[:, :2].T+eq[:, 2]) >= -1e-7:
            return False
    return True


def load_backbone(path=BACKBONE_FILE):
    # Cache by file contents, not filename: changed routes must be rechecked.
    return _validated_backbone(Path(path).read_bytes()).copy()


@lru_cache(maxsize=4)
def _validated_backbone(contents):
    data = json.loads(contents)
    route = np.asarray(data['route'], float).reshape(-1, 2)
    stations = np.vstack((np.zeros(2), route))
    if not certified(stations).all():
        raise AssertionError('Stored sparse backbone fails the fine-grid certificate')
    route.setflags(write=False)
    return route


class Policy(DispatchPolicy):
    """dispatch variant with the sparse backbone and the fine certificate."""

    def __init__(self, variant='dispatch', particles=32, seed=9411, backbone=BACKBONE_FILE):
        self.backbone = backbone
        super().__init__(variant, particles, seed)

    def initial_stations(self):
        # Avoid constructing/certifying the dense route just to discard it.
        return load_backbone(self.backbone)

    def certified(self, stations):
        return certified(stations)
