"""Deterministic offline demo of the visibility adjudication module.

Scene: two known pinhole cameras looking at the analytic world plane
z = 5 + 0.3x + 0.2y.  Depth maps are generated here by independent
ray-plane intersection (not by the module under test), and projections
are cross-checked against direct hand formulas.  Runs in well under 8s.
"""

from visibility import (
    DepthMap,
    PinholeCamera,
    Status,
    VisibilityError,
    classify_point,
    classify_points,
    points_visible_in_at_least,
)

# --- scene constants -------------------------------------------------------
W = H = 16                      # depth map / image size (<= 32 per contract)
FX = FY = 24.0
CX = CY = (W - 1) / 2.0         # 7.5
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# --- independent reference math (no library calls) -------------------------
def plane_depth(cam_tx, px, py):
    """Camera-z where the ray through pixel (px, py) hits the plane.

    Camera sits at world (cam_tx, 0, 0) with identity orientation.
    Ray world point: (t*dx + cam_tx, t*dy, t) with dx = (px - cx)/fx.
    Plane z = 5 + 0.3x + 0.2y  =>  t = (5 + 0.3*cam_tx) / (1 - 0.3dx - 0.2dy).
    """
    dx = (px - CX) / FX
    dy = (py - CY) / FY
    return (5.0 + 0.3 * cam_tx) / (1.0 - 0.3 * dx - 0.2 * dy)


def make_depth_map(cam_tx):
    return [plane_depth(cam_tx, px, py) for py in range(H) for px in range(W)]


def point_on_plane(cam_tx, px, py, scale=1.0):
    """World point on the plane along camera cam_tx's pixel (px, py) ray."""
    dx = (px - CX) / FX
    dy = (py - CY) / FY
    t = plane_depth(cam_tx, px, py) * scale
    return (t * dx + cam_tx, t * dy, t)


def main():
    cam_a = PinholeCamera(FX, FY, CX, CY, IDENTITY, (0.0, 0.0, 0.0))
    cam_b = PinholeCamera(FX, FY, CX, CY, IDENTITY, (-1.0, 0.0, 0.0))
    map_a = DepthMap(W, H, make_depth_map(0.0))
    disabled_b = [False] * (W * H)
    for py in (4, 5):
        for px in (4, 5):
            disabled_b[py * W + px] = True     # 2x2 disabled block
    map_b = DepthMap(W, H, make_depth_map(1.0), disabled_b)

    print("=== 1. Direct projection reference (camera A) ===")
    point = point_on_plane(0.0, 8, 8)
    verdict = classify_point(point, cam_a, map_a, abs_tol=1e-9)
    # Independent reference: u = fx*x/z + cx with x = t*dx, z = t.
    ref_u = FX * (point[0] / point[2]) + CX
    ref_v = FY * (point[1] / point[2]) + CY
    print(f"point on plane at pixel (8, 8): status={verdict.status.value}")
    print(f"  projection module=({verdict.u:.6f}, {verdict.v:.6f}) "
          f"reference=({ref_u:.6f}, {ref_v:.6f})")
    print(f"  z={verdict.z:.9f} sampled depth={verdict.depth:.9f}")

    print("\n=== 2. One point per status (camera A) ===")
    cases = [
        ("on the plane        ", point_on_plane(0.0, 8, 8)),
        ("behind the plane    ", point_on_plane(0.0, 8, 8, scale=1.5)),
        ("in front of plane   ", point_on_plane(0.0, 8, 8, scale=0.6)),
        ("behind the camera   ", (0.1, -0.2, -3.0)),
        ("outside the image   ", (100.0, 0.0, 5.0)),
        ("edge, no 4-neighbour", point_on_plane(0.0, W - 1, 8)),
    ]
    for label, p in cases:
        v = classify_point(p, cam_a, map_a, abs_tol=1e-9)
        print(f"  {label}: {v.status.value:<8} ({v.reason})")

    print("\n=== 3. Same point, two cameras: states kept separate ===")
    shared = point_on_plane(1.0, 4, 4)   # projects onto B's disabled block
    for name, cam, dmap in (("A", cam_a, map_a), ("B", cam_b, map_b)):
        v = classify_point(shared, cam, dmap, abs_tol=1e-9, rel_tol=0.01)
        print(f"  camera {name}: {v.status.value:<8} ({v.reason})")

    print("\n=== 4. Aggregation: points visible in at least k=2 views ===")
    points = [
        point_on_plane(0.0, 8, 8),               # visible in A and B
        point_on_plane(1.0, 4, 4),               # visible A, unknown B (mask)
        point_on_plane(0.0, 8, 8, scale=1.5),    # occluded in both
        point_on_plane(0.0, 10, 12),             # visible in A and B
    ]
    # rel_tol covers the bilinear interpolation error of the (rational)
    # plane depth function at non-integer projection coordinates.
    results = classify_points(points, [(cam_a, map_a), (cam_b, map_b)],
                              abs_tol=1e-9, rel_tol=0.01)
    rows = [[verdict.status for verdict in row] for row in results]
    for i, row in enumerate(rows):
        print(f"  point {i}: A={row[0].value:<8} B={row[1].value}")
    selected = points_visible_in_at_least(rows, 2)
    print(f"  visible in >= 2 views: {sorted(selected)}")

    print("\n=== 5. Triggered rejection (invalid input) ===")
    try:
        classify_point((float("nan"), 0.0, 1.0), cam_a, map_a)
    except VisibilityError as exc:
        print(f"  NaN point rejected: VisibilityError: {exc}")
    try:
        DepthMap(True, H, [1.0] * H)
    except VisibilityError as exc:
        print(f"  bool width rejected: VisibilityError: {exc}")


if __name__ == "__main__":
    main()
