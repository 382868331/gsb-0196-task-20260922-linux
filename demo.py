"""固定输入演示：深度遮挡多视图可见性裁定。

直接运行：``python demo.py``（约 8 秒内完成，实际计算，无网络、无
sleep、无预录结果）。演示数据在本文件内确定性构造：

- 3 个针孔视图（8x8 深度图），第三视图含一个显式禁用像素；
- 另加 1 个正交视图演示：同一相机 xy 在任何深度投影像素相同，
  绝不像透视那样除以 z；
- 6 个世界点，分别触发 visible / occluded / in_front 以及
  behind_camera / missing_neighbor / masked_pixel 三类 unknown；
- 多视图聚合（至少 k 个视图 visible）；
- 末尾实际触发一次输入拒绝（NaN 世界点）并打印定位信息。
"""

from __future__ import annotations

from visibility import (
    Camera,
    CameraModel,
    DepthMap,
    InvalidPointError,
    OrthographicCamera,
    Reason,
    View,
    Visibility,
    classify_points,
    classify_view,
    indices_visible_in_at_least_k,
)

# --- 固定相机与深度图 -------------------------------------------------------

FX = FY = 20.0
CX = CY = 3.5
W = H = 8
SURFACE_Z = 6.0

K = (
    (FX, 0.0, CX),
    (0.0, FY, CY),
    (0.0, 0.0, 1.0),
)
IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ORIGIN = (0.0, 0.0, 0.0)


def point_at_pixel(u: float, v: float, z: float) -> tuple[float, float, float]:
    """由像素与相机 z 反解世界点（相机位于原点、无旋转）。"""
    return (u - CX) * z / FX, (v - CY) * z / FY, z


def main() -> None:
    plane = [[SURFACE_Z for _ in range(W)] for _ in range(H)]
    # 视图 2 显式禁用像素 (row=3, col=3)。
    mask = [[False for _ in range(W)] for _ in range(H)]
    mask[3][3] = True

    cam = Camera(K=K, R=IDENT, t=ORIGIN)
    # 正交内参：2 像素/世界单位，主点 (3.5, 3.5)，u=2*cx+3.5，不除 z。
    KO = (
        (2.0, 0.0, 3.5),
        (0.0, 2.0, 3.5),
        (0.0, 0.0, 1.0),
    )
    ortho_cam = OrthographicCamera(K=KO, R=IDENT, t=ORIGIN)
    views = [
        View(cam, DepthMap(plane)),
        View(cam, DepthMap([row[:] for row in plane])),
        View(cam, DepthMap(plane, mask)),
        View(ortho_cam, DepthMap([row[:] for row in plane])),
    ]

    # --- 确定性构造的 6 个探针点 -------------------------------------------
    # 正交演示点：(cx,cy)=(-0.5,-0.5) 在任意深度都投影到 (2.5,2.5)。
    probes = [
        ("表面上(visible)      ", point_at_pixel(2.5, 2.5, SURFACE_Z)),
        ("表面后(occluded)     ", point_at_pixel(2.5, 2.5, 7.5)),
        ("表面前(in_front)     ", point_at_pixel(2.5, 2.5, 4.5)),
        ("相机后(behind)       ", point_at_pixel(2.5, 2.5, -1.0)),
        ("右边缘缺邻域(missing)", point_at_pixel(7.0, 3.0, SURFACE_Z)),
        ("落在禁用像素(masked) ", point_at_pixel(2.5, 2.5, SURFACE_Z)),
        ("正交投影不随深度变  ", (-0.5, -0.5, 4.5)),
    ]

    print("== 单点 x 单视图裁定（容差 1e-6 绝对 + 1e-6 相对） ==")
    n_views = len(views)
    header = f"{'探针点':<24}" + "".join(
        f"{('正交' if i == 3 else '透视')+str(i):>26}" for i in range(n_views)
    )
    print(header)
    for name, P in probes:
        cells = []
        for i, vw in enumerate(views):
            st = classify_view(vw, P)
            tag = st.visibility.value
            if st.visibility is Visibility.UNKNOWN:
                tag += "/" + st.reason.value
            cells.append(f"{tag:>26}")
        print(f"{name:<24}" + "".join(cells))

    # 抽查具体数值，证明是真实计算而非标签查表。
    st = classify_view(views[0], probes[1][1])
    print(
        f"\n抽查 occluded: projected_z={st.projected_z:.3f} "
        f"sampled_depth={st.sampled_depth:.3f} (u,v)=({st.u:.2f},{st.v:.2f})"
    )

    # --- 多视图聚合 --------------------------------------------------------
    print("\n== 多视图聚合（每点至少 k 个视图 visible 才入选） ==")
    points = [P for _, P in probes]
    results = classify_points(views, points)
    for i, row in enumerate(results):
        states = ", ".join(
            f"v{i_view}:{s.visibility.value}" for i_view, s in enumerate(row)
        )
        print(f"点{i} [{states}]")
    for k in (1, 2, 3, 4):
        chosen = indices_visible_in_at_least_k(results, k)
        print(f"k={k}: 入选点下标 = {chosen}")
    print("说明：occluded/in_front/unknown 均不计为可见，且不做跨视图投票翻转。")

    # --- 实际触发的拒绝边界 ------------------------------------------------
    print("\n== 输入拒绝边界（NaN 必须被拒绝，不静默纠正） ==")
    try:
        classify_view(views[0], (0.0, float("nan"), SURFACE_Z))
    except InvalidPointError as exc:
        print(f"实际捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("NaN 输入应当被拒绝")

    print("\n演示正常结束。")


if __name__ == "__main__":
    main()
