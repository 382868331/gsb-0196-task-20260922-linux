"""固定输入演示：深度遮挡多视图可见性裁定（含深度区间）。

直接运行：``python demo.py``（90 秒内完成，实际计算，无网络、无
sleep、无预录结果）。演示数据在本文件内确定性构造：

- 3 个针孔视图（8x8 深度图），第三视图含一个显式禁用像素；
- 6 个标量探针点，触发 visible / occluded / in_front 及三类 unknown；
- 深度区间探针：确定可见、确定遮挡、确定在前、区间接触/重叠
  （uncertain，不判可见）、跨相机原点退化、整条段在相机后方；
- 多视图逐视图原因保留（不融合成二值）与至少 k 视图可见聚合；
- 末尾实际触发两次输入拒绝（NaN 标量点、反向区间 z_near>z_far）。
"""

from __future__ import annotations

from visibility import (
    Camera,
    DepthInterval,
    DepthMap,
    IntervalQuery,
    InvalidParameterError,
    InvalidPointError,
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


def tag_of(st) -> str:
    t = st.visibility.value
    if st.visibility in (Visibility.UNKNOWN, Visibility.UNCERTAIN):
        t += "/" + st.reason.value
    return t


def main() -> None:
    plane = [[SURFACE_Z for _ in range(W)] for _ in range(H)]
    # 视图 2 显式禁用像素 (row=3, col=3)。
    mask = [[False for _ in range(W)] for _ in range(H)]
    mask[3][3] = True

    cam = Camera(K=K, R=IDENT, t=ORIGIN)
    views = [
        View(cam, DepthMap(plane)),
        View(cam, DepthMap([row[:] for row in plane])),
        View(cam, DepthMap(plane, mask)),
    ]

    # --- 标量探针 -----------------------------------------------------------
    probes = [
        ("表面上(visible)      ", point_at_pixel(2.5, 2.5, SURFACE_Z)),
        ("表面后(occluded)     ", point_at_pixel(2.5, 2.5, 7.5)),
        ("表面前(in_front)     ", point_at_pixel(2.5, 2.5, 4.5)),
        ("相机后(behind)       ", point_at_pixel(2.5, 2.5, -1.0)),
        ("右边缘缺邻域(missing)", point_at_pixel(7.0, 3.0, SURFACE_Z)),
        ("落在禁用像素(masked) ", point_at_pixel(2.5, 2.5, SURFACE_Z)),
    ]

    print("== A. 标量点 x 单视图裁定（容差 1e-6 绝对 + 1e-6 相对） ==")
    header = f"{'探针点':<24}" + "".join(f"{'视图'+str(i):>26}" for i in range(3))
    print(header)
    for name, P in probes:
        cells = [f"{tag_of(classify_view(vw, P)):>26}" for vw in views]
        print(f"{name:<24}" + "".join(cells))

    st = classify_view(views[0], probes[1][1])
    print(
        f"\n抽查 occluded: projected_z={st.projected_z:.3f} "
        f"sampled_depth={st.sampled_depth:.3f} (u,v)=({st.u:.2f},{st.v:.2f})"
    )

    # --- 深度区间探针（像素 2.5,2.5，表面 6.0；本节用零容差便于手算） -------
    print("\n== B. 深度区间裁定（表面 d=6.0，零容差，像素 2.5,2.5） ==")
    interval_probes = [
        ("零宽贴表面", DepthInterval(6.0, 6.0)),
        ("近端接触  ", DepthInterval(6.0, 7.0)),
        ("双向重叠  ", DepthInterval(5.5, 6.5)),
        ("确定遮挡  ", DepthInterval(6.5, 7.5)),
        ("确定在前  ", DepthInterval(4.5, 5.5)),
        ("跨原点退化", DepthInterval(-1.0, 6.0)),
        ("全在相机后", DepthInterval(-3.0, -1.0)),
    ]
    for name, iv in interval_probes:
        query = IntervalQuery(u=2.5, v=2.5, interval=iv)
        s = classify_view(views[0], query, abs_tol=0.0, rel_tol=0.0)
        extra = ""
        if s.surface_interval is not None:
            extra = (f" 表面带[{s.surface_interval[0]:.3f},"
                     f"{s.surface_interval[1]:.3f}]")
        print(f"  区间[{iv.z_near:5.1f},{iv.z_far:5.1f}] {name}: "
              f"{tag_of(s)}{extra}")
    print("  *零容差下非零宽度区间不可能被零宽带包含；真包含见下方默认容差例。")

    # 默认容差下，非零宽度区间可被容差带完全包含 -> 确定可见。
    band_q = IntervalQuery(u=2.5, v=2.5, interval=DepthInterval(6.0 - 1e-6, 6.0 + 1e-6))
    bs = classify_view(views[0], band_q)
    print(f"\n默认容差下区间[6-1e-6, 6+1e-6] -> {tag_of(bs)}，"
          f"表面带[{bs.surface_interval[0]:.7f}, {bs.surface_interval[1]:.7f}]")

    # --- 多视图：逐视图原因保留，不融合成二值 -------------------------------
    print("\n== C. 同一区间在三个不同表面深度视图下的逐视图裁定 ==")
    mv_views = [
        View(cam, DepthMap([[d] * W for _ in range(H)]))
        for d in (5.0, 6.0, 7.0)
    ]
    mv_query = IntervalQuery(u=2.5, v=2.5, interval=DepthInterval(5.5, 6.5))
    for i, vw in enumerate(mv_views):
        s = classify_view(vw, mv_query, abs_tol=0.0, rel_tol=0.0)
        print(f"  视图{i} 表面={5.0 + i:.1f}: {tag_of(s)}")

    # --- 标量+区间混合批次与 k 聚合 -----------------------------------------
    print("\n== D. 混合批次聚合（只计 visible，uncertain/occluded 不翻转） ==")
    rows = classify_points(
        views,
        [
            probes[0][1],                                   # 标量 visible
            IntervalQuery(2.5, 2.5, DepthInterval(6.0, 6.0)),  # 区间零宽 visible
            IntervalQuery(2.5, 2.5, DepthInterval(5.5, 6.5)),  # 重叠 uncertain
        ],
        abs_tol=0.0,
        rel_tol=0.0,
    )
    for i, row in enumerate(rows):
        print(f"  查询{i}: " + ", ".join(
            f"v{j}:{tag_of(s)}" for j, s in enumerate(row)))
    for k in (1, 2, 3):
        print(f"  k={k}: 入选查询下标 = {indices_visible_in_at_least_k(rows, k)}")

    # --- 实际触发的拒绝边界 -------------------------------------------------
    print("\n== E. 输入拒绝边界（真实抛错并捕获，不静默纠正） ==")
    try:
        classify_view(views[0], (0.0, float("nan"), SURFACE_Z))
    except InvalidPointError as exc:
        print(f"  标量 NaN 点 -> {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("NaN 输入应当被拒绝")

    try:
        DepthInterval(9.0, 1.0)  # z_near > z_far
    except InvalidParameterError as exc:
        print(f"  反向深度区间 -> {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("z_near > z_far 应当被拒绝")

    print("\n演示正常结束。")


if __name__ == "__main__":
    main()
