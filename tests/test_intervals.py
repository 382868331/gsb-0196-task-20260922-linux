"""深度区间裁定的测试（本轮迭代新增）。

覆盖：

- 表面区间（near/far）与投影深度区间（depth_eps）各自及组合的
  visible / occluded / in_front / uncertain 裁定；
- 区间**接触**不判可见、重叠不直接判可见/遮挡；
- 容差板在 near/far 两端分别取容差；
- 原点深度退化（区间跨过 z=0）；
- near/far 两层分别双线性（解析斜面 + 变厚度）；
- 多视图逐视图原因保留，uncertain 不强制融合、不计入可见数；
- 构造与参数校验、防御性拷贝、批次原子性；
- 旧单点行为在 depth_eps=0 下逐位保持。

期望值以手算常量为主来源（投影相机 fx=fy=10、主点 0、无旋无平移，
探针像素 (1.5,1.5) 四邻点 (1,1)(2,1)(1,2)(2,2) 均为图内点），
``tests/reference.py`` 的独立区间实现只做交叉核对。"""

from __future__ import annotations

import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import reference as ref  # noqa: E402

from visibility import (  # noqa: E402
    Camera,
    DepthMap,
    InvalidDepthMapError,
    InvalidParameterError,
    Reason,
    View,
    Visibility,
    classify_intervals,
    classify_points,
    classify_view,
    count_visible_views,
    indices_visible_in_at_least_k,
)

K0 = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 1.0))
IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ZERO = (0.0, 0.0, 0.0)

PROBE_U = PROBE_V = 1.5  # 四邻点全部在图内的亚像素位置


def camera():
    return Camera(K=K0, R=IDENT, t=ZERO)


def point_at(u, v, z):
    """无旋无平移相机下，像素 (u,v)、相机 z 处的世界点（手算逆投影）。"""
    return (u * z / 10.0, v * z / 10.0, z)


def slab(w, h, near, far, disabled=None):
    """常量厚度的区间深度图 [near, far]。"""
    return DepthMap(
        [[near for _ in range(w)] for _ in range(h)],
        disabled,
        [[far for _ in range(w)] for _ in range(h)],
    )


def plane(w, h, d):
    return DepthMap([[d for _ in range(w)] for _ in range(h)])


class SurfaceIntervalPointDegenerateTests(unittest.TestCase):
    """表面为区间 [5,7]，点为单点（depth_eps=0）。"""

    def setUp(self):
        self.view = View(camera(), slab(4, 4, 5.0, 7.0))

    def classify(self, z, abs_tol=0.0, rel_tol=0.0):
        return classify_view(
            self.view, point_at(PROBE_U, PROBE_V, z),
            abs_tol=abs_tol, rel_tol=rel_tol,
        )

    def test_containment_and_strict_separation(self):
        self.assertEqual(self.classify(6.0).visibility, Visibility.VISIBLE)
        # 闭区间：恰在 near/far 端面上仍属可见。
        self.assertEqual(self.classify(5.0).visibility, Visibility.VISIBLE)
        self.assertEqual(self.classify(7.0).visibility, Visibility.VISIBLE)
        self.assertEqual(self.classify(4.0).visibility, Visibility.IN_FRONT)
        self.assertEqual(self.classify(8.0).visibility, Visibility.OCCLUDED)

    def test_endpoint_tolerance_band(self):
        # abs_tol=0.2：容差板 [4.8, 7.2]。
        self.assertEqual(self.classify(4.8, abs_tol=0.2).visibility,
                         Visibility.VISIBLE)
        self.assertEqual(self.classify(7.2, abs_tol=0.2).visibility,
                         Visibility.VISIBLE)
        self.assertEqual(self.classify(4.79, abs_tol=0.2).visibility,
                         Visibility.IN_FRONT)
        self.assertEqual(self.classify(7.21, abs_tol=0.2).visibility,
                         Visibility.OCCLUDED)
        # 板内仍可见，微小越界立刻翻转为确定结论。
        self.assertEqual(self.classify(7.2 + 1e-12, abs_tol=0.2).visibility,
                         Visibility.OCCLUDED)

    def test_scalar_point_never_uncertain(self):
        # 点退化为标量时，它要么在板内要么严格分离，不可能 uncertain。
        for z in [x * 0.05 for x in range(60, 181)]:
            st = self.classify(z, abs_tol=0.02, rel_tol=0.01)
            self.assertIsNot(st.visibility, Visibility.UNCERTAIN,
                             msg=f"scalar z={z}")

    def test_status_carries_both_endpoints(self):
        st = self.classify(6.0)
        self.assertEqual(st.sampled_depth, 5.0)
        self.assertEqual(st.sampled_depth_far, 7.0)
        self.assertEqual(st.depth_interval, (5.0, 7.0))
        self.assertEqual(st.projected_interval, (6.0, 6.0))
        self.assertEqual(st.reason, Reason.NONE)


class PointIntervalSurfaceDegenerateTests(unittest.TestCase):
    """点为区间 [z-eps, z+eps]，表面退化为单点 5。"""

    def setUp(self):
        self.view = View(camera(), plane(4, 4, 5.0))

    def classify(self, z, eps, abs_tol=0.0, rel_tol=0.0):
        return classify_view(
            self.view, point_at(PROBE_U, PROBE_V, z),
            abs_tol=abs_tol, rel_tol=rel_tol, depth_eps=eps,
        )

    def test_overlapping_point_interval_is_uncertain(self):
        # 旧版会把中心 z=5 判 visible；现在区间 [4.5,5.5] 超出表面两侧，
        # 只是覆盖了表面 -> uncertain，绝不能直接判 visible。
        st = self.classify(5.0, 0.5)
        self.assertEqual(st.visibility, Visibility.UNCERTAIN)
        self.assertEqual(st.reason, Reason.DEPTH_INTERVAL_OVERLAP)

    def test_contact_only_is_uncertain(self):
        # [4,5] 与表面 {5} 仅端点接触 -> 不确定（不能判可见）。
        self.assertEqual(self.classify(4.5, 0.5).visibility,
                         Visibility.UNCERTAIN)
        # [5,6] 同样接触 -> 不确定（不能判确定遮挡）。
        self.assertEqual(self.classify(5.5, 0.5).visibility,
                         Visibility.UNCERTAIN)

    def test_strict_separation_survives_eps(self):
        # [3.9,4.9] 整体在表面前；[5.1,6.1] 整体在表面后。
        self.assertEqual(self.classify(4.4, 0.5).visibility,
                         Visibility.IN_FRONT)
        self.assertEqual(self.classify(5.6, 0.5).visibility,
                         Visibility.OCCLUDED)

    def test_visible_only_when_interval_inside_tolerance_board(self):
        # abs_tol=0.5 -> 板 [4.5,5.5]。
        self.assertEqual(self.classify(5.0, 0.5, abs_tol=0.5).visibility,
                         Visibility.VISIBLE)
        self.assertEqual(self.classify(5.0, 0.6, abs_tol=0.5).visibility,
                         Visibility.UNCERTAIN)
        self.assertEqual(self.classify(4.0, 0.4, abs_tol=0.5).visibility,
                         Visibility.IN_FRONT)
        self.assertEqual(self.classify(6.0, 0.4, abs_tol=0.5).visibility,
                         Visibility.OCCLUDED)


class BothIntervalsTests(unittest.TestCase):
    """表面 [5,7]，点也有厚度。题面核心：接触与重叠。"""

    def setUp(self):
        self.view = View(camera(), slab(4, 4, 5.0, 7.0))

    def classify(self, z, eps):
        return classify_view(
            self.view, point_at(PROBE_U, PROBE_V, z),
            abs_tol=0.0, rel_tol=0.0, depth_eps=eps,
        )

    def test_contact_at_either_face_is_uncertain(self):
        # [4,5] 与板 [5,7] 仅在 near 端面接触：不判可见。
        self.assertEqual(self.classify(4.5, 0.5).visibility,
                         Visibility.UNCERTAIN)
        # [7,8] 仅在 far 端面接触：不判确定遮挡。
        self.assertEqual(self.classify(7.5, 0.5).visibility,
                         Visibility.UNCERTAIN)

    def test_partial_overlap_never_visible_nor_occluded(self):
        # [4.5,5.5] 越过 near 面；[6.5,7.5] 越过 far 面。
        self.assertEqual(self.classify(5.0, 0.5).visibility,
                         Visibility.UNCERTAIN)
        self.assertEqual(self.classify(7.0, 0.5).visibility,
                         Visibility.UNCERTAIN)
        # 点区间大部分在表面后方 [6.7,7.7]：与表面重叠就必须 uncertain。
        st = self.classify(7.2, 0.5)
        self.assertEqual(st.visibility, Visibility.UNCERTAIN)
        # 同一中心在 eps=0（标量）时是确定 occluded，形成对照。
        self.assertEqual(self.classify(7.2, 0.0).visibility,
                         Visibility.OCCLUDED)
        # 大范围区间包住整块表面 [6,10]：仍只是 uncertain。
        self.assertEqual(self.classify(8.0, 2.0).visibility,
                         Visibility.UNCERTAIN)

    def test_deterministic_cases(self):
        self.assertEqual(self.classify(6.0, 0.5).visibility,
                         Visibility.VISIBLE)      # [5.5,6.5] 整块在内
        self.assertEqual(self.classify(4.0, 0.5).visibility,
                         Visibility.IN_FRONT)     # [3.5,4.5]
        self.assertEqual(self.classify(8.0, 0.5).visibility,
                         Visibility.OCCLUDED)     # [7.5,8.5]


class ClassifyIntervalsPureTests(unittest.TestCase):
    """直接核对纯函数裁定与容差口径（手算板边界）。"""

    def test_relative_tol_uses_each_endpoint_depth(self):
        # 表面 [4,10]，rel_tol=0.1：板 [3.6, 11.0]。
        self.assertIs(
            classify_intervals(3.6, 11.0, 4.0, 10.0, 0.0, 0.1),
            Visibility.VISIBLE,
        )
        self.assertIs(
            classify_intervals(3.59, 11.0, 4.0, 10.0, 0.0, 0.1),
            Visibility.UNCERTAIN,
        )
        self.assertIs(
            classify_intervals(3.0, 3.59, 4.0, 10.0, 0.0, 0.1),
            Visibility.IN_FRONT,
        )
        self.assertIs(
            classify_intervals(11.01, 12.0, 4.0, 10.0, 0.0, 0.1),
            Visibility.OCCLUDED,
        )

    def test_closed_boundaries(self):
        # 点 [2,3] 与板 [3,5] 仅接触 -> uncertain。
        self.assertIs(
            classify_intervals(2.0, 3.0, 3.0, 5.0), Visibility.UNCERTAIN
        )
        # 点 [3,4] 整体在板内（接触下界）-> visible。
        self.assertIs(
            classify_intervals(3.0, 4.0, 3.0, 5.0), Visibility.VISIBLE
        )

    def test_inverted_interval_rejected(self):
        with self.assertRaises(InvalidParameterError):
            classify_intervals(2.0, 1.0, 3.0, 5.0)
        with self.assertRaises(InvalidParameterError):
            classify_intervals(1.0, 2.0, 5.0, 3.0)

    def test_nonfinite_and_negative_tol_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidParameterError):
                classify_intervals(bad, 2.0, 1.0, 3.0)
        with self.assertRaises(InvalidParameterError):
            classify_intervals(1.0, 2.0, 1.0, 3.0, abs_tol=-1e-9)
        with self.assertRaises(InvalidParameterError):
            classify_intervals(1.0, 2.0, 1.0, 3.0, rel_tol=True)

    def test_lattice_set_oracle_enumeration(self):
        """独立预言：把闭区间建模成整数格点集合，遍历所有区间组合。

        容差 abs_tol=1、rel_tol=0，故板端点都是整数；闭区间在整数格
        上的包含/相交关系与连续闭区间语义一致（含端点接触）。
        """
        grid = range(0, 9)
        for a, b in itertools.combinations_with_replacement(grid, 2):
            for c, d in itertools.combinations_with_replacement(grid, 2):
                pts = frozenset(range(a, b + 1))
                board = frozenset(range(c - 1, d + 2))
                if pts <= board:
                    want = Visibility.VISIBLE
                elif min(pts) > max(board):
                    want = Visibility.OCCLUDED
                elif max(pts) < min(board):
                    want = Visibility.IN_FRONT
                else:
                    self.assertTrue(pts & board)
                    want = Visibility.UNCERTAIN
                got = classify_intervals(float(a), float(b), float(c),
                                         float(d), abs_tol=1.0)
                self.assertIs(got, want,
                              msg=f"point=[{a},{b}] surface=[{c},{d}]")


class OriginDegeneracyTests(unittest.TestCase):
    """投影深度区间跨过相机原点 z=0 的退化。"""

    def setUp(self):
        self.view = View(camera(), slab(4, 4, 5.0, 7.0))

    def classify(self, z_world, eps, t_z=0.0):
        cam = Camera(K=K0, R=IDENT, t=(0.0, 0.0, t_z))
        vw = View(cam, self.view.depth_map)
        return classify_view(
            vw, (0.0, 0.0, z_world), depth_eps=eps,
            abs_tol=0.0, rel_tol=0.0,
        )

    def test_zero_depth_still_behind_camera(self):
        # 旧语义：cz==0、eps=0 -> behind_camera。
        st = self.classify(0.0, 0.0)
        self.assertEqual(st.visibility, Visibility.UNKNOWN)
        self.assertEqual(st.reason, Reason.BEHIND_CAMERA)
        self.assertIsNone(st.u)

    def test_interval_straddling_origin_is_unknown(self):
        # cz=0.5, eps=1 -> [-0.5, 1.5]：1/z 在区间内无定义。
        st = self.classify(0.5, 1.0)
        self.assertEqual(st.visibility, Visibility.UNKNOWN)
        self.assertEqual(st.reason, Reason.DEPTH_INTERVAL_AT_ORIGIN)
        self.assertEqual((st.projected_z_low, st.projected_z_high),
                         (-0.5, 1.5))
        self.assertIsNone(st.u)
        self.assertIsNone(st.v)
        self.assertIsNone(st.sampled_depth)

    def test_interval_touching_origin_is_unknown(self):
        # z_lo 恰为 0（闭区间含原点）也算退化。
        st = self.classify(1.0, 1.0)
        self.assertEqual(st.reason, Reason.DEPTH_INTERVAL_AT_ORIGIN)

    def test_strictly_negative_interval_behind(self):
        st = self.classify(-2.0, 1.0)  # [-3,-1]
        self.assertEqual(st.visibility, Visibility.UNKNOWN)
        self.assertEqual(st.reason, Reason.BEHIND_CAMERA)

    def test_translated_camera_degeneracy(self):
        # cz = z_world + 10：世界 z=-9.5 -> cz=0.5，eps=1 跨原点。
        st = self.classify(-9.5, 1.0, t_z=10.0)
        self.assertEqual(st.reason, Reason.DEPTH_INTERVAL_AT_ORIGIN)
        # 世界 z=-10 -> cz=0 -> behind_camera。
        st0 = self.classify(-10.0, 0.0, t_z=10.0)
        self.assertEqual(st0.reason, Reason.BEHIND_CAMERA)

    def test_origin_check_has_priority(self):
        # 即便投影后必然越界，跨原点也先返回 at_origin，不继续投影。
        st = self.classify(0.2, 1.0)
        self.assertEqual(st.reason, Reason.DEPTH_INTERVAL_AT_ORIGIN)
        self.assertIsNone(st.u)


class IntervalBilinearTests(unittest.TestCase):
    """near/far 两层分别双线性。"""

    def test_analytic_ramps_both_layers(self):
        # near = 2 + 0.1c + 0.2r；far = near + 1.5（常量厚度）。
        w = h = 5
        near = [[2.0 + 0.1 * c + 0.2 * r for c in range(w)] for r in range(h)]
        far = [[v + 1.5 for v in row] for row in near]
        view = View(camera(), DepthMap(near, far=far))
        u, v = 1.3, 2.4
        d_near = 2.0 + 0.1 * u + 0.2 * v          # 2.61
        d_far = d_near + 1.5                       # 4.11
        # 端点用 1e-9 紧容差吸收双线性角点常量的 ~1e-16 级舍入，
        # 其余探针与板边界距离 >0.6，裁定不受影响。
        for z, want in (
            (d_near, Visibility.VISIBLE),
            (d_far, Visibility.VISIBLE),
            (3.3, Visibility.VISIBLE),
            (2.0, Visibility.IN_FRONT),
            (5.0, Visibility.OCCLUDED),
        ):
            st = classify_view(view, ref.world_point_at_pixel(
                K0, IDENT, ZERO, u, v, z), abs_tol=1e-9, rel_tol=0.0)
            self.assertEqual(st.visibility, want, msg=f"z={z}")
        st = classify_view(view, ref.world_point_at_pixel(
            K0, IDENT, ZERO, u, v, d_near), abs_tol=1e-9, rel_tol=0.0)
        self.assertAlmostEqual(st.sampled_depth, 2.61, places=12)
        self.assertAlmostEqual(st.sampled_depth_far, 4.11, places=12)

    def test_separate_endpoint_interpolation_thickness(self):
        # 列 1 的像素为薄表面 [5,5]，列 2 为厚表面 [6,10]；行 1、2 相同。
        # u=1.5 权重各半：端点分别插值 -> [5.5, 7.5]（不是并集包络到 10）。
        w = h = 4
        near = [[5.0] * w for _ in range(h)]
        far = [[5.0] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if c == 2:
                    near[r][c] = 6.0
                    far[r][c] = 10.0
        view = View(camera(), DepthMap(near, far=far))
        st = classify_view(view, point_at(1.5, 1.0, 7.5),
                           abs_tol=0.0, rel_tol=0.0)
        self.assertEqual(st.visibility, Visibility.VISIBLE)
        self.assertAlmostEqual(st.sampled_depth, 5.5, places=12)
        self.assertAlmostEqual(st.sampled_depth_far, 7.5, places=12)
        # 7.6 已越出插值得到的 far=7.5 -> 确定 occluded。
        st2 = classify_view(view, point_at(1.5, 1.0, 7.6),
                            abs_tol=0.0, rel_tol=0.0)
        self.assertEqual(st2.visibility, Visibility.OCCLUDED)


class IntervalReferenceCrossCheckTests(unittest.TestCase):
    """小规模枚举：与独立参考逐字段交叉核对。"""

    def test_enumerated_probes(self):
        w = h = 6
        near = [[3.0 + 0.07 * c + 0.11 * r for c in range(w)] for r in range(h)]
        far = [[v + 0.5 + 0.03 * c for (c, v) in enumerate(row)]
               for row in near]
        mask = [[False] * w for _ in range(h)]
        mask[1][2] = True
        view = View(camera(), DepthMap(near, mask, far))
        # 像素网格避开整数边界（往返投影有 ~1e-15 舍入，整数像素边缘
        # 的缺邻域判定已由旧测试的手算常量固定，不在这里赌末位）；
        # 仍扫到图外负坐标与图内亚像素、mask 邻点。
        for i in range(-1, 17):
            for j in range(-1, 19):
                u0 = 0.31 * i - 0.42   # -0.73 .. 4.54，离整数 >=0.02
                v0 = 0.27 * j - 0.30   # -0.57 .. 4.29，离整数 >=0.02
                for z in (0.5, 2.0, 5.0, 8.0):
                    P = ref.world_point_at_pixel(K0, IDENT, ZERO, u0, v0, z)
                    for eps in (0.0, 0.5, 1.5):
                        got = classify_view(
                            view, P, abs_tol=0.02, rel_tol=0.01,
                            depth_eps=eps,
                        )
                        exp = ref.classify_interval(
                            K0, IDENT, ZERO, near, far, mask, P,
                            eps, 0.02, 0.01,
                        )
                        self.assertEqual(
                            got.visibility.value, exp["visibility"],
                            msg=f"u={u0} v={v0} z={z} eps={eps}",
                        )
                        self.assertEqual(
                            got.reason.value, exp["reason"],
                            msg=f"u={u0} v={v0} z={z} eps={eps}",
                        )
                        if got.sampled_depth is not None:
                            self.assertAlmostEqual(
                                got.sampled_depth, exp["d_lo"], places=12)
                            self.assertAlmostEqual(
                                got.sampled_depth_far, exp["d_hi"], places=12)


class MultiViewIntervalTests(unittest.TestCase):
    """多视图：逐视图原因保留，不融合成二值。"""

    def test_per_view_reasons_preserved(self):
        # 探针 (0.5,0.5,6) -> 像素约 (0.83,0.83)（四邻点齐全的图内点）。
        # 三视图表面：
        # 视图0 [5,5] 薄；视图1 [4,7] 厚；视图2 [2,3] 远在点前。
        views = [
            View(camera(), slab(4, 4, 5.0, 5.0)),
            View(camera(), slab(4, 4, 4.0, 7.0)),
            View(camera(), slab(4, 4, 2.0, 3.0)),
        ]
        # 点 z=6，eps=1.5 -> 区间 [4.5,7.5]。
        results = classify_points(
            views, [(0.5, 0.5, 6.0)], depth_eps=1.5,
            abs_tol=0.0, rel_tol=0.0,
        )
        row = results[0]
        # 视图0：[4.5,7.5] 越过表面 {5} 的两侧 -> uncertain。
        self.assertEqual(row[0].visibility, Visibility.UNCERTAIN)
        self.assertEqual(row[0].reason, Reason.DEPTH_INTERVAL_OVERLAP)
        # 视图1：[4.5,7.5] 越出板 [4,7] 两侧 -> uncertain。
        self.assertEqual(row[1].visibility, Visibility.UNCERTAIN)
        # 视图2：[4.5,7.5] 整体在 [2,3] 之后 -> 确定 occluded。
        self.assertEqual(row[2].visibility, Visibility.OCCLUDED)
        self.assertEqual(row[2].reason, Reason.NONE)
        # 没有任何视图被融合翻转；可见数为 0。
        self.assertEqual(count_visible_views(row), 0)
        self.assertEqual(indices_visible_in_at_least_k(results, 1), [])

    def test_mixed_visible_uncertain_unknown_aggregation(self):
        slab_view = View(camera(), slab(4, 4, 5.0, 7.0))
        plane_view = View(camera(), plane(4, 4, 6.0))
        mask_all = [[True] * 4 for _ in range(4)]
        masked_view = View(camera(), slab(4, 4, 5.0, 7.0, mask_all))
        views = [slab_view, plane_view, masked_view]
        P = (0.5, 0.5, 6.0)  # 像素约 (0.83,0.83)
        # eps=0：视图0 板内 visible；视图1 表面 6 visible；视图2 unknown。
        results = classify_points(views, [P], abs_tol=0.0, rel_tol=0.0)
        self.assertEqual(
            [s.visibility for s in results[0]],
            [Visibility.VISIBLE, Visibility.VISIBLE, Visibility.UNKNOWN],
        )
        self.assertEqual(results[0][2].reason, Reason.MASKED_PIXEL)
        self.assertEqual(indices_visible_in_at_least_k(results, 2), [0])
        self.assertEqual(indices_visible_in_at_least_k(results, 3), [])
        # eps=1.2：点 [4.8,7.2] 对两视图都只重叠 -> 无 visible 计数。
        results_u = classify_points(
            views, [P], abs_tol=0.0, rel_tol=0.0, depth_eps=1.2)
        self.assertEqual(results_u[0][0].visibility, Visibility.UNCERTAIN)
        self.assertEqual(results_u[0][1].visibility, Visibility.UNCERTAIN)
        self.assertEqual(count_visible_views(results_u[0]), 0)


class IntervalConstructionTests(unittest.TestCase):
    def test_from_intervals_and_is_interval(self):
        dm = DepthMap.from_intervals([[1.0, 2.0]], [[1.5, 2.0]])
        self.assertAlmostEqual(dm.values[0][0], 1.0)
        self.assertAlmostEqual(dm.far[0][0], 1.5)
        self.assertTrue(dm.is_interval)
        # 全退化 -> is_interval False，far 与 values 相同。
        thin = DepthMap([[2.0, 2.0]])
        self.assertFalse(thin.is_interval)
        self.assertEqual(thin.far, ((2.0, 2.0),))

    def test_far_must_be_ge_near(self):
        with self.assertRaises(InvalidDepthMapError) as cm:
            DepthMap([[2.0, 3.0]], far=[[2.0, 2.9]])
        self.assertIn("far[0][1]", str(cm.exception))

    def test_far_shape_and_finiteness(self):
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([[1.0, 1.0]], far=[[1.0]])            # 形状不一致
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([[1.0]], far=[[float("nan")]])
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([[1.0]], far=[[0.0]])
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([[2.0]], far=[[-1.0]])

    def test_far_defensive_copy(self):
        far_src = [[7.0, 7.0]]
        dm = DepthMap([[5.0, 5.0]], far=far_src)
        far_src[0][0] = 99.0
        self.assertEqual(dm.far[0][0], 7.0)

    def test_depth_eps_validation(self):
        view = View(camera(), plane(2, 2, 5.0))
        P = (0.0, 0.0, 5.0)
        for bad in (-1e-12, float("nan"), float("inf"), True, "0.1"):
            with self.assertRaises(InvalidParameterError):
                classify_view(view, P, depth_eps=bad)

    def test_batch_validates_all_before_computing(self):
        good = View(camera(), plane(2, 2, 5.0))
        # 非法点：整体失败，无部分结果。
        with self.assertRaises(Exception):
            classify_points([good],
                            [(0.0, 0.0, 5.0), (0.0, float("nan"), 5.0)],
                            depth_eps=0.1)
        # 非法 depth_eps 同样先于任何计算被拒。
        with self.assertRaises(InvalidParameterError):
            classify_points([good], [(0.0, 0.0, 5.0)], depth_eps=-1.0)


class BackwardCompatTests(unittest.TestCase):
    """旧单点调用在默认 depth_eps=0 下的字段语义。"""

    def test_degenerate_fields_equal_old_semantics(self):
        view = View(camera(), slab(4, 4, 5.0, 5.0))
        st = classify_view(view, (0.75, 0.75, 5.0),
                           abs_tol=0.0, rel_tol=0.0)
        self.assertEqual(st.visibility, Visibility.VISIBLE)
        self.assertEqual(st.sampled_depth, st.sampled_depth_far)
        self.assertEqual(st.projected_z_low, st.projected_z_high)
        self.assertEqual(st.projected_z_low, 5.0)
        # 默认构造的 DepthMap 也有 far 层且等于 values。
        dm = plane(3, 3, 4.0)
        self.assertEqual(dm.far, dm.values)

    def test_old_constructor_two_args_still_works(self):
        dm = DepthMap([[1.0, 2.0]], [[False, True]])
        self.assertEqual(dm.disabled, ((False, True),))
        self.assertEqual(dm.far, ((1.0, 2.0),))


if __name__ == "__main__":
    unittest.main()
