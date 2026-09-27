"""核心裁定测试。

期望值的来源：

1. 像素中心手工推算的常量（深度图值、投影像素、世界点由参考模块
   独立逆投影生成，断言常量状态）；
2. ``tests/reference.py`` 的独立实现交叉核对（不导入被测核心）；
3. 往返一致性仅作为补充检查。
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import reference as ref  # noqa: E402

from visibility import (  # noqa: E402
    Camera,
    CameraModel,
    DepthMap,
    InvalidCameraError,
    InvalidDepthMapError,
    InvalidParameterError,
    InvalidPointError,
    Reason,
    OrthographicCamera,
    View,
    Visibility,
    classify_points,
    classify_view,
    count_visible_views,
    indices_visible_in_at_least_k,
)

# 简单相机：fx=fy=10, 主点 (0,0)，无旋转无平移。深度即世界 z。
K0 = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 1.0))
IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ZERO = (0.0, 0.0, 0.0)


def make_camera(K=K0, R=IDENT, t=ZERO):
    return Camera(K=K, R=R, t=t)


def plane_depth(w, h, d):
    return DepthMap([[d for _ in range(w)] for _ in range(h)])


class ProjectionAndPlaneTests(unittest.TestCase):
    """解析平面深度图 + 直接相机投影参考。"""

    def test_pixel_center_exact_constants(self):
        # 4x4 平面，表面深度 5。点 (1.0, 0.5, 5) -> (u,v)=(2,1) 像素中心。
        dm = plane_depth(4, 4, 5.0)
        view = View(make_camera(), dm)
        st = classify_view(view, (1.0, 0.5, 5.0), abs_tol=0.0, rel_tol=0.0)
        self.assertEqual(st.visibility, Visibility.VISIBLE)
        self.assertEqual(st.reason, Reason.NONE)
        self.assertEqual((st.u, st.v), (2.0, 1.0))
        self.assertEqual(st.sampled_depth, 5.0)
        self.assertEqual(st.projected_z, 5.0)

    def test_plane_occluded_infront_by_hand(self):
        dm = plane_depth(4, 4, 5.0)
        view = View(make_camera(), dm)
        # 点在表面后方 1 个单位 -> occluded（投影仍在图内）。
        st_back = classify_view(view, (1.0, 0.5, 6.0), abs_tol=0.0, rel_tol=0.0)
        self.assertEqual(st_back.visibility, Visibility.OCCLUDED)
        self.assertEqual(st_back.projected_z, 6.0)
        self.assertEqual(st_back.sampled_depth, 5.0)
        # 点在表面前方 -> in_front。
        st_front = classify_view(view, (1.0, 0.5, 4.0), abs_tol=0.0, rel_tol=0.0)
        self.assertEqual(st_front.visibility, Visibility.IN_FRONT)

    def test_tolerance_band(self):
        dm = plane_depth(4, 4, 10.0)
        view = View(make_camera(), dm)
        # T = abs 0.5 + rel 0.01*10 = 0.6。z=10.6 恰在边界 -> visible。
        st_edge = classify_view(view, (1.0, 0.5, 10.6), abs_tol=0.5, rel_tol=0.01)
        self.assertEqual(st_edge.visibility, Visibility.VISIBLE)
        # 超出 1e-12 -> occluded，不用宽容差掩盖边界。
        st_over = classify_view(
            view, (1.0, 0.5, 10.6 + 1e-9), abs_tol=0.5, rel_tol=0.01
        )
        self.assertEqual(st_over.visibility, Visibility.OCCLUDED)
        st_under = classify_view(
            view, (1.0, 0.5, 9.4 - 1e-9), abs_tol=0.5, rel_tol=0.01
        )
        self.assertEqual(st_under.visibility, Visibility.IN_FRONT)

    def test_bilinear_weights_analytic_ramp(self):
        # 深度从 (0,0)=0(实际用1) 起做解析斜面 d = 2 + 0.1*c + 0.2*r，
        # 在非整数像素 (1.3, 2.4) 处双线性值应等于该仿射函数本身。
        w = h = 5
        vals = [[2.0 + 0.1 * c + 0.2 * r for c in range(w)] for r in range(h)]
        dm = DepthMap(vals)
        view = View(make_camera(), dm)
        u, v = 1.3, 2.4
        # z 取为该点仿射深度；用逆投影反解世界点（独立参考）。
        z = 2.0 + 0.1 * u + 0.2 * v
        P = ref.world_point_at_pixel(K0, IDENT, ZERO, u, v, z)
        st = classify_view(view, P, abs_tol=0.0, rel_tol=0.0)
        self.assertEqual(st.visibility, Visibility.VISIBLE)
        self.assertAlmostEqual(st.sampled_depth, z, places=12)

    def test_reference_cross_check_many_points(self):
        w = h = 6
        vals = [[3.0 + 0.07 * c + 0.11 * r for c in range(w)] for r in range(h)]
        mask = [[False for _ in range(w)] for _ in range(h)]
        mask[1][2] = True
        dm = DepthMap(vals, mask)
        view = View(make_camera(), dm)
        # 覆盖像素中心、亚像素、边缘、图外、mask 邻点、相机后。
        probes = [
            (0.3 * i - 0.4, 0.27 * j - 0.3, z)
            for i in range(-1, 9)
            for j in range(-1, 9)
            for z in (2.0, 5.0, 8.0)
        ]
        for (u0, v0, z) in probes:
            P = ref.world_point_at_pixel(K0, IDENT, ZERO, u0, v0, z)
            got = classify_view(view, P, abs_tol=0.02, rel_tol=0.01)
            exp = ref.classify(K0, IDENT, ZERO, vals, mask, P, 0.02, 0.01)
            self.assertEqual(
                got.visibility.value, exp["visibility"],
                msg=f"probe u={u0} v={v0} z={z}",
            )
            self.assertEqual(got.reason.value, exp["reason"])


class UnknownReasonTests(unittest.TestCase):
    def test_behind_camera_has_priority(self):
        # z <= 0：即使其他什么都不查，也必须先判 z，返回 behind_camera。
        dm = plane_depth(4, 4, 5.0)
        view = View(make_camera(), dm)
        for P in ((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (100.0, 100.0, -5.0)):
            st = classify_view(view, P)
            self.assertEqual(st.visibility, Visibility.UNKNOWN)
            self.assertEqual(st.reason, Reason.BEHIND_CAMERA)
            self.assertIsNone(st.sampled_depth)
            self.assertIsNone(st.u)
            self.assertIsNone(st.v)
            self.assertLessEqual(st.projected_z, 0.0)

    def test_translated_camera_behind(self):
        # cz = z_world + t_z。t_z=10 时世界点 z=-15 -> cz=-5（相机后方）。
        cam = Camera(K=K0, R=IDENT, t=(0.0, 0.0, 10.0))
        view = View(cam, plane_depth(4, 4, 2.0))
        st = classify_view(view, (0.0, 0.0, -15.0))
        self.assertEqual(st.reason, Reason.BEHIND_CAMERA)
        self.assertEqual(st.projected_z, -5.0)

    def test_out_of_bounds(self):
        dm = plane_depth(4, 4, 5.0)
        view = View(make_camera(), dm)
        # u=5 -> 列 5 越界（宽 4）。
        P = ref.world_point_at_pixel(K0, IDENT, ZERO, 5.0, 1.0, 5.0)
        st = classify_view(view, P, abs_tol=0.0, rel_tol=0.0)
        self.assertEqual((st.visibility, st.reason),
                         (Visibility.UNKNOWN, Reason.OUT_OF_BOUNDS))
        # 负像素坐标同样越界。
        Pn = ref.world_point_at_pixel(K0, IDENT, ZERO, -0.5, 0.0, 5.0)
        stn = classify_view(view, Pn)
        self.assertEqual(stn.reason, Reason.OUT_OF_BOUNDS)

    def test_missing_neighbor_at_far_edge(self):
        # 投影到右下角像素中心 (3,3)：stencil 需要 (4,*) / (*,4)，缺失。
        dm = plane_depth(4, 4, 5.0)
        view = View(make_camera(), dm)
        P = ref.world_point_at_pixel(K0, IDENT, ZERO, 3.0, 3.0, 5.0)
        st = classify_view(view, P, abs_tol=0.0, rel_tol=0.0)
        self.assertEqual((st.visibility, st.reason),
                         (Visibility.UNKNOWN, Reason.MISSING_NEIGHBOR))
        # 1x1 深度图：任何投影都缺邻域。
        tiny = View(make_camera(), plane_depth(1, 1, 5.0))
        P0 = ref.world_point_at_pixel(K0, IDENT, ZERO, 0.0, 0.0, 5.0)
        st0 = classify_view(tiny, P0)
        self.assertEqual(st0.reason, Reason.MISSING_NEIGHBOR)

    def test_occlusion_edge_missing_vs_valid(self):
        # 遮挡边缘语义：同一投影行，左侧四邻点齐全（有效 -> occluded），
        # 右移到整数边界列 3 后 stencil 需要列 4 -> unknown，绝不用 0 补。
        # 直接构造使投影精确的点（z=10，x=u*z/fx），避免逆投影往返误差。
        dm = plane_depth(4, 4, 5.0)
        view = View(make_camera(), dm)
        st = classify_view(view, (2.999, 1.0, 10.0), abs_tol=0.0, rel_tol=0.0)
        self.assertEqual(st.visibility, Visibility.OCCLUDED)
        st2 = classify_view(view, (3.0, 1.0, 10.0), abs_tol=0.0, rel_tol=0.0)
        self.assertEqual(st2.visibility, Visibility.UNKNOWN)
        self.assertEqual(st2.reason, Reason.MISSING_NEIGHBOR)

    def test_masked_any_corner_unknown(self):
        w = h = 4
        mask = [[False for _ in range(w)] for _ in range(h)]
        mask[2][3] = True  # 投影 (2.5,1.5) 的四个邻点含 (r=2,c=3)
        dm = DepthMap([[5.0] * w for _ in range(h)], mask)
        view = View(make_camera(), dm)
        P = ref.world_point_at_pixel(K0, IDENT, ZERO, 2.5, 1.5, 5.0)
        st = classify_view(view, P, abs_tol=0.0, rel_tol=0.0)
        self.assertEqual((st.visibility, st.reason),
                         (Visibility.UNKNOWN, Reason.MASKED_PIXEL))

    def test_mask_false_default_all_enabled(self):
        dm = DepthMap([[2.0, 2.0], [2.0, 2.0]])
        self.assertFalse(any(any(r) for r in dm.disabled))


class MultiViewAggregationTests(unittest.TestCase):
    def _views(self, depths, masks=None):
        views = []
        for i, d in enumerate(depths):
            m = None if masks is None else masks[i]
            views.append(View(make_camera(), DepthMap(
                [[d] * 4 for _ in range(4)], m)))
        return views

    def test_per_view_states_preserved(self):
        # 点投影到像素 (1,1)：三相机表面深度 4/5/8，点 z=5。
        views = self._views([4.0, 5.0, 8.0])
        P = (0.5, 0.5, 5.0)
        results = classify_points(views, [P], abs_tol=0.0, rel_tol=0.0)
        states = [s.visibility for s in results[0]]
        self.assertEqual(
            states,
            [Visibility.OCCLUDED, Visibility.VISIBLE, Visibility.IN_FRONT],
        )
        self.assertEqual(count_visible_views(results[0]), 1)
        # 多数票不能翻转 occluded：k=2 时该点不入选。
        self.assertEqual(indices_visible_in_at_least_k(results, 2), [])
        self.assertEqual(indices_visible_in_at_least_k(results, 1), [0])

    def test_aggregation_matrix_and_k(self):
        views = self._views([5.0, 5.0, 7.0])
        points = [(0.5, 0.5, 5.0), (0.5, 0.5, 7.0), (0.5, 0.5, 5.0)]
        results = classify_points(views, points, abs_tol=0.0, rel_tol=0.0)
        # 点0: 前两机 visible；点1: 仅第三机 visible；点2: 同点0。
        self.assertEqual(count_visible_views(results[0]), 2)
        self.assertEqual(count_visible_views(results[1]), 1)
        self.assertEqual(indices_visible_in_at_least_k(results, 2), [0, 2])
        self.assertEqual(indices_visible_in_at_least_k(results, 1), [0, 1, 2])

    def test_unknown_not_counted(self):
        # 第三机给一个全禁用 mask -> 该视图恒 unknown，不影响可见计数。
        mask_all = [[True] * 4 for _ in range(4)]
        views = self._views([5.0, 5.0, 5.0], masks=[None, None, mask_all])
        results = classify_points(views, [(0.5, 0.5, 5.0)])
        self.assertEqual(results[0][2].reason, Reason.MASKED_PIXEL)
        self.assertEqual(indices_visible_in_at_least_k(results, 2), [0])
        self.assertEqual(indices_visible_in_at_least_k(results, 3), [])

    def test_rotated_camera_consistency(self):
        # 绕 y 轴旋转的相机：独立参考给定期望，结果需一致（补充性往返）。
        a = math.radians(20.0)
        ca, sa = math.cos(a), math.sin(a)
        R = ((ca, 0.0, sa), (0.0, 1.0, 0.0), (-sa, 0.0, ca))
        t = (0.1, -0.2, 0.3)
        w = h = 8
        # 以相机前方一块平面构造深度：取一批世界点投影后赋常量深度。
        z_surface = 6.0
        vals = [[z_surface for _ in range(w)] for _ in range(h)]
        cam = Camera(K=K0, R=R, t=t)
        view = View(cam, DepthMap(vals))
        for du in (-2.0, 0.0, 1.37):
            P = ref.world_point_at_pixel(K0, R, t, 3.0 + du, 3.0, z_surface)
            st = classify_view(view, P)
            exp = ref.classify(K0, R, t, vals,
                               [[False] * w for _ in range(h)], P, 1e-6, 1e-6)
            self.assertEqual(st.visibility.value, exp["visibility"])
            self.assertAlmostEqual(st.projected_z, z_surface, places=9)
        # 平面后方的点应判 occluded（旋转相机下同样成立）。
        P_back = ref.world_point_at_pixel(K0, R, t, 3.0, 3.0, z_surface + 1.5)
        self.assertEqual(classify_view(view, P_back).visibility,
                         Visibility.OCCLUDED)


class InputValidationTests(unittest.TestCase):
    def test_reject_nan_inf(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidPointError):
                classify_view(View(make_camera(), plane_depth(2, 2, 5.0)),
                              (0.0, 0.0, bad))
            with self.assertRaises(InvalidDepthMapError):
                DepthMap([[bad]])
            with self.assertRaises(InvalidCameraError):
                Camera(K=((bad, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                       R=IDENT, t=ZERO)

    def test_reject_bool_as_int_param(self):
        view = View(make_camera(), plane_depth(2, 2, 5.0))
        results = classify_points([view], [(0.0, 0.0, 5.0)])
        with self.assertRaises(InvalidParameterError):
            indices_visible_in_at_least_k(results, True)  # type: ignore[arg-type]
        with self.assertRaises(InvalidParameterError):
            classify_view(view, (0.0, 0.0, 5.0), abs_tol=True)  # type: ignore[arg-type]

    def test_depth_shape_and_rectangularity(self):
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([[1.0, 1.0], [1.0]])  # 非矩形
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([])  # 0 行
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([[1.0] * 33 for _ in range(2)])  # 超宽上限
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([[1.0] * 2 for _ in range(33)])  # 超高上限
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([[0.0]])  # 深度必须 > 0
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([[-1.0]])
        # mask 形状必须一致
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([[1.0, 1.0]], [[True]])
        with self.assertRaises(InvalidDepthMapError):
            DepthMap([[1.0, 1.0]], [[True, 0]])  # 0 不是 bool

    def test_camera_structure(self):
        with self.assertRaises(InvalidCameraError):
            Camera(K=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                   R=IDENT, t=ZERO)  # fx=0
        with self.assertRaises(InvalidCameraError):
            Camera(K=((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
                   R=IDENT, t=ZERO)  # fy<0
        with self.assertRaises(InvalidCameraError):
            Camera(K=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 2.0)),
                   R=IDENT, t=ZERO)  # K[2][2]!=1
        with self.assertRaises(InvalidCameraError):
            Camera(K=K0, R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), t=ZERO)
        with self.assertRaises(InvalidCameraError):
            Camera(K=K0, R=IDENT, t=(0.0, 0.0))

    def test_k_range(self):
        view = View(make_camera(), plane_depth(2, 2, 5.0))
        results = classify_points([view], [(0.0, 0.0, 5.0)])
        with self.assertRaises(InvalidParameterError):
            indices_visible_in_at_least_k(results, 0)
        with self.assertRaises(InvalidParameterError):
            indices_visible_in_at_least_k(results, 2)  # 超过视图数
        with self.assertRaises(InvalidParameterError):
            indices_visible_in_at_least_k(results, 1.0)

    def test_negative_tolerance_rejected(self):
        view = View(make_camera(), plane_depth(2, 2, 5.0))
        with self.assertRaises(InvalidParameterError):
            classify_view(view, (0.0, 0.0, 5.0), abs_tol=-1e-9)
        with self.assertRaises(InvalidParameterError):
            classify_view(view, (0.0, 0.0, 5.0), rel_tol=-1e-9)

    def test_locatable_error_path(self):
        try:
            DepthMap([[1.0, 1.0], [1.0, float("nan")]])
        except InvalidDepthMapError as e:
            self.assertIn("values[1][1]", str(e))
        else:
            self.fail("expected InvalidDepthMapError")

    def test_defensive_copy(self):
        src = [[5.0, 5.0], [5.0, 5.0]]
        dm = DepthMap(src)
        src[0][0] = 99.0
        self.assertEqual(dm.values[0][0], 5.0)
        mask = [[False, False], [False, False]]
        dm2 = DepthMap([[5.0, 5.0], [5.0, 5.0]], mask)
        mask[0][0] = True
        self.assertFalse(dm2.disabled[0][0])

    def test_batch_validates_before_computing(self):
        good = View(make_camera(), plane_depth(2, 2, 5.0))
        # 非法点混在批次中 -> 整体报错，且不返回部分结果。
        with self.assertRaises(InvalidPointError):
            classify_points([good], [(0.0, 0.0, 5.0), (0.0, float("nan"), 5.0)])
        with self.assertRaises(InvalidParameterError):
            classify_points([good], [(0.0, 0.0, 5.0)], abs_tol=float("inf"))


class SkewIntrinsicTests(unittest.TestCase):
    def test_skew_K01_cross_check(self):
        # 斜切内参：u = fx*cx/z + skew*cy/z + cx0。独立参考给定期望。
        skew = 2.0
        Ks = ((10.0, skew, 1.5), (0.0, 8.0, 2.5), (0.0, 0.0, 1.0))
        w = h = 6
        vals = [[4.0 + 0.05 * c + 0.03 * r for c in range(w)] for r in range(h)]
        mask = [[False] * w for _ in range(h)]
        view = View(make_camera(K=Ks), DepthMap(vals, mask))
        for (u0, v0, z) in [(1.2, 1.3, 4.5), (3.7, 2.2, 5.5), (0.5, 0.5, 4.0)]:
            P = ref.world_point_at_pixel(Ks, IDENT, ZERO, u0, v0, z)
            got = classify_view(view, P, abs_tol=0.01, rel_tol=0.01)
            exp = ref.classify(Ks, IDENT, ZERO, vals, mask, P, 0.01, 0.01)
            self.assertEqual(got.visibility.value, exp["visibility"])
            self.assertAlmostEqual(got.u, u0, places=9)
            self.assertAlmostEqual(got.v, v0, places=9)


class BoundarySizeTests(unittest.TestCase):
    def test_max_size_accepted(self):
        dm = DepthMap([[3.0] * 32 for _ in range(32)])
        self.assertEqual((dm.height, dm.width), (32, 32))
        view = View(make_camera(), dm)
        # 投影在 (0,0) 像素中心：缺行/列 1？列1存在（宽32），有效。
        st = classify_view(view, (0.0, 0.0, 3.0), abs_tol=0.0, rel_tol=0.0)
        self.assertEqual(st.visibility, Visibility.VISIBLE)


class OrthographicCameraTests(unittest.TestCase):
    """正交相机：投影不除 z；深度语义与 unknown 规则与透视一致。"""

    # 正交内参：fx=fy=2 像素/世界单位，主点 (1,1)，即 u=2*cx+1, v=2*cy+1。
    KO = ((2.0, 0.0, 1.0), (0.0, 2.0, 1.0), (0.0, 0.0, 1.0))

    def _view(self, w=4, h=4, d=5.0, mask=None):
        cam = OrthographicCamera(K=self.KO, R=IDENT, t=ZERO)
        vals = [[d for _ in range(w)] for _ in range(h)]
        return View(cam, DepthMap(vals, mask))

    def test_projection_has_no_perspective_divide(self):
        # 同一相机 xy=(0.75, 0.25) 在任意深度都投到像素 (2.5, 1.5)。
        view = self._view()
        for z in (1.0, 5.0, 50.0):
            st = classify_view(view, (0.75, 0.25, z),
                               abs_tol=0.0, rel_tol=0.0)
            self.assertEqual((st.u, st.v), (2.5, 1.5), msg=f"z={z}")
        # z=5 恰在表面 -> visible；z=1/50 与深度无关，只做深度比较。
        self.assertEqual(
            classify_view(view, (0.75, 0.25, 5.0)).visibility,
            Visibility.VISIBLE,
        )
        self.assertEqual(
            classify_view(view, (0.75, 0.25, 6.0)).visibility,
            Visibility.OCCLUDED,
        )
        self.assertEqual(
            classify_view(view, (0.75, 0.25, 4.0)).visibility,
            Visibility.IN_FRONT,
        )

    def test_depth_still_camera_z(self):
        # 平移相机：深度 cz = z_world + t_z，与透视一致。
        cam = OrthographicCamera(K=self.KO, R=IDENT, t=(0.0, 0.0, 10.0))
        view = View(cam, plane_depth(4, 4, 5.0))
        st = classify_view(view, (0.75, 0.25, -5.0))  # cz = 5
        self.assertEqual(st.projected_z, 5.0)
        self.assertEqual(st.visibility, Visibility.VISIBLE)
        st_behind = classify_view(view, (0.75, 0.25, -15.0))  # cz = -5
        self.assertEqual(st_behind.visibility, Visibility.UNKNOWN)
        self.assertEqual(st_behind.reason, Reason.BEHIND_CAMERA)

    def test_boundary_and_missing_neighbor_unknown(self):
        # cx 使 u 恰为 4（=宽）-> out_of_bounds；u=3 像素中心 -> 缺邻域。
        view = self._view()
        st = classify_view(view, (1.5, 0.25, 5.0), abs_tol=0.0, rel_tol=0.0)
        self.assertEqual((st.visibility, st.reason),
                         (Visibility.UNKNOWN, Reason.OUT_OF_BOUNDS))
        st_edge = classify_view(view, (1.0, 1.0, 5.0),
                                abs_tol=0.0, rel_tol=0.0)
        self.assertEqual((st_edge.visibility, st_edge.reason),
                         (Visibility.UNKNOWN, Reason.MISSING_NEIGHBOR))

    def test_masked_pixel_unknown(self):
        w = h = 4
        mask = [[False for _ in range(w)] for _ in range(h)]
        mask[2][3] = True  # 投影 (2.5,1.5) 的四邻点之一
        view = self._view(mask=mask)
        st = classify_view(view, (0.75, 0.25, 5.0),
                           abs_tol=0.0, rel_tol=0.0)
        self.assertEqual((st.visibility, st.reason),
                         (Visibility.UNKNOWN, Reason.MASKED_PIXEL))

    def test_skew_and_rotation_cross_check(self):
        # 带斜切与旋转的正交相机：与独立参考逐点核对。
        Ks = ((2.0, 0.5, 1.5), (0.0, 1.5, 0.5), (0.0, 0.0, 1.0))
        a = math.radians(15.0)
        ca, sa = math.cos(a), math.sin(a)
        R = ((ca, 0.0, sa), (0.0, 1.0, 0.0), (-sa, 0.0, ca))
        t = (0.2, -0.1, 0.3)
        w = h = 8
        vals = [[5.0 + 0.03 * c + 0.02 * r for c in range(w)] for r in range(h)]
        mask = [[False] * w for _ in range(h)]
        cam = OrthographicCamera(K=Ks, R=R, t=t)
        view = View(cam, DepthMap(vals, mask))
        for (u0, v0, z) in [(1.2, 1.3, 4.5), (3.7, 2.2, 5.5),
                            (0.5, 0.5, 4.0), (7.0, 7.0, 6.0)]:
            P = ref.world_point_at_pixel(Ks, R, t, u0, v0, z,
                                         model="orthographic")
            got = classify_view(view, P, abs_tol=0.01, rel_tol=0.01)
            exp = ref.classify(Ks, R, t, vals, mask, P, 0.01, 0.01,
                               model="orthographic")
            self.assertEqual(got.visibility.value, exp["visibility"],
                             msg=f"u={u0} v={v0} z={z}")
            if got.u is not None:
                self.assertAlmostEqual(got.u, u0, places=9)
                self.assertAlmostEqual(got.v, v0, places=9)

    def test_mixed_views_are_independent(self):
        # 同一点：正交视图与透视视图各自投影、互不影响，聚合规则不变。
        ortho = self._view(d=5.0)
        # 透视相机：同一点 (0.5,0.5,5) -> (1,1)，表面深度 4 -> occluded。
        persp = View(make_camera(), plane_depth(4, 4, 4.0))
        # 第三透视视图全禁用 -> unknown。
        mask_all = [[True] * 4 for _ in range(4)]
        masked = View(make_camera(), DepthMap(
            [[4.0] * 4 for _ in range(4)], mask_all))
        P = (0.5, 0.5, 5.0)
        results = classify_points([ortho, persp, masked], [P],
                                  abs_tol=0.0, rel_tol=0.0)
        states = [s.visibility for s in results[0]]
        # 正交下 u=2*0.5+1=2, v=2；深度 5 对表面 5 -> visible。
        self.assertEqual(states[0], Visibility.VISIBLE)
        self.assertEqual(states[1], Visibility.OCCLUDED)
        self.assertEqual(states[2], Visibility.UNKNOWN)
        # 改不改相机类型都不影响其他视图：只有 1 个 visible。
        self.assertEqual(count_visible_views(results[0]), 1)
        self.assertEqual(indices_visible_in_at_least_k(results, 2), [])
        self.assertEqual(indices_visible_in_at_least_k(results, 1), [0])

    def test_model_field_and_invalid(self):
        cam = Camera(K=K0, R=IDENT, t=ZERO)
        self.assertIs(cam.model, CameraModel.PERSPECTIVE)
        ocam = OrthographicCamera(K=K0, R=IDENT, t=ZERO)
        self.assertIs(ocam.model, CameraModel.ORTHOGRAPHIC)
        self.assertIsInstance(ocam, Camera)
        # 显式构造正交相机与子类字段完全一致。
        ocam2 = Camera(K=K0, R=IDENT, t=ZERO,
                       model=CameraModel.ORTHOGRAPHIC)
        self.assertEqual((ocam.K, ocam.R, ocam.t, ocam.model),
                         (ocam2.K, ocam2.R, ocam2.t, ocam2.model))
        with self.assertRaises(InvalidCameraError):
            Camera(K=K0, R=IDENT, t=ZERO, model="orthographic")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
