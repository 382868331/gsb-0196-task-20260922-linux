"""深度区间裁定测试。

期望值来源：

1. 常量深度平面上的**手算闭区间比较**（容差带端点为手工常量，
   不调用被测实现生成）；
2. ``tests/reference.py`` 的独立区间参考 ``interval_classify``
   （不导入被测包）在斜面深度图上的交叉核对；
3. 零宽度区间 ``[z,z]`` 与标量裁定逐位一致（同一条像素射线）。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import reference as ref  # noqa: E402

from visibility import (  # noqa: E402
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
    count_visible_views,
    indices_visible_in_at_least_k,
)

K0 = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 1.0))
IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ZERO = (0.0, 0.0, 0.0)


def plane_view(d, w=6, h=6, mask=None):
    cam = Camera(K=K0, R=IDENT, t=ZERO)
    vals = [[d for _ in range(w)] for _ in range(h)]
    return View(cam, DepthMap(vals, mask))


def q(u, v, lo, hi):
    return IntervalQuery(u=u, v=v, interval=DepthInterval(lo, hi))


class IntervalHandConstantsTests(unittest.TestCase):
    """常量平面 d=5、零容差：手算闭区间关系。"""

    def setUp(self):
        self.view = plane_view(5.0)
        self.u = self.v = 2.0  # 内部像素中心，四邻点齐全
        self.kw = dict(abs_tol=0.0, rel_tol=0.0)

    def test_degenerate_point_interval_matches_scalar(self):
        # [5,5] 恰在表面上 -> visible（闭区间）。
        st = classify_view(self.view, q(self.u, self.v, 5.0, 5.0), **self.kw)
        self.assertEqual(st.visibility, Visibility.VISIBLE)
        self.assertEqual(st.reason, Reason.NONE)
        self.assertEqual(st.depth_interval, (5.0, 5.0))
        self.assertEqual(st.surface_interval, (5.0, 5.0))
        self.assertEqual(st.sampled_depth, 5.0)
        self.assertIsNone(st.projected_z)

    def test_containing_interval_visible(self):
        # 非零宽度但完全落在容差带内（此处带即 {5}）不可能，
        # 改用带容差平面验证真包含，见 tolerance 测试；此处验证等宽。
        st = classify_view(self.view, q(self.u, self.v, 5.0, 5.0), **self.kw)
        self.assertEqual(st.visibility, Visibility.VISIBLE)

    def test_overlap_is_not_visible(self):
        # [5,6]：近端贴在表面、远端在表面之后 -> 可能遮挡 -> uncertain。
        st = classify_view(self.view, q(self.u, self.v, 5.0, 6.0), **self.kw)
        self.assertEqual(st.visibility, Visibility.UNCERTAIN)
        self.assertEqual(st.reason, Reason.INTERVAL_OVERLAP)
        # [4.9,5.1]：双向重叠 -> uncertain。
        st2 = classify_view(self.view, q(self.u, self.v, 4.9, 5.1), **self.kw)
        self.assertEqual(st2.visibility, Visibility.UNCERTAIN)

    def test_endpoint_contact_not_visible(self):
        # [4,5]：仅远端点 z=5 与表面接触 -> 不判 visible 也不判 in_front。
        st = classify_view(self.view, q(self.u, self.v, 4.0, 5.0), **self.kw)
        self.assertEqual((st.visibility, st.reason),
                         (Visibility.UNCERTAIN, Reason.INTERVAL_OVERLAP))
        # [5,5] 零宽度端点接触是合法 visible（与标量闭区间一致）。
        st0 = classify_view(self.view, q(self.u, self.v, 5.0, 5.0), **self.kw)
        self.assertEqual(st0.visibility, Visibility.VISIBLE)

    def test_definite_occluded(self):
        # 近端严格在表面之后 -> 确定遮挡。
        st = classify_view(self.view, q(self.u, self.v, 5.1, 6.0), **self.kw)
        self.assertEqual((st.visibility, st.reason),
                         (Visibility.OCCLUDED, Reason.NONE))
        st2 = classify_view(self.view, q(self.u, self.v, 6.0, 9.0), **self.kw)
        self.assertEqual(st2.visibility, Visibility.OCCLUDED)

    def test_definite_in_front(self):
        # 远端严格在表面之前 -> 确定在前。
        st = classify_view(self.view, q(self.u, self.v, 3.0, 4.9), **self.kw)
        self.assertEqual((st.visibility, st.reason),
                         (Visibility.IN_FRONT, Reason.NONE))

    def test_tolerance_band_containment_and_contact(self):
        # 平面 d=10，T=0.5+0.01*10=0.6 -> 表面带 [9.4, 10.6]。
        view = plane_view(10.0)
        kw = dict(abs_tol=0.5, rel_tol=0.01)
        # 完全包含（端点恰在带边界）-> visible。
        st = classify_view(view, q(2.0, 2.0, 9.4, 10.6), **kw)
        self.assertEqual(st.visibility, Visibility.VISIBLE)
        self.assertEqual(st.surface_interval, (9.4, 10.6))
        # 远端超出 1e-9 -> 重叠 -> uncertain，而不是 visible。
        st_over = classify_view(view, q(2.0, 2.0, 9.4, 10.6 + 1e-9), **kw)
        self.assertEqual(st_over.visibility, Visibility.UNCERTAIN)
        # 近端已在带外之后 -> 确定 occluded。
        st_occ = classify_view(view, q(2.0, 2.0, 10.6 + 1e-9, 11.0), **kw)
        self.assertEqual(st_occ.visibility, Visibility.OCCLUDED)
        # 远端恰在带的下边界（接触）-> uncertain，不判 in_front。
        st_touch = classify_view(view, q(2.0, 2.0, 9.0, 9.4), **kw)
        self.assertEqual(st_touch.visibility, Visibility.UNCERTAIN)
        # 严格在带之前 -> in_front。
        st_front = classify_view(view, q(2.0, 2.0, 9.0, 9.4 - 1e-9), **kw)
        self.assertEqual(st_front.visibility, Visibility.IN_FRONT)
        # 单点 10.6（带边界）-> visible，与标量闭区间口径一致。
        st_edge = classify_view(view, q(2.0, 2.0, 10.6, 10.6), **kw)
        self.assertEqual(st_edge.visibility, Visibility.VISIBLE)


class IntervalOriginDegeneracyTests(unittest.TestCase):
    def setUp(self):
        self.view = plane_view(5.0)
        self.kw = dict(abs_tol=0.0, rel_tol=0.0)

    def test_crossing_origin_is_uncertain(self):
        # [-1, 5]：区间跨过相机原点 z=0，针孔投影在 0 处无定义。
        st = classify_view(self.view, q(2.0, 2.0, -1.0, 5.0), **self.kw)
        self.assertEqual((st.visibility, st.reason),
                         (Visibility.UNCERTAIN,
                          Reason.DEPTH_SPAN_CROSSES_ORIGIN))
        # [0, 5]：端点恰在 0 同样跨原点。
        st0 = classify_view(self.view, q(2.0, 2.0, 0.0, 5.0), **self.kw)
        self.assertEqual(st0.reason, Reason.DEPTH_SPAN_CROSSES_ORIGIN)
        self.assertEqual(st0.depth_interval, (0.0, 5.0))
        self.assertEqual(st0.surface_interval, (5.0, 5.0))

    def test_entirely_behind(self):
        st = classify_view(self.view, q(2.0, 2.0, -3.0, -1.0), **self.kw)
        self.assertEqual((st.visibility, st.reason),
                         (Visibility.UNKNOWN, Reason.BEHIND_CAMERA))
        # [0,0]：z_far<=0 -> behind_camera。
        st0 = classify_view(self.view, q(2.0, 2.0, 0.0, 0.0), **self.kw)
        self.assertEqual(st0.reason, Reason.BEHIND_CAMERA)

    def test_sampling_failure_precedes_origin_reason(self):
        # 跨原点但投影像素本身越界/缺邻域/被禁用：采样原因优先。
        st_oob = classify_view(self.view, q(99.0, 2.0, -1.0, 5.0), **self.kw)
        self.assertEqual((st_oob.visibility, st_oob.reason),
                         (Visibility.UNKNOWN, Reason.OUT_OF_BOUNDS))
        st_edge = classify_view(self.view, q(5.0, 5.0, -1.0, 5.0), **self.kw)
        self.assertEqual((st_edge.visibility, st_edge.reason),
                         (Visibility.UNKNOWN, Reason.MISSING_NEIGHBOR))
        mask = [[False] * 6 for _ in range(6)]
        mask[2][2] = True
        masked = plane_view(5.0, mask=mask)
        st_mask = classify_view(masked, q(2.0, 2.0, -1.0, 5.0), **self.kw)
        self.assertEqual((st_mask.visibility, st_mask.reason),
                         (Visibility.UNKNOWN, Reason.MASKED_PIXEL))

    def test_unknown_sampling_with_positive_interval(self):
        # 正区间同样继承全部采样 unknown 原因。
        st_oob = classify_view(self.view, q(-0.5, 2.0, 1.0, 5.0), **self.kw)
        self.assertEqual(st_oob.reason, Reason.OUT_OF_BOUNDS)
        st_edge = classify_view(self.view, q(5.0, 5.0, 1.0, 5.0), **self.kw)
        self.assertEqual(st_edge.reason, Reason.MISSING_NEIGHBOR)
        mask = [[False] * 6 for _ in range(6)]
        mask[1][3] = True  # 投影 (2.5,1.5) 的邻点
        masked = plane_view(5.0, mask=mask)
        st_mask = classify_view(masked, q(2.5, 1.5, 1.0, 5.0), **self.kw)
        self.assertEqual(st_mask.reason, Reason.MASKED_PIXEL)


class IntervalMultiviewTests(unittest.TestCase):
    def _views(self, depths):
        return [plane_view(d) for d in depths]

    def test_per_view_reasons_preserved_no_fusion(self):
        # 同一条像素射线，三机表面深度 4 / 5 / 8，点区间 [4.5,5.5]。
        views = self._views([4.0, 5.0, 8.0])
        query = q(2.0, 2.0, 4.5, 5.5)
        results = classify_points(views, [query], abs_tol=0.0, rel_tol=0.0)
        row = results[0]
        self.assertEqual(
            [s.visibility for s in row],
            [Visibility.OCCLUDED, Visibility.UNCERTAIN, Visibility.IN_FRONT],
        )
        self.assertEqual(row[1].reason, Reason.INTERVAL_OVERLAP)
        # uncertain 既不算 visible 也不会被其他视图翻转为 visible。
        self.assertEqual(count_visible_views(row), 0)
        self.assertEqual(indices_visible_in_at_least_k(results, 1), [])

    def test_mixed_scalar_and_interval_batch(self):
        views = self._views([5.0, 5.0])
        # 标量点 (x,y,z)=(1,1,5) -> 像素 (2,2)；区间查询同像素。
        scalar_pt = (1.0, 1.0, 5.0)
        interval_q = q(2.0, 2.0, 4.9, 5.1)
        results = classify_points(views, [scalar_pt, interval_q],
                                  abs_tol=0.0, rel_tol=0.0)
        # 标量：两视图都 visible；区间：两视图都 uncertain（逐视图保留）。
        self.assertTrue(all(
            s.visibility is Visibility.VISIBLE for s in results[0]))
        self.assertTrue(all(
            s.visibility is Visibility.UNCERTAIN for s in results[1]))
        self.assertEqual(indices_visible_in_at_least_k(results, 2), [0])
        # 视图下标逐格写入。
        self.assertEqual([s.view_index for s in results[1]], [0, 1])

    def test_zero_width_interval_equals_scalar_grid(self):
        # 零宽度区间裁定必须与同射线标量点裁定逐位一致。
        w = h = 6
        vals = [[3.0 + 0.07 * c + 0.11 * r for c in range(w)] for r in range(h)]
        cam = Camera(K=K0, R=IDENT, t=ZERO)
        view = View(cam, DepthMap(vals))
        for u0 in (0.3 * i - 0.4 for i in range(-1, 9)):
            for v0 in (0.27 * j - 0.3 for j in range(-1, 9)):
                for z in (2.0, 5.0, 8.0):
                    P = ref.world_point_at_pixel(K0, IDENT, ZERO, u0, v0, z)
                    ss = classify_view(view, P, abs_tol=0.02, rel_tol=0.01)
                    si = classify_view(
                        view, q(u0, v0, z, z), abs_tol=0.02, rel_tol=0.01
                    )
                    self.assertEqual(si.visibility, ss.visibility,
                                     msg=f"u={u0} v={v0} z={z}")
                    self.assertEqual(si.reason, ss.reason)
                    self.assertAlmostEqual(si.u, ss.u, places=9)
                    self.assertAlmostEqual(si.v, ss.v, places=9)
                    # 标量路径经投影反算像素含 1 ULP 舍入，故用近似比较。
                    self.assertAlmostEqual(
                        si.sampled_depth, ss.sampled_depth, places=12)


class IntervalReferenceCrossCheckTests(unittest.TestCase):
    def test_ramp_grid_cross_check(self):
        # 解析斜面深度图：在内部像素网格上枚举区间，与独立参考对照。
        w = h = 7
        vals = [[2.0 + 0.13 * c + 0.21 * r for c in range(w)] for r in range(h)]
        mask = [[False] * w for _ in range(h)]
        mask[1][2] = True
        cam = Camera(K=K0, R=IDENT, t=ZERO)
        view = View(cam, DepthMap(vals, mask))
        a_tol, r_tol = 0.03, 0.02
        us = [0.3 * i - 0.5 for i in range(10)]
        vs = [0.27 * j - 0.4 for j in range(10)]
        # 确定性枚举的区间端点（含跨原点、接触、包含、分离）。
        spans = [
            (-3.0, -1.0), (-1.0, 5.0), (0.0, 5.0),
            (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 7.0),
            (1.0, 100.0),
        ]
        for u in us:
            for v in vs:
                for lo, hi in spans:
                    got = classify_view(view, q(u, v, lo, hi),
                                        abs_tol=a_tol, rel_tol=r_tol)
                    exp = ref.interval_classify(
                        vals, mask, u, v, lo, hi, a_tol, r_tol
                    )
                    self.assertEqual(got.visibility.value, exp["visibility"],
                                     msg=f"u={u} v={v} [{lo},{hi}]")
                    self.assertEqual(got.reason.value, exp["reason"])
                    if "surface_lo" in exp:
                        self.assertAlmostEqual(
                            got.surface_interval[0], exp["surface_lo"], places=12)
                        self.assertAlmostEqual(
                            got.surface_interval[1], exp["surface_hi"], places=12)


class IntervalInputValidationTests(unittest.TestCase):
    def test_bad_interval_bounds(self):
        with self.assertRaises(InvalidParameterError):
            DepthInterval(5.0, 4.0)  # near > far
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidParameterError):
                DepthInterval(bad, 5.0)
            with self.assertRaises(InvalidParameterError):
                DepthInterval(1.0, bad)

    def test_reject_bool_bounds(self):
        with self.assertRaises(InvalidParameterError):
            DepthInterval(True, 5.0)
        with self.assertRaises(InvalidParameterError):
            DepthInterval(1.0, False)

    def test_bad_query(self):
        with self.assertRaises(InvalidParameterError):
            IntervalQuery(u=0.0, v=0.0, interval=(1.0, 5.0))  # 类型错误
        with self.assertRaises(InvalidParameterError):
            IntervalQuery(u=float("nan"), v=0.0,
                          interval=DepthInterval(1.0, 5.0))
        with self.assertRaises(InvalidParameterError):
            IntervalQuery(u=0.0, v=float("inf"),
                          interval=DepthInterval(1.0, 5.0))

    def test_view_must_be_view(self):
        with self.assertRaises(InvalidParameterError):
            classify_view(object(), q(0.0, 0.0, 1.0, 5.0))  # type: ignore[arg-type]

    def test_negative_tolerance_rejected_interval(self):
        view = plane_view(5.0)
        with self.assertRaises(InvalidParameterError):
            classify_view(view, q(2.0, 2.0, 1.0, 5.0), abs_tol=-1e-9)
        with self.assertRaises(InvalidParameterError):
            classify_view(view, q(2.0, 2.0, 1.0, 5.0), rel_tol=-1e-9)

    def test_batch_atomicity_mixed(self):
        view = plane_view(5.0)
        # 非法标量点在批次后部：整体失败，无部分结果。
        with self.assertRaises(InvalidPointError):
            classify_points(
                [view],
                [q(2.0, 2.0, 5.0, 5.0), (0.0, float("nan"), 5.0)],
            )
        # 非法区间（near>far）在构造时即拒绝，批次同样整体失败。
        with self.assertRaises(InvalidParameterError):
            classify_points(
                [view],
                [(1.0, 1.0, 5.0), q(2.0, 2.0, 9.0, 1.0)],
            )
        # 容差非法时先于任何计算报错。
        with self.assertRaises(InvalidParameterError):
            classify_points([view], [q(2.0, 2.0, 5.0, 5.0)],
                            abs_tol=float("inf"))

    def test_frozen_value_objects(self):
        iv = DepthInterval(1.0, 5.0)
        with self.assertRaises(Exception):
            iv.z_far = 9.0  # type: ignore[misc]
        iq = IntervalQuery(0.0, 0.0, iv)
        with self.assertRaises(Exception):
            iq.u = 3.0  # type: ignore[misc]
        self.assertEqual(iv.width, 4.0)
        self.assertEqual(iv.as_tuple(), (1.0, 5.0))


if __name__ == "__main__":
    unittest.main()
