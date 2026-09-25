"""版本化可见性缓存测试。

每次变更后，用同一批相机/点/容差构造全新视图并调用
``classify_points`` 从头按原公式全算一遍作为参考，逐对比较
缓存状态；并用 ``last_recomputed`` 核对局部性（只重算受影响对）。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visibility import (  # noqa: E402
    Camera,
    CameraDepthUpdate,
    DepthMap,
    InvalidDepthMapError,
    InvalidParameterError,
    Reason,
    VersionMismatchError,
    View,
    Visibility,
    VisibilityCache,
    classify_points,
    indices_visible_in_at_least_k,
)

# 相机：fx=fy=10，主点 (4,4)，无旋转无平移；深度即世界 z。
K = ((10.0, 0.0, 4.0), (0.0, 10.0, 4.0), (0.0, 0.0, 1.0))
IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ZERO = (0.0, 0.0, 0.0)
W = H = 8
SURF = 5.0


def point_at(u, v, z):
    return ((u - 4.0) * z / 10.0, (v - 4.0) * z / 10.0, z)


def plane(d=SURF):
    return [[d for _ in range(W)] for _ in range(H)]


def blank_mask():
    return [[False for _ in range(W)] for _ in range(H)]


# 点布局（像素坐标 -> 世界点）：
# P0 (2.0,2.0) 表面；P1 (3.0,2.0) 表面（与 P0 共享第 3 列像素）；
# P2 (6.0,6.0) 表面（远离 P0/P1）；P3 相机后方；P4 投影出界；
# P5 (1.0,1.0) 表面（视图2 中被 mask 禁用）；P6 (7.0,7.0) 角点缺邻域。
POINTS = [
    point_at(2.0, 2.0, SURF),   # 0
    point_at(3.0, 2.0, SURF),   # 1
    point_at(6.0, 6.0, SURF),   # 2
    point_at(2.0, 2.0, -1.0),   # 3 相机后方
    point_at(20.0, 20.0, SURF), # 4 出界
    point_at(1.0, 1.0, SURF),   # 5
    point_at(7.0, 7.0, SURF),   # 6 缺邻域
]
K_VIEWS = 2


def make_views(mask5=False):
    cam = Camera(K=K, R=IDENT, t=ZERO)
    m = blank_mask()
    if mask5:
        m[1][1] = True  # 禁用 P5 stencil 的像素 (row=1,col=1)
    return [
        View(cam, DepthMap(plane())),
        View(cam, DepthMap(plane())),
        View(cam, DepthMap(plane(), m)),
    ]


def reference(views, points, k):
    """从头按原公式全算：状态矩阵 + 每点计数 + 入选集合。"""
    results = classify_points(views, points)
    counts = [
        sum(1 for s in row if s.visibility is Visibility.VISIBLE)
        for row in results
    ]
    return results, counts, indices_visible_in_at_least_k(results, k)


class CacheTestBase(unittest.TestCase):
    def setUp(self):
        self.views = make_views(mask5=True)
        self.cache = VisibilityCache(self.views, POINTS, K_VIEWS)

    def assert_matches_reference(self, views):
        results, counts, chosen = reference(views, POINTS, K_VIEWS)
        for pi, row in enumerate(results):
            for vi, st in enumerate(row):
                got = self.cache.status(pi, vi)
                self.assertEqual(
                    got.visibility, st.visibility, f"({pi},{vi}) vis"
                )
                self.assertEqual(got.reason, st.reason, f"({pi},{vi}) reason")
                self.assertEqual(
                    got.sampled_depth, st.sampled_depth, f"({pi},{vi}) depth"
                )
                self.assertEqual(got.u, st.u, f"({pi},{vi}) u")
                self.assertEqual(got.v, st.v, f"({pi},{vi}) v")
            self.assertEqual(
                self.cache.visible_count(pi), counts[pi], f"count[{pi}]"
            )
        self.assertEqual(self.cache.selected(), chosen)

    def view_with(self, idx, values=None, disabled=None):
        views = list(self.views)
        old = views[idx]
        views[idx] = View(
            old.camera,
            DepthMap(
                old.depth_map.values if values is None else values,
                old.depth_map.disabled if disabled is None else disabled,
            ),
        )
        return views


class InitialStateTests(CacheTestBase):
    def test_initial_matches_full_recompute(self):
        self.assertEqual(self.cache.version, 0)
        self.assert_matches_reference(self.views)

    def test_initial_states(self):
        c = self.cache
        self.assertIs(c.status(0, 0).visibility, Visibility.VISIBLE)
        self.assertIs(c.status(3, 0).visibility, Visibility.UNKNOWN)
        self.assertIs(c.status(3, 0).reason, Reason.BEHIND_CAMERA)
        self.assertIs(c.status(4, 0).reason, Reason.OUT_OF_BOUNDS)
        self.assertIs(c.status(5, 2).reason, Reason.MASKED_PIXEL)
        self.assertIs(c.status(6, 0).reason, Reason.MISSING_NEIGHBOR)
        # P5 在视图 0/1 可见、视图 2 masked -> 计数 2，k=2 入选。
        self.assertEqual(c.visible_count(5), 2)
        self.assertIn(5, c.selected())


class LocalityTests(CacheTestBase):
    def test_shared_pixel_recomputes_only_dependents(self):
        # 改像素 (row=2,col=2)：P0 的 stencil 含它，P1 不含（列 3/4）。
        vals = plane()
        vals[2][2] = 100.0
        self.cache.update(
            [CameraDepthUpdate(0, values=vals)], expected_version=0
        )
        # P0 与 P5（stencil 行1-2/列1-2）都依赖 (2,2)；P1 不依赖。
        self.assertEqual(self.cache.last_recomputed, ((0, 0), (5, 0)))
        self.assertIs(
            self.cache.status(0, 0).visibility, Visibility.IN_FRONT
        )
        self.assertIs(self.cache.status(1, 0).visibility, Visibility.VISIBLE)
        self.assert_matches_reference(self.view_with(0, values=vals))

    def test_shared_column_hits_both_neighbors(self):
        # 改像素 (row=2,col=3)：P0（列2/3）与 P1（列3/4）共享。
        vals = plane()
        vals[2][3] = 100.0
        self.cache.update(
            [CameraDepthUpdate(0, values=vals)], expected_version=0
        )
        self.assertEqual(self.cache.last_recomputed, ((0, 0), (1, 0)))
        self.assert_matches_reference(self.view_with(0, values=vals))

    def test_untracked_pixel_recomputes_nothing(self):
        # 像素 (row=0,col=7) 不在任何点的 stencil 内。
        vals = plane()
        vals[0][7] = 100.0
        self.cache.update(
            [CameraDepthUpdate(0, values=vals)], expected_version=0
        )
        self.assertEqual(self.cache.last_recomputed, ())
        self.assert_matches_reference(self.view_with(0, values=vals))

    def test_missing_neighbor_registers_partial_stencil(self):
        # P6 在角点 (7,7)，stencil 只有 (7,7) 真实存在；改它必须重算 P6。
        vals = plane()
        vals[7][7] = 100.0
        self.cache.update(
            [CameraDepthUpdate(0, values=vals)], expected_version=0
        )
        # P2（stencil 行6-7/列6-7）与 P6（角点，仅 (7,7) 存在）都依赖它。
        self.assertEqual(self.cache.last_recomputed, ((2, 0), (6, 0)))
        # 仍然缺邻域（不存在的像素永不登记），状态保持 unknown。
        self.assertIs(
            self.cache.status(6, 0).reason, Reason.MISSING_NEIGHBOR
        )
        self.assert_matches_reference(self.view_with(0, values=vals))


class UnknownToVisibleTests(CacheTestBase):
    def test_masked_unknown_becomes_visible_on_restore(self):
        # 视图2 的 mask 禁用 (1,1) -> P5 unknown；恢复后必须变 visible。
        self.assertIs(self.cache.status(5, 2).reason, Reason.MASKED_PIXEL)
        self.cache.update(
            [CameraDepthUpdate(2, disabled=blank_mask())],
            expected_version=0,
        )
        self.assertEqual(self.cache.last_recomputed, ((5, 2),))
        self.assertIs(
            self.cache.status(5, 2).visibility, Visibility.VISIBLE
        )
        self.assertEqual(self.cache.visible_count(5), 3)
        views = make_views(mask5=False)
        self.assert_matches_reference(views)

    def test_mask_disable_then_restore_roundtrip(self):
        # 禁用视图0 的 (2,2) -> P0 unknown；再恢复 -> visible。
        m = blank_mask()
        m[2][2] = True
        self.cache.update(
            [CameraDepthUpdate(0, disabled=m)], expected_version=0
        )
        self.assertEqual(self.cache.last_recomputed, ((0, 0), (5, 0)))
        self.assertIs(
            self.cache.status(0, 0).reason, Reason.MASKED_PIXEL
        )
        self.assertEqual(self.cache.visible_count(0), 2)
        self.assert_matches_reference(self.view_with(0, disabled=m))

        self.cache.update(
            [CameraDepthUpdate(0, disabled=blank_mask())],
            expected_version=1,
        )
        self.assertIs(
            self.cache.status(0, 0).visibility, Visibility.VISIBLE
        )
        self.assertEqual(self.cache.visible_count(0), 3)
        self.assert_matches_reference(self.view_with(0))


class MultiCameraBatchTests(CacheTestBase):
    def test_multi_camera_single_batch(self):
        vals0 = plane()
        vals0[2][2] = 100.0   # 影响 P0@视图0
        vals1 = plane()
        vals1[6][6] = 0.5     # 影响 P2@视图1
        new_ver = self.cache.update(
            [
                CameraDepthUpdate(0, values=vals0),
                CameraDepthUpdate(1, values=vals1),
            ],
            expected_version=0,
        )
        self.assertEqual(new_ver, 1)
        self.assertEqual(
            self.cache.last_recomputed, ((0, 0), (2, 1), (5, 0))
        )
        views = self.view_with(0, values=vals0)
        views = self.view_with(1, values=vals1) if False else views
        # 构造参考视图：同时应用两处变更。
        cam = Camera(K=K, R=IDENT, t=ZERO)
        m = blank_mask()
        m[1][1] = True
        ref_views = [
            View(cam, DepthMap(vals0)),
            View(cam, DepthMap(vals1)),
            View(cam, DepthMap(plane(), m)),
        ]
        self.assert_matches_reference(ref_views)

    def test_selected_set_tracks_counts(self):
        # k=2：禁用视图0 与视图1 的 (2,2) 后 P0 计数 1 -> 落选。
        m = blank_mask()
        m[2][2] = True
        self.assertIn(0, self.cache.selected())
        self.cache.update(
            [
                CameraDepthUpdate(0, disabled=m),
                CameraDepthUpdate(1, disabled=m),
            ],
            expected_version=0,
        )
        self.assertEqual(self.cache.visible_count(0), 1)
        self.assertNotIn(0, self.cache.selected())
        cam = Camera(K=K, R=IDENT, t=ZERO)
        m5 = blank_mask()
        m5[1][1] = True
        ref_views = [
            View(cam, DepthMap(plane(), m)),
            View(cam, DepthMap(plane(), m)),
            View(cam, DepthMap(plane(), m5)),
        ]
        self.assert_matches_reference(ref_views)


class VersionAndAtomicityTests(CacheTestBase):
    def test_stale_version_rejected_and_state_untouched(self):
        vals = plane()
        vals[2][2] = 100.0
        self.cache.update(
            [CameraDepthUpdate(0, values=vals)], expected_version=0
        )
        before = self.cache.status(0, 0)
        with self.assertRaises(VersionMismatchError):
            self.cache.update(
                [CameraDepthUpdate(0, values=plane())],
                expected_version=0,  # 过期：当前已是 1
            )
        self.assertEqual(self.cache.version, 1)
        self.assertIs(self.cache.status(0, 0), before)
        self.assertEqual(self.cache.last_recomputed, ((0, 0), (5, 0)))
        self.assert_matches_reference(self.view_with(0, values=vals))

    def test_failed_batch_is_atomic(self):
        # 批内第二项含 NaN：整批拒绝，旧快照可查询、无半份计数。
        good = plane()
        good[2][2] = 100.0
        bad = plane()
        bad[6][6] = float("nan")
        snapshot = [
            [self.cache.status(pi, vi) for vi in range(3)]
            for pi in range(len(POINTS))
        ]
        counts = [self.cache.visible_count(pi) for pi in range(len(POINTS))]
        with self.assertRaises(InvalidDepthMapError):
            self.cache.update(
                [
                    CameraDepthUpdate(0, values=good),
                    CameraDepthUpdate(1, values=bad),
                ],
                expected_version=0,
            )
        self.assertEqual(self.cache.version, 0)
        self.assertEqual(self.cache.last_recomputed, ())
        for pi in range(len(POINTS)):
            for vi in range(3):
                self.assertIs(self.cache.status(pi, vi), snapshot[pi][vi])
            self.assertEqual(self.cache.visible_count(pi), counts[pi])
        self.assert_matches_reference(self.views)

    def test_invalid_camera_index_and_duplicate(self):
        with self.assertRaises(InvalidParameterError):
            self.cache.update(
                [CameraDepthUpdate(7, values=plane())], expected_version=0
            )
        with self.assertRaises(InvalidParameterError):
            self.cache.update(
                [
                    CameraDepthUpdate(0, values=plane()),
                    CameraDepthUpdate(0, values=plane()),
                ],
                expected_version=0,
            )
        with self.assertRaises(InvalidParameterError):
            self.cache.update(
                [CameraDepthUpdate(True, values=plane())],
                expected_version=0,
            )
        with self.assertRaises(InvalidParameterError):
            self.cache.update(
                [CameraDepthUpdate(0, values=plane())],
                expected_version=True,
            )
        self.assertEqual(self.cache.version, 0)
        self.assert_matches_reference(self.views)

    def test_empty_update_rejected(self):
        with self.assertRaises(InvalidParameterError):
            self.cache.update([CameraDepthUpdate(0)], expected_version=0)


if __name__ == "__main__":
    unittest.main()
