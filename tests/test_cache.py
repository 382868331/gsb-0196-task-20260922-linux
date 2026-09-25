"""版本化可见性缓存的增量更新测试。

参考口径：每次变更后用 ``tests/reference.py``（不导入被测核心）从头按
原公式全算所有 (点, 相机)，与缓存逐对核对；``UpdateReport.recomputed``
再与由反向依赖推导出的精确受影响集合核对局部性。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import reference as ref  # noqa: E402

from visibility import (  # noqa: E402
    Camera,
    DepthMap,
    DepthUpdate,
    InvalidUpdateError,
    MaskUpdate,
    Reason,
    StaleVersionError,
    View,
    Visibility,
    VisibilityCache,
    classify_points,
)

K0 = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 1.0))
IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ZERO = (0.0, 0.0, 0.0)
W = H = 6
D = 5.0


def point_at(u, v, z):
    return (u * z / 10.0, v * z / 10.0, z)


class CacheTestBase(unittest.TestCase):
    def setUp(self):
        self.cam = Camera(K=K0, R=IDENT, t=ZERO)
        vals = [[D for _ in range(W)] for _ in range(H)]
        mask0 = [[False] * W for _ in range(H)]
        mask1 = [[False] * W for _ in range(H)]
        mask2 = [[False] * W for _ in range(H)]
        mask2[3][3] = True  # F 的 stencil 角点之一
        self.masks = [mask0, mask1, mask2]
        self.dms = [
            DepthMap(vals, mask0),
            DepthMap([row[:] for row in vals], mask1),
            DepthMap([row[:] for row in vals], mask2),
        ]
        # A/B 共享 (1,1)(1,2)(2,1)(2,2)；C 顶到最后行列；D 相机后；
        # E 投影出界；F 被 (3,3) 禁用挡住（四邻点 (2,2)(2,3)(3,2)(3,3)）。
        self.points = [
            point_at(1.5, 1.5, D),   # 0 A
            point_at(1.2, 1.2, D),   # 1 B
            point_at(5.5, 5.5, D),   # 2 C missing_neighbor
            (0.0, 0.0, -1.0),        # 3 D behind_camera
            point_at(6.5, 0.0, D),   # 4 E out_of_bounds
            point_at(2.5, 2.5, D),   # 5 F masked on cam2, visible cam0/1
        ]
        self.cache = VisibilityCache([self.cam] * 3, self.dms, self.points, k=2)

    def assert_matches_reference(self, cache):
        for j in range(cache.n_views):
            dm = cache.depth_map(j)
            vals = [list(r) for r in dm.values]
            mask = [list(r) for r in dm.disabled]
            for p, P in enumerate(self.points):
                exp = ref.classify(K0, IDENT, ZERO, vals, mask, P, 1e-6, 1e-6)
                got = cache.status(p, j)
                self.assertEqual(
                    got.visibility.value, exp["visibility"],
                    msg=f"point={p} view={j}",
                )
                self.assertEqual(
                    got.reason.value, exp["reason"], msg=f"point={p} view={j}"
                )
        views = [View(self.cam, cache.depth_map(j)) for j in range(cache.n_views)]
        full = classify_points(views, self.points)
        for p, row in enumerate(full):
            cnt = sum(1 for s in row if s.visibility is Visibility.VISIBLE)
            self.assertEqual(cache.visible_count(p), cnt)
        self.assertEqual(
            cache.selected_indices(),
            tuple(p for p in range(cache.n_points)
                  if cache.visible_count(p) >= cache.k),
        )

    def expected_affected(self, cache, changed_by_view):
        deps = [cache.dependencies(j) for j in range(cache.n_views)]
        affected = set()
        for j, pixels in changed_by_view.items():
            for px in pixels:
                affected.update((p, j) for p in deps[j].get(px, ()))
        return affected


class InitialStateTests(CacheTestBase):
    def test_initial_statuses_and_counts(self):
        self.assert_matches_reference(self.cache)
        for p in (0, 1):
            self.assertEqual(self.cache.visible_count(p), 3)
        for p in (2, 3, 4):
            self.assertEqual(self.cache.visible_count(p), 0)
        self.assertEqual(self.cache.visible_count(5), 2)
        self.assertEqual(self.cache.selected_indices(), (0, 1, 5))
        self.assertEqual(self.cache.status(2, 0).reason, Reason.MISSING_NEIGHBOR)
        self.assertEqual(self.cache.status(3, 0).reason, Reason.BEHIND_CAMERA)
        self.assertEqual(self.cache.status(4, 0).reason, Reason.OUT_OF_BOUNDS)
        self.assertEqual(self.cache.status(5, 2).reason, Reason.MASKED_PIXEL)

    def test_no_fake_dependencies_behind_or_oob(self):
        for j in range(3):
            pts = set()
            for users in self.cache.dependencies(j).values():
                pts.update(users)
            self.assertNotIn(3, pts)
            self.assertNotIn(4, pts)

    def test_unknown_masked_pair_registers_dependencies(self):
        deps2 = self.cache.dependencies(2)
        for px in ((2, 2), (2, 3), (3, 2), (3, 3)):
            self.assertIn(5, deps2[px])
        deps0 = self.cache.dependencies(0)
        for px in ((5, 5),):
            self.assertIn(2, deps0[px])


class IncrementalUpdateTests(CacheTestBase):
    def test_shared_pixel_locality_and_depth_change(self):
        # 相邻点 A/B 共享像素；把相机0 的 (1,1) 深度改浅：A/B 转 occluded，
        # 其余点不重算。
        dm = self.cache.depth_map(0)
        vals = [list(r) for r in dm.values]
        vals[1][1] = 4.0
        report = self.cache.apply_updates(
            [DepthUpdate(0, DepthMap(vals, [list(r) for r in dm.disabled]), 0)]
        )
        expected = self.expected_affected(self.cache, {0: {(1, 1)}})
        self.assertEqual(set(report.recomputed), expected)
        self.assertEqual(set(report.recomputed), {(0, 0), (1, 0)})
        self.assertEqual(self.cache.status(0, 0).visibility, Visibility.OCCLUDED)
        self.assertEqual(self.cache.status(1, 0).visibility, Visibility.OCCLUDED)
        for j in (1, 2):
            self.assertEqual(
                self.cache.status(0, j).visibility, Visibility.VISIBLE
            )
        self.assert_matches_reference(self.cache)
        self.assertEqual(report.versions, (1, 0, 0))

    def test_unknown_to_visible_on_mask_restore(self):
        # F 在相机2 masked/unknown；恢复 (3,3) 后转 visible，计数 2->3。
        self.assertEqual(self.cache.visible_count(5), 2)
        old_snapshot = self.cache.snapshot()
        report = self.cache.apply_updates([MaskUpdate(2, [(3, 3, True)], 0)])
        self.assertEqual(set(report.recomputed), {(5, 2)})
        self.assertEqual(self.cache.status(5, 2).visibility, Visibility.VISIBLE)
        self.assertEqual(self.cache.visible_count(5), 3)
        self.assert_matches_reference(self.cache)
        self.assertEqual(report.versions, (0, 0, 1))
        # 旧快照仍可查询旧结论。
        self.assertEqual(old_snapshot.status(5, 2).reason, Reason.MASKED_PIXEL)
        self.assertEqual(old_snapshot.visible_count(5), 2)
        self.assertEqual(old_snapshot.versions, (0, 0, 0))

    def test_mask_disable_then_restore_roundtrip(self):
        cache = VisibilityCache(
            [self.cam] * 3,
            [self.cache.depth_map(j) for j in range(3)],
            self.points,
            k=3,
        )
        r1 = cache.apply_updates([MaskUpdate(0, [(1, 1, False)], 0)])
        self.assertEqual(set(r1.recomputed), {(0, 0), (1, 0)})
        for p in (0, 1):
            self.assertEqual(cache.status(p, 0).visibility, Visibility.UNKNOWN)
            self.assertEqual(cache.status(p, 0).reason, Reason.MASKED_PIXEL)
            self.assertEqual(cache.visible_count(p), 2)
        self.assertEqual(cache.selected_indices(), ())
        r2 = cache.apply_updates([MaskUpdate(0, [(1, 1, True)], 1)])
        self.assertEqual(set(r2.recomputed), {(0, 0), (1, 0)})
        for p in (0, 1):
            self.assertEqual(cache.status(p, 0).visibility, Visibility.VISIBLE)
            self.assertEqual(cache.visible_count(p), 3)
        self.assertEqual(cache.selected_indices(), (0, 1))

    def test_multi_camera_same_batch(self):
        # 同批禁用相机0 (1,1) 与相机1 (2,2)：受影响对跨两相机，一次发布。
        report = self.cache.apply_updates([
            MaskUpdate(0, [(1, 1, False)], 0),
            MaskUpdate(1, [(2, 2, False)], 0),
        ])
        # 相机0: A,B；相机1: A,B,F（(2,2) 是 F 的角点）。
        self.assertEqual(
            set(report.recomputed),
            {(0, 0), (1, 0), (0, 1), (1, 1), (5, 1)},
        )
        self.assertEqual(report.versions, (1, 1, 0))
        self.assertEqual(self.cache.visible_count(0), 1)
        self.assertEqual(self.cache.visible_count(1), 1)
        self.assertEqual(self.cache.visible_count(5), 1)
        self.assertEqual(self.cache.selected_indices(), ())
        self.assert_matches_reference(self.cache)

    def test_depth_update_is_also_mask_replace(self):
        # DepthUpdate 同时替换 mask：恢复相机2 的 (3,3) 让 F visible。
        dm = self.cache.depth_map(2)
        mask = [list(r) for r in dm.disabled]
        mask[3][3] = False
        report = self.cache.apply_updates(
            [DepthUpdate(2, DepthMap([list(r) for r in dm.values], mask), 0)]
        )
        self.assertEqual(set(report.recomputed), {(5, 2)})
        self.assertEqual(self.cache.status(5, 2).visibility, Visibility.VISIBLE)
        self.assert_matches_reference(self.cache)


class FailureAtomicityTests(CacheTestBase):
    def _fingerprint(self):
        return (
            tuple(
                self.cache.visible_count(p)
                for p in range(self.cache.n_points)
            ),
            self.cache.selected_indices(),
            self.cache.versions,
            tuple(
                self.cache.status(p, j).reason
                for p in range(self.cache.n_points)
                for j in range(self.cache.n_views)
            ),
        )

    def test_stale_version_rejected_and_unchanged(self):
        before = self._fingerprint()
        with self.assertRaises(StaleVersionError):
            self.cache.apply_updates([MaskUpdate(0, [(1, 1, False)], 5)])
        with self.assertRaises(StaleVersionError):
            self.cache.apply_updates([DepthUpdate(1, self.cache.depth_map(1), 1)])
        # 批次中任一更新版本过期 -> 整批不发布，不留半份计数。
        with self.assertRaises(StaleVersionError):
            self.cache.apply_updates([
                MaskUpdate(0, [(1, 1, False)], 0),
                MaskUpdate(1, [(2, 2, False)], 9),
            ])
        self.assertEqual(self._fingerprint(), before)
        self.assert_matches_reference(self.cache)

    def test_bad_coords_rejected_and_unchanged(self):
        before = self._fingerprint()
        with self.assertRaises(InvalidUpdateError):
            self.cache.apply_updates([MaskUpdate(0, [(H, 0, False)], 0)])
        with self.assertRaises(InvalidUpdateError):
            self.cache.apply_updates([MaskUpdate(0, [(0, W, True)], 0)])
        with self.assertRaises(InvalidUpdateError):
            self.cache.apply_updates([MaskUpdate(0, [(0, 0, 1)], 0)])
        self.assertEqual(self._fingerprint(), before)

    def test_shape_mismatch_and_duplicate_rejected(self):
        before = self._fingerprint()
        small = DepthMap([[D] * (W - 1) for _ in range(H)])
        with self.assertRaises(InvalidUpdateError):
            self.cache.apply_updates([DepthUpdate(0, small, 0)])
        with self.assertRaises(InvalidUpdateError):
            self.cache.apply_updates([
                MaskUpdate(0, [(1, 1, False)], 0),
                MaskUpdate(0, [(2, 2, True)], 0),
            ])
        with self.assertRaises(InvalidUpdateError):
            self.cache.apply_updates([MaskUpdate(9, [(0, 0, False)], 0)])
        self.assertEqual(self._fingerprint(), before)

    def test_validation_failure_leaves_no_partial_state(self):
        # 第一个更新合法、第二个含非法坐标：两者都不得生效。
        before = self._fingerprint()
        with self.assertRaises(InvalidUpdateError):
            self.cache.apply_updates([
                MaskUpdate(0, [(1, 1, False)], 0),
                MaskUpdate(1, [(W, 2, False)], 0),
            ])
        self.assertEqual(self._fingerprint(), before)
        self.assert_matches_reference(self.cache)

    def test_no_full_cache_clear_on_update(self):
        # 更新一个像素后，未受影响的 (点,相机) 状态对象身份不变，
        # 证明实现没有清空全部缓存再全量重算。
        untouched = self.cache.status(0, 2)
        self.cache.apply_updates([MaskUpdate(0, [(1, 1, False)], 0)])
        self.assertIs(self.cache.status(0, 2), untouched)


if __name__ == "__main__":
    unittest.main()
