"""纹理相机选择（树 DP）的独立测试。

对照实现：:func:brute_force_best 是不导入被测模块的暴力枚举，
对小树枚举所有可见候选组合取最小总成本，与
:func:visibility.select_texture_cameras 的精确解交叉核对。
"""

from __future__ import annotations

import itertools
import unittest

from visibility import (
    FaceCandidate,
    InvalidFaceSetError,
    NoVisibleCandidateError,
    Reason,
    ViewStatus,
    Visibility,
    select_texture_cameras,
)


def make_status(view_index: int, visibility: Visibility) -> ViewStatus:
    return ViewStatus(
        view_index=view_index,
        visibility=visibility,
        reason=Reason.NONE,
        projected_z=1.0,
        sampled_depth=1.0,
        u=0.0,
        v=0.0,
    )


def cand(view_index: int, cost: float, vis: Visibility = Visibility.VISIBLE) -> FaceCandidate:
    return FaceCandidate(status=make_status(view_index, vis), data_cost=cost)


def brute_force_best(faces, edges):
    """独立参考：枚举全部可见组合，返回 (最小总成本, 相机向量列表)。"""
    face_ids = sorted(f for f, _ in faces)
    options = []
    for fid in face_ids:
        cands = next(c for f, c in faces if f == fid)
        vis = [(c.status.view_index, c.data_cost) for c in cands
               if c.status.visibility is Visibility.VISIBLE]
        if not vis:
            return None
        options.append(vis)
    best = None
    best_vecs = []
    for combo in itertools.product(*options):
        total = sum(c for _, c in combo)
        vec = [v for v, _ in combo]
        pos = {f: i for i, f in enumerate(face_ids)}
        for a, b, seam in edges:
            if vec[pos[a]] != vec[pos[b]]:
                total += seam
        if best is None or total < best:
            best = total
            best_vecs = [tuple(vec)]
        elif total == best:
            best_vecs.append(tuple(vec))
    return best, best_vecs


class TestTreeDPExact(unittest.TestCase):
    def test_single_face_no_edges(self):
        faces = [(7, [cand(0, 3.0), cand(2, 1.5)])]
        sel = select_texture_cameras(faces, [])
        self.assertEqual(sel.camera_vector, (2,))
        self.assertEqual(sel.total_cost, 1.5)
        self.assertEqual(sel.seams, ())
        self.assertEqual(len(sel.faces), 1)
        self.assertEqual(sel.faces[0].face_id, 7)

    def test_chain_matches_brute_force(self):
        # 链 0-1-2-3，混合可见/不可见候选。
        faces = [
            (0, [cand(0, 1.0), cand(1, 2.0)]),
            (1, [cand(0, 1.2), cand(1, 0.8), cand(2, 5.0, Visibility.OCCLUDED)]),
            (2, [cand(1, 1.1), cand(3, 0.9)]),
            (3, [cand(0, 2.0), cand(3, 1.0), cand(4, 0.5, Visibility.UNKNOWN)]),
        ]
        edges = [(0, 1, 2.0), (1, 2, 1.0), (2, 3, 3.0)]
        sel = select_texture_cameras(faces, edges)
        best, vecs = brute_force_best(faces, edges)
        self.assertAlmostEqual(sel.total_cost, best)
        self.assertIn(sel.camera_vector, vecs)
        # 账本自洽：总成本 = 数据项 + 接缝项。
        self.assertAlmostEqual(
            sel.total_cost,
            sum(f.data_cost for f in sel.faces) + sum(s.cost for s in sel.seams),
        )

    def test_branching_tree_matches_brute_force(self):
        faces = [
            (10, [cand(0, 1.0), cand(1, 1.4)]),
            (20, [cand(0, 0.6), cand(2, 0.7)]),
            (30, [cand(1, 0.9), cand(2, 1.3), cand(3, 2.2)]),
            (40, [cand(2, 0.4), cand(3, 0.4)]),
            (50, [cand(0, 1.0), cand(3, 1.0)]),
        ]
        edges = [(10, 20, 1.0), (10, 30, 2.0), (30, 40, 0.5), (30, 50, 1.5)]
        sel = select_texture_cameras(faces, edges)
        best, vecs = brute_force_best(faces, edges)
        self.assertAlmostEqual(sel.total_cost, best)
        self.assertIn(sel.camera_vector, vecs)
        # 相机向量按面 ID 升序。
        self.assertEqual(len(sel.camera_vector), 5)

    def test_invisible_candidates_never_chosen(self):
        faces = [
            (0, [cand(0, 0.1, Visibility.OCCLUDED), cand(1, 9.9)]),
            (1, [cand(0, 0.1, Visibility.IN_FRONT), cand(1, 9.9)]),
        ]
        sel = select_texture_cameras(faces, [(0, 1, 100.0)])
        self.assertEqual(sel.camera_vector, (1, 1))
        self.assertAlmostEqual(sel.total_cost, 19.8)

    def test_individually_optimal_but_many_seams_counterexample(self):
        # 反例：每面单独最优（都选视图0，代价0）却产生大量接缝；
        # 全局最优是少数面改用视图1。
        faces = [
            (0, [cand(0, 0.0), cand(1, 0.3)]),
            (1, [cand(0, 0.0), cand(1, 0.3)]),
            (2, [cand(0, 0.0), cand(1, 0.3)]),
            (3, [cand(1, 0.1), cand(0, 5.0)]),
        ]
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        sel = select_texture_cameras(faces, edges)
        best, vecs = brute_force_best(faces, edges)
        self.assertAlmostEqual(sel.total_cost, best)
        # 逐面各自最优 (0,0,0,1)：数据 0.1 + 一条接缝 1.0 = 1.1；
        # 全局最优 (1,1,1,1)：数据 1.0、零接缝。DP 必须严格优于贪心。
        greedy_vec = (0, 0, 0, 1)
        greedy_cost = 0.1 + 1.0  # = 1.1
        self.assertLess(sel.total_cost, greedy_cost)
        self.assertEqual(sel.camera_vector, (1, 1, 1, 1))
        self.assertIn(sel.camera_vector, vecs)

    def test_strong_seam_forces_shared_camera(self):
        # 高接缝代价下，次优数据代价的相机被选中以统一相机。
        faces = [
            (0, [cand(0, 0.0), cand(1, 0.4)]),
            (1, [cand(0, 5.0), cand(1, 0.0)]),
        ]
        sel = select_texture_cameras(faces, [(0, 1, 10.0)])
        # 各自最优 (0,1) 需付 10；统一选1只需 0.4。
        self.assertEqual(sel.camera_vector, (1, 1))
        self.assertAlmostEqual(sel.total_cost, 0.4)

    def test_face_ids_need_not_be_contiguous(self):
        faces = [
            (100, [cand(2, 1.0)]),
            (3, [cand(2, 1.0), cand(5, 2.0)]),
        ]
        sel = select_texture_cameras(faces, [(3, 100, 0.5)])
        self.assertEqual(sel.camera_vector, (2, 2))  # 按 face_id 3,100 排序
        self.assertEqual([f.face_id for f in sel.faces], [3, 100])


class TestValidation(unittest.TestCase):
    def test_missing_visible_candidate_reports_face(self):
        faces = [
            (0, [cand(0, 1.0)]),
            (1, [cand(0, 1.0, Visibility.OCCLUDED),
                 cand(1, 1.0, Visibility.UNKNOWN)]),
            (2, [cand(0, 1.0)]),
        ]
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        with self.assertRaises(NoVisibleCandidateError) as ctx:
            select_texture_cameras(faces, edges)
        self.assertEqual(ctx.exception.face_id, 1)
        self.assertIn("1", str(ctx.exception))

    def test_cycle_rejected(self):
        faces = [(0, [cand(0, 1.0)]), (1, [cand(0, 1.0)]), (2, [cand(0, 1.0)])]
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        with self.assertRaises(InvalidFaceSetError):
            select_texture_cameras(faces, edges)

    def test_disconnected_rejected(self):
        faces = [(i, [cand(0, 1.0)]) for i in range(4)]
        edges = [(0, 1, 1.0), (2, 3, 1.0), (0, 2, 1.0)]
        # 4 面 3 边但 (0,1,2) 连通、3 孤立 -> 不连通。
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        with self.assertRaises(InvalidFaceSetError):
            select_texture_cameras(faces, edges)

    def test_wrong_edge_count_rejected(self):
        faces = [(0, [cand(0, 1.0)]), (1, [cand(0, 1.0)]), (2, [cand(0, 1.0)])]
        with self.assertRaises(InvalidFaceSetError):
            select_texture_cameras(faces, [(0, 1, 1.0)])

    def test_unknown_face_in_edge_rejected(self):
        faces = [(0, [cand(0, 1.0)]), (1, [cand(0, 1.0)])]
        with self.assertRaises(InvalidFaceSetError):
            select_texture_cameras(faces, [(0, 99, 1.0)])

    def test_limits_and_types(self):
        # 65 个面超出上限。
        faces = [(i, [cand(0, 1.0)]) for i in range(65)]
        edges = [(i, i + 1, 1.0) for i in range(64)]
        with self.assertRaises(InvalidFaceSetError):
            select_texture_cameras(faces, edges)
        # 每面 9 个候选超出上限。
        faces = [(0, [cand(i, 1.0) for i in range(9)])]
        with self.assertRaises(InvalidFaceSetError):
            select_texture_cameras(faces, [])
        # bool 面 ID 拒绝。
        with self.assertRaises(InvalidFaceSetError):
            select_texture_cameras([(True, [cand(0, 1.0)])], [])
        # 负接缝代价拒绝。
        with self.assertRaises(InvalidFaceSetError):
            select_texture_cameras(
                [(0, [cand(0, 1.0)]), (1, [cand(0, 1.0)])], [(0, 1, -0.5)]
            )
        # NaN 数据代价拒绝。
        with self.assertRaises(InvalidFaceSetError):
            select_texture_cameras([(0, [cand(0, float("nan"))])], [])
        # 重复面 ID 拒绝。
        with self.assertRaises(InvalidFaceSetError):
            select_texture_cameras(
                [(0, [cand(0, 1.0)]), (0, [cand(1, 1.0)])], []
            )

    def test_ledger_ordering(self):
        faces = [
            (5, [cand(0, 1.0), cand(1, 2.0)]),
            (1, [cand(0, 1.0), cand(1, 2.0)]),
            (9, [cand(0, 1.0), cand(1, 2.0)]),
        ]
        edges = [(9, 1, 1.0), (5, 9, 1.0)]
        sel = select_texture_cameras(faces, edges)
        self.assertEqual([f.face_id for f in sel.faces], [1, 5, 9])
        self.assertEqual([(s.face_a, s.face_b) for s in sel.seams], [(1, 9), (5, 9)])
        self.assertEqual(sel.camera_vector, (0, 0, 0))
        self.assertTrue(all(s.cost == 0.0 for s in sel.seams))


if __name__ == "__main__":
    unittest.main()
