"""树 DP 纹理相机选择测试。

期望值来源：
1. 手工推算的小案例常量（含"每面单独最优却接缝很多"的反例）；
2. 本文件内的暴力枚举参考实现（不调用被测核心 select_cameras），
   对多棵小树枚举全部合法选择交叉核对总成本与账本自洽性。
"""

from __future__ import annotations

import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visibility import (  # noqa: E402
    CandidateView,
    FaceCandidates,
    InfeasibleSelectionError,
    InvalidSelectionError,
    SeamEdge,
    select_cameras,
)


def brute_force(faces, edges):
    """独立参考：枚举所有合法选择，返回最小总成本。"""
    face_list = sorted(faces, key=lambda f: f.face_id)
    visible = [[c.view_id for c in f.candidates if c.visible] for f in face_list]
    best_cost = None
    for combo in itertools.product(*visible):
        cost = 0.0
        for f, v in zip(face_list, combo):
            for c in f.candidates:
                if c.view_id == v:
                    cost += c.data_cost
                    break
        by_id = {f.face_id: v for f, v in zip(face_list, combo)}
        for e in edges:
            if by_id[e.face_a] != by_id[e.face_b]:
                cost += e.cost
        if best_cost is None or cost < best_cost - 1e-12:
            best_cost = cost
    return best_cost


def make_face(face_id, cands):
    return FaceCandidates(face_id, [CandidateView(*c) for c in cands])


class TestSelectCamerasBasic(unittest.TestCase):
    def test_single_face_picks_cheapest_visible(self):
        faces = [make_face(0, [(3, 5.0, True), (1, 2.0, True), (2, 0.5, False)])]
        res = select_cameras(faces, [])
        self.assertEqual(res.camera_vector, (1,))
        self.assertEqual(res.assignment, ((0, 1),))
        self.assertEqual(res.total_cost, 2.0)
        self.assertEqual(res.total_seam_cost, 0.0)
        self.assertEqual(len(res.data_ledger), 1)
        self.assertEqual(res.seam_ledger, ())

    def test_two_faces_seam_tradeoff(self):
        # 面0 单独最优是相机1(1.0)，面1 单独最优是相机2(1.0)，
        # 但接缝代价 10：全局最优是统一用相机1（1.0+3.0+0）。
        faces = [
            make_face(0, [(1, 1.0, True), (2, 5.0, True)]),
            make_face(1, [(1, 3.0, True), (2, 1.0, True)]),
        ]
        edges = [SeamEdge(0, 1, 10.0)]
        res = select_cameras(faces, edges)
        self.assertEqual(res.camera_vector, (1, 1))
        self.assertEqual(res.total_cost, 4.0)
        self.assertEqual(res.total_data_cost, 4.0)
        self.assertEqual(res.total_seam_cost, 0.0)

    def test_greedy_per_face_counterexample(self):
        # 反例：每面单独最优 -> 相机 (1,2,1)，两条边都付接缝 4+4=8，
        # 数据 1+1+1=3，合计 11；树 DP 全选相机1 -> 数据 1+2+1=4，
        # 接缝 0，合计 4。贪心"每面最优"明显劣于全局最优。
        faces = [
            make_face(0, [(1, 1.0, True), (2, 9.0, True)]),
            make_face(1, [(1, 2.0, True), (2, 1.0, True)]),
            make_face(2, [(1, 1.0, True), (2, 9.0, True)]),
        ]
        edges = [SeamEdge(0, 1, 4.0), SeamEdge(1, 2, 4.0)]
        greedy_vec = (1, 2, 1)
        greedy_cost = 1.0 + 1.0 + 1.0 + 4.0 + 4.0
        res = select_cameras(faces, edges)
        self.assertEqual(res.camera_vector, (1, 1, 1))
        self.assertEqual(res.total_cost, 4.0)
        self.assertLess(res.total_cost, greedy_cost)
        self.assertNotEqual(res.camera_vector, greedy_vec)
        # 账本逐条核对
        self.assertEqual(
            [(d.face_id, d.view_id, d.data_cost) for d in res.data_ledger],
            [(0, 1, 1.0), (1, 1, 2.0), (2, 1, 1.0)],
        )
        self.assertEqual(
            [(s.face_a, s.face_b, s.view_a, s.view_b, s.cost) for s in res.seam_ledger],
            [(0, 1, 1, 1, 0.0), (1, 2, 1, 1, 0.0)],
        )
        self.assertEqual(res.total_data_cost, 4.0)
        self.assertEqual(res.total_seam_cost, 0.0)

    def test_invisible_candidate_never_chosen(self):
        # 唯一便宜的候选不可见时，必须选贵的可见候选。
        faces = [
            make_face(0, [(1, 0.1, False), (2, 8.0, True)]),
            make_face(1, [(1, 0.1, False), (2, 8.0, True)]),
        ]
        edges = [SeamEdge(0, 1, 100.0)]
        res = select_cameras(faces, edges)
        self.assertEqual(res.camera_vector, (2, 2))
        self.assertEqual(res.total_cost, 16.0)

    def test_camera_vector_ordered_by_face_id(self):
        # 输入面顺序打乱，输出仍按面 ID 升序。
        faces = [
            make_face(7, [(0, 1.0, True)]),
            make_face(3, [(0, 1.0, True), (1, 2.0, True)]),
            make_face(5, [(1, 1.0, True)]),
        ]
        edges = [SeamEdge(3, 5, 0.5), SeamEdge(5, 7, 0.5)]
        res = select_cameras(faces, edges)
        self.assertEqual([d.face_id for d in res.data_ledger], [3, 5, 7])
        self.assertEqual(res.camera_vector, (0, 1, 0))
        self.assertEqual(res.assignment, ((3, 0), (5, 1), (7, 0)))


class TestSelectCamerasBruteForce(unittest.TestCase):
    """小树全枚举交叉核对（独立参考实现）。"""

    def check(self, faces, edges):
        res = select_cameras(faces, edges)
        ref_cost = brute_force(faces, edges)
        self.assertAlmostEqual(res.total_cost, ref_cost, places=9)
        # 账本自洽：总成本 = 数据项和 + 接缝项和
        self.assertAlmostEqual(
            res.total_cost,
            sum(d.data_cost for d in res.data_ledger)
            + sum(s.cost for s in res.seam_ledger),
            places=9,
        )
        # 所选视图必须可见
        by_id = {f.face_id: f for f in faces}
        for face_id, view_id in res.assignment:
            cand = next(
                c for c in by_id[face_id].candidates if c.view_id == view_id
            )
            self.assertTrue(cand.visible)

    def test_path_tree(self):
        # 6 个面的链，每面 3 个候选，部分不可见。
        data = [
            [(0, 2.0, True), (1, 3.0, True), (2, 1.5, False)],
            [(0, 1.0, True), (1, 0.5, True), (2, 2.0, True)],
            [(0, 4.0, True), (1, 1.0, False), (2, 2.5, True)],
            [(0, 0.5, True), (1, 3.5, True), (2, 1.0, True)],
            [(0, 2.0, False), (1, 1.0, True), (2, 3.0, True)],
            [(0, 1.5, True), (1, 2.5, True), (2, 0.5, True)],
        ]
        faces = [make_face(i, d) for i, d in enumerate(data)]
        edges = [SeamEdge(i, i + 1, 1.0 + 0.25 * i) for i in range(5)]
        self.check(faces, edges)

    def test_branching_tree(self):
        # 星形 + 分支：0-1, 0-2, 1-3, 1-4, 4-5
        data = [
            [(0, 1.0, True), (1, 2.0, True)],
            [(0, 3.0, True), (1, 0.5, True)],
            [(0, 0.5, True), (1, 3.0, True)],
            [(0, 2.0, True), (1, 2.0, False)],
            [(0, 1.5, True), (1, 1.5, True)],
            [(0, 0.5, False), (1, 0.5, True)],
        ]
        faces = [make_face(i, d) for i, d in enumerate(data)]
        edges = [
            SeamEdge(0, 1, 2.0),
            SeamEdge(0, 2, 1.0),
            SeamEdge(1, 3, 3.0),
            SeamEdge(1, 4, 0.5),
            SeamEdge(4, 5, 4.0),
        ]
        self.check(faces, edges)

    def test_all_identical_candidates(self):
        faces = [make_face(i, [(0, 1.0, True), (1, 1.0, True)]) for i in range(4)]
        edges = [SeamEdge(0, 1, 1.0), SeamEdge(1, 2, 1.0), SeamEdge(2, 3, 1.0)]
        self.check(faces, edges)


class TestSelectCamerasValidation(unittest.TestCase):
    def test_missing_visible_candidate_reports_face(self):
        faces = [
            make_face(0, [(1, 1.0, True)]),
            make_face(42, [(1, 1.0, False), (2, 2.0, False)]),
        ]
        edges = [SeamEdge(0, 42, 1.0)]
        with self.assertRaises(InfeasibleSelectionError) as ctx:
            select_cameras(faces, edges)
        self.assertIn("42", str(ctx.exception))

    def test_non_tree_edges_rejected(self):
        faces = [make_face(i, [(0, 1.0, True)]) for i in range(3)]
        # 三角形：3 条边不是树
        edges = [SeamEdge(0, 1, 1.0), SeamEdge(1, 2, 1.0), SeamEdge(2, 0, 1.0)]
        with self.assertRaises(InvalidSelectionError):
            select_cameras(faces, edges)
        # 不连通（4 面 3 边但成环+孤立点）
        faces4 = [make_face(i, [(0, 1.0, True)]) for i in range(4)]
        edges2 = [SeamEdge(0, 1, 1.0), SeamEdge(1, 2, 1.0), SeamEdge(2, 0, 1.0)]
        with self.assertRaises(InvalidSelectionError):
            select_cameras(faces4, edges2)

    def test_unknown_face_in_edge_rejected(self):
        faces = [make_face(0, [(0, 1.0, True)]), make_face(1, [(0, 1.0, True)])]
        with self.assertRaises(InvalidSelectionError):
            select_cameras(faces, [SeamEdge(0, 9, 1.0)])

    def test_limits(self):
        # 超过 64 面
        faces = [make_face(i, [(0, 1.0, True)]) for i in range(65)]
        edges = [SeamEdge(i, i + 1, 1.0) for i in range(64)]
        with self.assertRaises(InvalidSelectionError):
            select_cameras(faces, edges)
        # 超过 8 候选
        with self.assertRaises(InvalidSelectionError):
            make_face(0, [(i, 1.0, True) for i in range(9)])
        # 空候选
        with self.assertRaises(InvalidSelectionError):
            FaceCandidates(0, [])

    def test_bad_values_rejected(self):
        with self.assertRaises(InvalidSelectionError):
            CandidateView(0, float("nan"), True)
        with self.assertRaises(InvalidSelectionError):
            CandidateView(0, -1.0, True)
        with self.assertRaises(InvalidSelectionError):
            CandidateView(True, 1.0, True)
        with self.assertRaises(InvalidSelectionError):
            CandidateView(0, 1.0, 1)
        with self.assertRaises(InvalidSelectionError):
            SeamEdge(0, 1, -0.5)

    def test_duplicate_face_id_rejected(self):
        faces = [make_face(0, [(0, 1.0, True)]), make_face(0, [(0, 2.0, True)])]
        with self.assertRaises(InvalidSelectionError):
            select_cameras(faces, [])


if __name__ == "__main__":
    unittest.main()
