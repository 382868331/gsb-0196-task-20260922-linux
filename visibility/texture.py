"""树状邻接面上的纹理相机选择（精确树 DP）。

问题
----
给定一组面（face），每个面有若干候选视图（来自
:func:~visibility.classify_points 的可见性裁定与每视图数据代价），
相邻面若选用不同相机需支付接缝代价。目标：为每个面选一个
**可见** 候选视图，使 数据代价之和 + 接缝代价之和 最小。

邻接关系限定为一棵树（连通且无环，n 个面恰有 n-1 条边），
因此用树形动态规划可在 O(面数 x 候选数^2) 内求得精确解。

约定
----
- 候选的可见性直接取自 :class:visibility.ViewStatus，只有
  Visibility.VISIBLE 的候选可选；occluded / in_front /
  unknown 一律不可选。
- 某面没有任何可见候选时抛出 NoVisibleCandidateError，
  错误中报告具体面 ID，不伪造默认相机。
- 规模上限（接口边界）：最多 64 个面，每面最多 8 个候选。
- 数据代价与接缝代价必须为有限的非负数。
- 同价并列时按 view_index 升序取前者，结果确定。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .errors import InvalidFaceSetError, NoVisibleCandidateError
from .types import ViewStatus, Visibility

MAX_FACES = 64
"""允许的最大面数（接口边界）。"""
MAX_CANDIDATES_PER_FACE = 8
"""每面允许的最大候选视图数（接口边界）。"""


@dataclass(frozen=True)
class FaceCandidate:
    """一个面的一个候选视图。

    - status: 该视图对面（代表点）的可见性裁定结果；
      status.view_index 即相机下标。
    - data_cost: 选用该视图的数据代价（有限、>= 0）。
    """

    status: ViewStatus
    data_cost: float

    @property
    def view_index(self) -> int:
        return self.status.view_index

    @property
    def visible(self) -> bool:
        return self.status.visibility is Visibility.VISIBLE


@dataclass(frozen=True)
class FaceChoices:
    """一个面的最终选择（账本数据项）。"""

    face_id: int
    view_index: int
    data_cost: float


@dataclass(frozen=True)
class SeamChoice:
    """一条邻接边的最终接缝项（账本接缝项）。

    两端相机相同则 cost 为 0；不同则为输入的接缝代价。
    """

    face_a: int
    face_b: int
    view_a: int
    view_b: int
    cost: float


@dataclass(frozen=True)
class TextureSelection:
    """选择结果：每面账本、接缝账本、总成本与按面 ID 排序的相机向量。"""

    faces: tuple[FaceChoices, ...]
    """按 face_id 升序排列的每面选择。"""
    seams: tuple[SeamChoice, ...]
    """按 (小面ID, 大面ID) 升序排列的接缝项。"""
    total_cost: float
    camera_vector: tuple[int, ...]
    """按 face_id 升序排列的每面所选 view_index。"""


def _require_int(x: Any, path: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise InvalidFaceSetError(f"{path} must be an int, got {type(x).__name__}")
    return x


def _require_cost(x: Any, path: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise InvalidFaceSetError(
            f"{path} must be a finite non-negative number, got {type(x).__name__}"
        )
    v = float(x)
    if not math.isfinite(v):
        raise InvalidFaceSetError(f"{path} must be finite, got {x}")
    if v < 0.0:
        raise InvalidFaceSetError(f"{path} must be >= 0, got {v}")
    return v


def _validate_faces(
    faces: Any,
) -> tuple[list[int], dict[int, list[tuple[int, float]]]]:
    """校验面集合。返回 (升序面ID列表, face_id -> [(view_index, data_cost)])。

    只保留可见候选；此处不抛“无可见候选”，由调用方在全部结构
    校验通过后按面 ID 升序报告第一个问题面。
    """
    if not isinstance(faces, Sequence) or isinstance(faces, (str, bytes)):
        raise InvalidFaceSetError("faces must be a sequence of (face_id, candidates)")
    n = len(faces)
    if not 1 <= n <= MAX_FACES:
        raise InvalidFaceSetError(
            f"number of faces must be in [1, {MAX_FACES}], got {n}"
        )
    table: dict[int, list[tuple[int, float]]] = {}
    for i, entry in enumerate(faces):
        path = f"faces[{i}]"
        if (
            not isinstance(entry, Sequence)
            or isinstance(entry, (str, bytes))
            or len(entry) != 2
        ):
            raise InvalidFaceSetError(f"{path} must be a (face_id, candidates) pair")
        face_id = _require_int(entry[0], f"{path}.face_id")
        if face_id in table:
            raise InvalidFaceSetError(f"duplicate face_id {face_id} at {path}")
        cands = entry[1]
        if not isinstance(cands, Sequence) or isinstance(cands, (str, bytes)):
            raise InvalidFaceSetError(f"{path}.candidates must be a sequence")
        if not 1 <= len(cands) <= MAX_CANDIDATES_PER_FACE:
            raise InvalidFaceSetError(
                f"{path}.candidates must contain 1..{MAX_CANDIDATES_PER_FACE} "
                f"entries, got {len(cands)}"
            )
        visible: list[tuple[int, float]] = []
        seen_views: set[int] = set()
        for j, cand in enumerate(cands):
            cpath = f"{path}.candidates[{j}]"
            if not isinstance(cand, FaceCandidate):
                raise InvalidFaceSetError(
                    f"{cpath} must be a FaceCandidate, got {type(cand).__name__}"
                )
            if not isinstance(cand.status, ViewStatus):
                raise InvalidFaceSetError(
                    f"{cpath}.status must be a ViewStatus, "
                    f"got {type(cand.status).__name__}"
                )
            vi = cand.status.view_index
            if isinstance(vi, bool) or not isinstance(vi, int) or vi < 0:
                raise InvalidFaceSetError(
                    f"{cpath}.status.view_index must be a non-negative int, got {vi!r}"
                )
            cost = _require_cost(cand.data_cost, f"{cpath}.data_cost")
            if not cand.visible:
                continue  # 不可见候选不可选，直接跳过
            if vi in seen_views:
                raise InvalidFaceSetError(
                    f"{cpath}: duplicate visible candidate for view_index {vi} "
                    f"on face {face_id}"
                )
            seen_views.add(vi)
            visible.append((vi, cost))
        visible.sort(key=lambda vc: vc[0])
        table[face_id] = visible
    return sorted(table), table


def _validate_edges(
    edges: Any, face_ids: list[int]
) -> list[tuple[int, int, float]]:
    """校验邻接边并确认其构成一棵树。返回 [(a, b, seam_cost)]（a<b）。"""
    n = len(face_ids)
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        raise InvalidFaceSetError("edges must be a sequence of (face_a, face_b, seam_cost)")
    if len(edges) != n - 1:
        raise InvalidFaceSetError(
            f"a tree on {n} faces needs exactly {n - 1} edges, got {len(edges)}"
        )
    id_set = set(face_ids)
    parent = {f: f for f in face_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    norm: list[tuple[int, int, float]] = []
    seen_pairs: set[frozenset] = set()
    for i, edge in enumerate(edges):
        path = f"edges[{i}]"
        if (
            not isinstance(edge, Sequence)
            or isinstance(edge, (str, bytes))
            or len(edge) != 3
        ):
            raise InvalidFaceSetError(
                f"{path} must be a (face_a, face_b, seam_cost) triple"
            )
        a = _require_int(edge[0], f"{path}[0]")
        b = _require_int(edge[1], f"{path}[1]")
        cost = _require_cost(edge[2], f"{path}[2]")
        if a == b:
            raise InvalidFaceSetError(f"{path}: self-loop on face {a} is not a tree edge")
        if a not in id_set:
            raise InvalidFaceSetError(f"{path}: unknown face_id {a}")
        if b not in id_set:
            raise InvalidFaceSetError(f"{path}: unknown face_id {b}")
        pair = frozenset((a, b))
        if pair in seen_pairs:
            raise InvalidFaceSetError(f"{path}: duplicate edge between {a} and {b}")
        seen_pairs.add(pair)
        ra, rb = find(a), find(b)
        if ra == rb:
            raise InvalidFaceSetError(
                f"{path}: edge ({a}, {b}) creates a cycle; adjacency must be a tree"
            )
        parent[ra] = rb
        norm.append((a, b, cost) if a < b else (b, a, cost))
    roots = {find(f) for f in face_ids}
    if len(roots) != 1:
        raise InvalidFaceSetError(
            "adjacency edges do not connect all faces into a single tree"
        )
    return norm


def select_texture_cameras(
    faces,
    edges,
) -> TextureSelection:
    """为树状邻接的面集合精确选择纹理相机（树 DP）。

    参数
    ----
    faces:
        (face_id, candidates) 序列；candidates 为 FaceCandidate
        序列（1..8 个）。只有可见候选可选。
    edges:
        (face_a, face_b, seam_cost) 序列，必须构成一棵树
        （n 面恰 n-1 条边、连通、无环）。两端选不同相机时支付
        seam_cost。

    返回
    ----
    TextureSelection，含每面账本（数据项）、接缝账本（接缝项）、
    总成本与按面 ID 升序的相机向量。

    异常
    ----
    - InvalidFaceSetError：结构/代价非法。
    - NoVisibleCandidateError：某面没有任何可见候选，报告具体
      面 ID，不伪造默认相机。
    """
    face_ids, table = _validate_faces(faces)
    norm_edges = _validate_edges(edges, face_ids)
    # 结构全部合法后，再按面 ID 升序报告第一个无可见候选的面。
    for fid in face_ids:
        if not table[fid]:
            raise NoVisibleCandidateError(fid)

    adj: dict[int, list[tuple[int, float]]] = {f: [] for f in face_ids}
    for a, b, cost in norm_edges:
        adj[a].append((b, cost))
        adj[b].append((a, cost))

    # 以最小面 ID 为根，迭代后序遍历，保证确定性。
    root = face_ids[0]
    order: list[int] = []
    parent_of: dict[int, int] = {root: -1}
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        for nb, _ in adj[node]:
            if nb not in parent_of:
                parent_of[nb] = node
                stack.append(nb)

    # dp[node][k] = 该子树在 node 选第 k 个可见候选时的最小成本；
    # choice[node][k] = 各子节点应选的候选下标。并列取 view_index 小者。
    dp: dict[int, list[float]] = {}
    choice: dict[int, list[dict[int, int]]] = {}
    for node in reversed(order):
        cands = table[node]
        sub_dp = [cost for _, cost in cands]
        sub_choice: list[dict[int, int]] = [dict() for _ in cands]
        for nb, seam in adj[node]:
            if parent_of[node] == nb or nb not in dp:
                continue
            child_cands = table[nb]
            child_dp = dp[nb]
            for k, (vi, _) in enumerate(cands):
                best_val = math.inf
                best_j = 0
                for j, (vj, _) in enumerate(child_cands):
                    val = child_dp[j] + (0.0 if vj == vi else seam)
                    if val < best_val or (val == best_val and vj < child_cands[best_j][0]):
                        best_val = val
                        best_j = j
                sub_dp[k] += best_val
                sub_choice[k][nb] = best_j
        dp[node] = sub_dp
        choice[node] = sub_choice

    # 根节点选择：最小 dp，并列取 view_index 较小者（候选已按 view 升序）。
    root_cands = table[root]
    best_k = 0
    for k in range(1, len(root_cands)):
        if dp[root][k] < dp[root][best_k]:
            best_k = k
    assign_idx: dict[int, int] = {root: best_k}
    for node in order:
        k = assign_idx[node]
        for nb, j in choice[node][k].items():
            assign_idx[nb] = j

    assignment = {f: table[f][assign_idx[f]] for f in face_ids}
    total = dp[root][best_k]

    faces_out = tuple(
        FaceChoices(face_id=f, view_index=vi, data_cost=cost)
        for f, (vi, cost) in sorted(assignment.items())
    )
    seams_out = tuple(
        SeamChoice(
            face_a=a,
            face_b=b,
            view_a=assignment[a][0],
            view_b=assignment[b][0],
            cost=0.0 if assignment[a][0] == assignment[b][0] else seam,
        )
        for a, b, seam in sorted(norm_edges)
    )
    return TextureSelection(
        faces=faces_out,
        seams=seams_out,
        total_cost=total,
        camera_vector=tuple(assignment[f][0] for f in face_ids),
    )
