"""树状邻接面集合的纹理相机选择（精确树 DP）。

问题定义
--------
给定一棵面邻接树（最多 64 个面），每个面有最多 8 个候选视图，
每个候选带数据代价 data_cost 与可见性标记 visible
（通常来自 visibility.core 的裁定结果）。为每个面选一个
**可见**候选视图，目标最小化::

    总成本 = Σ 面数据代价 + Σ 相邻面选了不同相机的接缝代价

因为邻接关系是树，用树 DP 求**精确**最优解；不做贪心、不做近似。
某个面没有任何可见候选时抛出 InfeasibleSelectionError，
错误信息指明具体面 ID，绝不伪造默认相机。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .errors import InfeasibleSelectionError, InvalidSelectionError
from .validation import _require_finite_number

MAX_FACES = 64
"""树状邻接面集合允许的最大面数（接口边界）。"""

MAX_CANDIDATES_PER_FACE = 8
"""每个面允许的最大候选视图数（接口边界）。"""


@dataclass(frozen=True)
class CandidateView:
    """一个面的一个候选视图。

    - view_id: 相机/视图编号（非负整数，按面 ID 顺序输出时即
      相机向量的分量）。
    - data_cost: 选该视图的数据代价（有限且 >= 0）。
    - visible: 该视图对此面是否可见；False 的候选不可选。
    """

    view_id: int
    data_cost: float
    visible: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.view_id, bool) or not isinstance(self.view_id, int):
            raise InvalidSelectionError(
                f"candidate.view_id must be an int, got {type(self.view_id).__name__}"
            )
        if self.view_id < 0:
            raise InvalidSelectionError(
                f"candidate.view_id must be >= 0, got {self.view_id}"
            )
        cost = _require_finite_number(
            self.data_cost, "candidate.data_cost", InvalidSelectionError
        )
        if cost < 0.0:
            raise InvalidSelectionError(
                f"candidate.data_cost must be >= 0, got {cost}"
            )
        object.__setattr__(self, "data_cost", cost)
        if not isinstance(self.visible, bool):
            raise InvalidSelectionError(
                f"candidate.visible must be bool, got {type(self.visible).__name__}"
            )


@dataclass(frozen=True)
class FaceCandidates:
    """一个面的全部候选视图（1..8 个，构造时防御性冻结为元组）。"""

    face_id: int
    candidates: tuple[CandidateView, ...] = field(default=())

    def __init__(self, face_id: int, candidates: Sequence[CandidateView]) -> None:
        if isinstance(face_id, bool) or not isinstance(face_id, int):
            raise InvalidSelectionError(
                f"face_id must be an int, got {type(face_id).__name__}"
            )
        if face_id < 0:
            raise InvalidSelectionError(f"face_id must be >= 0, got {face_id}")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise InvalidSelectionError(
                f"faces[{face_id}].candidates must be a sequence of CandidateView"
            )
        if not 1 <= len(candidates) <= MAX_CANDIDATES_PER_FACE:
            raise InvalidSelectionError(
                f"faces[{face_id}] must have 1..{MAX_CANDIDATES_PER_FACE} "
                f"candidates, got {len(candidates)}"
            )
        frozen = []
        for i, cand in enumerate(candidates):
            if not isinstance(cand, CandidateView):
                raise InvalidSelectionError(
                    f"faces[{face_id}].candidates[{i}] must be a CandidateView, "
                    f"got {type(cand).__name__}"
                )
            frozen.append(cand)
        object.__setattr__(self, "face_id", face_id)
        object.__setattr__(self, "candidates", tuple(frozen))


@dataclass(frozen=True)
class SeamEdge:
    """相邻面之间的一条边；两端选了不同相机时支付 cost。"""

    face_a: int
    face_b: int
    cost: float

    def __post_init__(self) -> None:
        for name in ("face_a", "face_b"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise InvalidSelectionError(
                    f"edge.{name} must be an int, got {type(v).__name__}"
                )
        cost = _require_finite_number(self.cost, "edge.cost", InvalidSelectionError)
        if cost < 0.0:
            raise InvalidSelectionError(f"edge.cost must be >= 0, got {cost}")
        object.__setattr__(self, "cost", cost)


@dataclass(frozen=True)
class DataLedgerEntry:
    """账本数据项：一个面选了哪个视图、付了多少数据代价。"""

    face_id: int
    view_id: int
    data_cost: float


@dataclass(frozen=True)
class SeamLedgerEntry:
    """账本接缝项：一条边两端的选择与实际支付的接缝代价。"""

    face_a: int
    face_b: int
    view_a: int
    view_b: int
    cost: float
    """两端相机相同为 0.0，不同为边代价。"""


@dataclass(frozen=True)
class SelectionResult:
    """选择结果与账本。数据项按面 ID 升序，接缝项按输入边顺序。"""

    assignment: tuple[tuple[int, int], ...]
    """(face_id, view_id) 对，按面 ID 升序。"""
    camera_vector: tuple[int, ...]
    """按面 ID 升序排列的所选相机编号。"""
    data_ledger: tuple[DataLedgerEntry, ...]
    seam_ledger: tuple[SeamLedgerEntry, ...]
    total_data_cost: float
    total_seam_cost: float
    total_cost: float


def _validate_inputs(
    faces: object, edges: object
) -> tuple[list[FaceCandidates], list[SeamEdge]]:
    if not isinstance(faces, Sequence) or isinstance(faces, (str, bytes)):
        raise InvalidSelectionError("faces must be a sequence of FaceCandidates")
    if not 1 <= len(faces) <= MAX_FACES:
        raise InvalidSelectionError(
            f"faces must contain 1..{MAX_FACES} faces, got {len(faces)}"
        )
    face_list = []
    seen: set[int] = set()
    for i, f in enumerate(faces):
        if not isinstance(f, FaceCandidates):
            raise InvalidSelectionError(
                f"faces[{i}] must be a FaceCandidates, got {type(f).__name__}"
            )
        if f.face_id in seen:
            raise InvalidSelectionError(f"duplicate face_id {f.face_id}")
        seen.add(f.face_id)
        face_list.append(f)
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        raise InvalidSelectionError("edges must be a sequence of SeamEdge")
    edge_list = []
    for i, e in enumerate(edges):
        if not isinstance(e, SeamEdge):
            raise InvalidSelectionError(
                f"edges[{i}] must be a SeamEdge, got {type(e).__name__}"
            )
        for name in ("face_a", "face_b"):
            if getattr(e, name) not in seen:
                raise InvalidSelectionError(
                    f"edges[{i}].{name}={getattr(e, name)} is not a known face_id"
                )
        if e.face_a == e.face_b:
            raise InvalidSelectionError(
                f"edges[{i}] connects face {e.face_a} to itself"
            )
        edge_list.append(e)
    return face_list, edge_list


def _build_tree(
    face_list: list[FaceCandidates], edge_list: list[SeamEdge]
) -> tuple[list[list[tuple[int, int, float]]], int]:
    """校验树结构并返回 (邻接表, 根下标)。邻接项为 (邻居下标, 边下标, 边代价)。"""
    n = len(face_list)
    index_of = {f.face_id: i for i, f in enumerate(face_list)}
    if len(edge_list) != n - 1:
        raise InvalidSelectionError(
            f"a tree over {n} faces needs exactly {n - 1} edges, "
            f"got {len(edge_list)}"
        )
    adj: list[list[tuple[int, int, float]]] = [[] for _ in range(n)]
    for ei, e in enumerate(edge_list):
        a, b = index_of[e.face_a], index_of[e.face_b]
        adj[a].append((b, ei, e.cost))
        adj[b].append((a, ei, e.cost))
    # n-1 条边 + 连通 <=> 树；用迭代 DFS 验证连通性。
    root = min(range(n), key=lambda i: face_list[i].face_id)
    visited = [False] * n
    stack = [root]
    visited[root] = True
    while stack:
        node = stack.pop()
        for nxt, _, _ in adj[node]:
            if not visited[nxt]:
                visited[nxt] = True
                stack.append(nxt)
    if not all(visited):
        raise InvalidSelectionError(
            "edges do not form a single tree over all faces (disconnected)"
        )
    return adj, root


def select_cameras(
    faces: Sequence[FaceCandidates], edges: Sequence[SeamEdge]
) -> SelectionResult:
    """为树状邻接面集合精确选择纹理相机。

    返回 SelectionResult，含每面视图、数据项与接缝项账本、
    按面 ID 升序的相机向量。任何面缺少可见候选时抛出
    InfeasibleSelectionError 并指明具体面 ID。
    """
    face_list, edge_list = _validate_inputs(faces, edges)
    n = len(face_list)

    # 每面的可选候选（可见）下标；缺候选立即报告具体面。
    usable: list[list[int]] = []
    for f in face_list:
        idxs = [i for i, c in enumerate(f.candidates) if c.visible]
        if not idxs:
            raise InfeasibleSelectionError(
                f"face {f.face_id} has no visible candidate view; "
                f"cannot select a texture camera for it"
            )
        usable.append(idxs)

    adj, root = _build_tree(face_list, edge_list)

    # 迭代后序遍历（根固定为面 ID 最小者，保证确定性）。
    order: list[int] = []
    parent = [-2] * n
    parent[root] = -1
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        for nxt, _, _ in adj[node]:
            if parent[nxt] == -2:
                parent[nxt] = node
                stack.append(nxt)
    order.reverse()  # 子先于父

    # dp[node][k]：node 选其第 k 个可用候选时，子树最小成本。
    # choice[node][k][child]：该状态下每个子节点的最优候选位置。
    dp: list[list[float]] = [[] for _ in range(n)]
    choice: list[list[dict[int, int]]] = [[] for _ in range(n)]
    for node in order:
        cands = usable[node]
        dp[node] = [0.0] * len(cands)
        choice[node] = [{} for _ in cands]
        for k, ci in enumerate(cands):
            total = face_list[node].candidates[ci].data_cost
            for nxt, _, seam in adj[node]:
                if nxt == parent[node]:
                    continue
                child_cands = usable[nxt]
                best_cost = None
                best_j = 0
                my_view = face_list[node].candidates[ci].view_id
                for j, cj in enumerate(child_cands):
                    child_view = face_list[nxt].candidates[cj].view_id
                    cost = dp[nxt][j] + (seam if child_view != my_view else 0.0)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_j = j
                total += best_cost  # type: ignore[operator]
                choice[node][k][nxt] = best_j
            dp[node][k] = total

    # 根取最小；平局保留候选列表中靠前者（确定性）。
    root_best = min(range(len(usable[root])), key=lambda k: dp[root][k])
    picked = [0] * n  # picked[node] = usable[node] 中的位置
    picked[root] = root_best
    stack = [root]
    while stack:
        node = stack.pop()
        for nxt, _, _ in adj[node]:
            if nxt == parent[node]:
                continue
            picked[nxt] = choice[node][picked[node]][nxt]
            stack.append(nxt)

    chosen_view = [
        face_list[node].candidates[usable[node][picked[node]]].view_id
        for node in range(n)
    ]

    # 账本：数据项按面 ID 升序，接缝项按输入边顺序。
    ordered = sorted(range(n), key=lambda i: face_list[i].face_id)
    data_ledger = tuple(
        DataLedgerEntry(
            face_id=face_list[i].face_id,
            view_id=chosen_view[i],
            data_cost=face_list[i].candidates[usable[i][picked[i]]].data_cost,
        )
        for i in ordered
    )
    index_of = {f.face_id: i for i, f in enumerate(face_list)}
    seam_ledger = tuple(
        SeamLedgerEntry(
            face_a=e.face_a,
            face_b=e.face_b,
            view_a=chosen_view[index_of[e.face_a]],
            view_b=chosen_view[index_of[e.face_b]],
            cost=(
                e.cost
                if chosen_view[index_of[e.face_a]] != chosen_view[index_of[e.face_b]]
                else 0.0
            ),
        )
        for e in edge_list
    )
    total_data = sum(entry.data_cost for entry in data_ledger)
    total_seam = sum(entry.cost for entry in seam_ledger)
    return SelectionResult(
        assignment=tuple((entry.face_id, entry.view_id) for entry in data_ledger),
        camera_vector=tuple(entry.view_id for entry in data_ledger),
        data_ledger=data_ledger,
        seam_ledger=seam_ledger,
        total_data_cost=total_data,
        total_seam_cost=total_seam,
        total_cost=total_data + total_seam,
    )
