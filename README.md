# 多视图可见性裁定与树状纹理相机选择（visibility）

离线视觉重建用的可复用小库，含两个部分：

1. **可见性裁定**：已知针孔相机、世界点与每相机深度图，按“深度为
   相机 z、四邻域双线性、严格 unknown 语义”裁定每个世界点在每个视图
   下是 **visible / occluded / in_front / unknown**，并聚合出至少在
   k 个视图中可见的点集。
2. **纹理相机选择（本轮新增）**：利用已有可见性结果，为树状邻接的
   面集合（最多 64 面、每面最多 8 候选）用树 DP 精确最小化
   “数据代价之和 + 相邻面换相机的接缝代价之和”，返回每面视图与
   数据/接缝账本；无可见候选的面被点名拒绝，不伪造默认相机。

- Python：**3.14.7**（仅用标准库：`math`、`dataclasses`、`enum`、`unittest` 等）
- 依赖：**无第三方依赖**，装好 Python 即可运行；不联网、不需要账号或密钥
- 平台：Linux 原生；核心、测试、演示全部离线、确定性

## 运行命令

```bash
# 准备 venv（Linux 原生 Python 3.14.7，仅标准库，无第三方依赖）
python3.14 -m venv .venv
source .venv/bin/activate

# 全部测试（可见性 26 个 + 纹理选择 15 个，共 41 个用例）
python -m unittest discover -s tests -v

# 固定输入演示（真实计算，实测 < 0.1 秒，远低于约 8 秒要求）
python demo.py

# 纹理相机选择演示（贪心 vs 树DP 账本对比 + 无可见候选边界，<0.1 秒）
python demo_iteration.py
```

## 接口与数据结构

```python
from visibility import (
    Camera, DepthMap, View,            # 输入对象（不可变）
    Visibility, Reason, ViewStatus,    # 输出
    classify_view, classify_points,    # 裁定
    count_visible_views, indices_visible_in_at_least_k,  # 聚合
)
```

- `Camera(K, R, t)`：针孔相机。
  - `K` 为 3×3 内参（行主序），`R` 为 3×3 世界→相机旋转，`t` 为 3 维平移。
  - 相机坐标 `p_cam = R @ P + t`；**深度 = 相机坐标 z 分量**。
  - 像素（像素中心在整数坐标）：
    `u = (K00·cx + K01·cy)/cz + K02`，`v = K11·cy/cz + K12`。
  - 结构要求（按本库实现的有限子集）：`K00>0`、`K11>0`、`K10==0`、
    `K[2]==(0,0,1)`（后两者必须严格相等）；允许斜切 `K01`。
- `DepthMap(values, disabled=None)`：行主序二维深度图，
  `values[row][col]` 为该像素表面的相机 z（必须 > 0 且有限）。
  `disabled[row][col]` 为 `True` 表示该像素显式禁用；缺省为全 False。
  构造时做防御性拷贝，事后改外部列表不影响已构造对象。
- `View(camera, depth_map)`：相机与深度图的绑定。
- `classify_view(view, point, abs_tol=1e-6, rel_tol=1e-6, *, view_index=0) -> ViewStatus`。
- `classify_points(views, points, abs_tol=1e-6, rel_tol=1e-6)`：
  返回 `results[point_index][view_index]` 的 `ViewStatus` 矩阵；
  同一点在不同相机的状态各自独立保留。
- `count_visible_views(statuses) -> int`。
- `indices_visible_in_at_least_k(results, k) -> list[int]`：只计
  `visible`，返回可见视图数 `>= k` 的点下标。**不做多数票**，
  `occluded` 永远不会被其他视图翻转为 `visible`。

`ViewStatus` 字段：`view_index`、`visibility`、`reason`、
`projected_z`、`sampled_depth`、`u`、`v`。

## 纹理相机选择（树 DP）

```python
from visibility import (
    FaceCandidate,          # (ViewStatus, data_cost)
    select_texture_cameras, # 精确树 DP 选择
    TextureSelection,       # 结果：faces / seams / total_cost / camera_vector
    FaceChoices, SeamChoice,
    InvalidFaceSetError, NoVisibleCandidateError,
)
```

- 输入：
  - `faces`：`(face_id, [FaceCandidate(...), ...])` 序列，1..64 个面、
    每面 1..8 个候选。`face_id` 为非负整数（拒绝 bool、拒绝重复），
    候选顺序无关（按 `view_index` 去重与并列裁决）。
  - `FaceCandidate(status, data_cost)`：`status` 必须是
    `classify_view/classify_points` 产出的 `ViewStatus`；只有
    `visibility == VISIBLE` 的候选可选，其余一律忽略。
    `data_cost` 为有限非负数。
  - `edges`：`(face_a, face_b, seam_cost)` 序列，必须构成**一棵树**
    （n 面恰 n-1 条边、连通、无环、无自环、无重复边、端点存在）；
    `seam_cost` 为有限非负数，两端相机不同时支付。
- 算法：树形动态规划（迭代后序遍历），状态
  `dp[面][候选] = 数据代价 + Σ子树 min(dp[子][候选] + 接缝)`，
  复杂度 O(n·k²)，给出**精确全局最优**；并列时取 `view_index` 较小者，
  结果确定。
- 输出 `TextureSelection`：
  - `faces`：按面 ID 升序的 `FaceChoices(face_id, view_index, data_cost)`，
    即每面数据项账本；
  - `seams`：按 (小面ID, 大面ID) 升序的
    `SeamChoice(face_a, face_b, view_a, view_b, cost)`，即接缝项账本
    （两端同相机时 `cost == 0`）；
  - `camera_vector`：按面 ID 升序的每面所选相机下标元组；
  - `total_cost == Σ 数据项 + Σ 接缝项`。
- 边界与错误：
  - 某面没有任何**可见**候选时抛 `NoVisibleCandidateError`，异常对象的
    `face_id` 字段点名具体面（结构全部校验通过后按 ID 升序报第一个），
    **不伪造默认相机**；
  - 面数/候选数超限、代价为负或 NaN、ID 重复、边不构成树等抛
    `InvalidFaceSetError`，信息携带 `faces[i]/edges[i]` 下标路径。

## 裁定规则（顺序即契约）

1. 计算相机坐标，**先判 `cz > 0`**：`cz <= 0` → `unknown/behind_camera`，
   不继续投影。
2. 投影到像素 `(u, v)`；落在图覆盖范围之外（`u<0 或 v<0 或 u>=宽 或 v>=高`）
   → `unknown/out_of_bounds`。
3. 取 `(floor u, floor v)` 的四邻域 stencil；stencil 任一像素越界
   （图像最后一行/列）→ `unknown/missing_neighbor`。
4. 四个邻点任一在 `disabled` mask 中为 True → `unknown/masked_pixel`。
   **缺邻点绝不以 0 或默认值补齐。**
5. 四点全有效才做双线性插值得到表面深度 `d`。
6. 容差 `T = abs_tol + rel_tol * d`：
   - `|cz - d| <= T` → `visible`（含边界，闭区间）
   - `cz - d > T` → `occluded`（点在表面之后）
   - `d - cz > T` → `in_front`（点在表面之前）

## 输入上限与错误语义

- 深度图尺寸：高、宽均在 **[1, 32]**，必须矩形；32×32 是接口边界。
- 所有数值输入（点、内参、外参、深度、容差）拒绝 **NaN / ±Infinity**。
- 整数参数（`k`、`view_index`）拒绝 **bool**（`True` 不当作 1）。
- 容差必须 `>= 0`；`k` 必须是 `[1, 视图数]` 内的整数。
- 非法输入抛出可定位的异常，信息含字段/下标路径
  （如 `values[3][7]`、`point[1]`、`K[1][0]`），不静默纠正：
  `InvalidCameraError` / `InvalidDepthMapError` /
  `InvalidPointError` / `InvalidParameterError`，均继承
  `VisibilityError`（`ValueError`）。
- 批次原子性：`classify_points` 先整体校验全部点、视图与参数，再计算；
  函数不修改输入、不持有可变状态，校验失败时不产生部分结果。
- 本问题不存在“搜索预算耗尽”的裁定路径；深度裁定只有确定的四态，
  因此无“预算耗尽 vs 证明无解”的区分需要。

## 浮点误差口径

- 投影、刚体变换与双线性插值使用 IEEE-754 双精度（Python `float`），
  常规数量级下误差为机器精度量级（约 1e-15 的相对舍入）。
- 可见性判据是 **闭区间** `|cz-d| <= abs_tol + rel_tol*d`；
  默认 `abs_tol=rel_tol=1e-6`。恰好在边界上的点判 `visible`，
  超出边界（哪怕 1e-9）即判 `occluded/in_front`（测试
  `test_tolerance_band` 固定了这一行为），不用宽容差掩盖边界错误。
- 相对项以**采样深度 d** 为基准，不以投影 z 为基准。
- 双线性只对四邻点齐全的像素执行；像素中心、图像边缘不做外推。

## 设计取舍

- 深度口径固定为相机 z（非欧氏距离），与题目一致。
- 像素中心位于整数坐标；投影在最后一行/列（缺少 +1 邻点）一律
  unknown，而非退化到单点/双点插值，避免歧义。
- 内参仅支持“针孔 + 可选斜切”的严格子集（`K10==0`、`K[2]==(0,0,1)`），
  不宣称与完整行业标准（畸变模型、任意 K 结构等）兼容。
- 纯函数式、无隐藏状态；输入对象不可变（frozen dataclass + 元组）。

## 已知限制

- 不支持镜头畸变、正交/全景相机、非矩形深度图。
- 不做亚像素外推与图像边界外的任何采样。
- 深度图只携带表面单层深度，不处理透明/多表面。
- 批量规模按 Python 列表/元组传入；32×32 是单图边界，库不额外限制
  视图数或点数，但演示不穷举上限规模。

## 测试与独立性说明

- `tests/test_visibility.py`：像素中心手算常量、容差边界、遮挡边缘、
  相机后点、缺邻域、mask 禁用、多视图聚合与输入校验。
- `tests/test_texture.py`：树 DP 的独立对照测试——`brute_force_best`
  不导入被测核心，对小树（链/分支树）枚举所有可见候选组合取最小总成本，
  与 `select_texture_cameras` 的结果、相机向量和账本交叉核对；
  另含“每面单独最优却接缝更多”的反例（贪心 1.1 > 全局 1.0）、
  不可见候选绝不被选、无可见候选点名报错、环/不连通/超限/类型拒绝等。
- `tests/reference.py`：**独立参考实现**，不导入被测核心；测试用它
  交叉核对，另以手工常量作为期望值主来源，投影↔逆投影的往返一致
  仅作补充检查。
- `demo.py`：固定确定性输入，真实计算并打印全部状态类别及一次实际
  触发的输入拒绝（NaN）；无 sleep、无预录结果。
- `demo_iteration.py`：先用 `classify_points` 真实裁定 4 相机 x 6
  面的可见性，再打印“选择前（逐面贪心）”与“选择后（树 DP 最优）”
  的数据/接缝账本与相机向量，最后演示一个在所有视图中不可见的面被
  `NoVisibleCandidateError` 点名拒绝；实测 < 0.1 秒（要求 < 90 秒）。
  不生成网页、不使用真实相机数据，全部为仓库内确定性合成输入。
