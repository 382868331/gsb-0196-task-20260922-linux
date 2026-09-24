# 深度遮挡的多视图可见性裁定（visibility）

离线视觉重建用的可复用小库：已知针孔相机、世界点与每相机深度图，
按“深度为相机 z、四邻域双线性、严格 unknown 语义”裁定每个世界点
在每个视图下是 **visible / occluded / in_front / unknown**，并聚合
出至少在 k 个视图中可见的点集。本轮迭代把点深度推广为**深度区间**
`[z_near, z_far]`：同一套入口额外返回**确定可见 / 确定遮挡 /
确定在前 / uncertain（不确定）**，重叠或仅端点接触的区间不判可见，
区间跨过相机原点 z=0 作为显式退化裁定。

- Python：**3.14.7**（仅用标准库：`math`、`dataclasses`、`enum`、`unittest` 等）
- 依赖：**无第三方依赖**，装好 Python 即可运行；不联网、不需要账号或密钥
- 平台：Linux 原生；核心、测试、演示全部离线、确定性

## 运行命令

```bash
# 全部测试（49 个用例：27 个标量旧用例 + 22 个区间新用例）
python -m unittest discover -s tests -v

# 固定输入演示（真实计算，实测 < 0.1 秒，远低于 90 秒要求）
python demo.py
```

## 接口与数据结构

```python
from visibility import (
    Camera, DepthMap, View,                  # 输入对象（不可变）
    DepthInterval, IntervalQuery,            # 深度区间裁定输入（新增）
    Visibility, Reason, ViewStatus,          # 输出
    classify_view, classify_points,          # 裁定（接受点或区间查询）
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
  `point` 为三元组时走标量裁定；传入 `IntervalQuery` 时走深度区间裁定。
- `DepthInterval(z_near, z_far)`：沿针孔视线的相机 z **闭区间**（不可变）。
  要求两端有限（拒绝 bool/NaN/Infinity）且 `z_near <= z_far`；
  允许 `<= 0`（跨原点是裁定阶段的显式退化，不在构造时吞掉）。
  `z_near == z_far` 为合法零宽度区间。
- `IntervalQuery(u, v, interval)`：像素射线 `(u, v)`（像素中心在整数
  坐标）与该光线上的深度区间。同一条光线上任意 z>0 的点投影像素恒为
  `(u, v)`，因此区间裁定不需要世界点；`u/v` 必须有限，允许落在图外
  （裁定为 `unknown/out_of_bounds`）。
- `classify_points(views, points, ...)`：`points` 中可**混用**世界点
  三元组与 `IntervalQuery`；返回 `results[point_index][view_index]` 的
  `ViewStatus` 矩阵。同一点在不同相机的状态（含每视图 `uncertain`
  及其原因）各自独立保留，不做跨视图融合。
- `count_visible_views(statuses) -> int`。
- `indices_visible_in_at_least_k(results, k) -> list[int]`：只计
  `visible`，返回可见视图数 `>= k` 的点下标。**不做多数票**，
  `occluded`/`in_front`/`unknown`/`uncertain` 永远不会被其他视图
  翻转为 `visible`。

`ViewStatus` 字段：`view_index`、`visibility`、`reason`、
`projected_z`、`sampled_depth`、`u`、`v`，以及区间裁定新增的
`depth_interval`（查询区间 `[z_near,z_far]`）与
`surface_interval`（表面容差带 `[d-T, d+T]`）；标量裁定时后两者为
`None`，区间裁定时 `projected_z` 为 `None`。

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

## 深度区间裁定规则（本轮新增）

输入为像素射线 `(u, v)` 与点深度闭区间 `[z_near, z_far]` 时：

1. 上面的 2–5 步（越界/缺邻点/禁用/双线性）**完全相同**；四点齐全后
   得到同一表面深度 `d` 与容差 `T = abs_tol + rel_tol*d`，
   表面可能深度为闭区间 **`[d-T, d+T]`**（相对项以采样深度 d 为基准）。
2. 原点退化在采样成功后判定：
   - `z_far <= 0`：整条视线段在相机后方 → `unknown/behind_camera`；
   - `z_near <= 0 < z_far`：区间跨过相机原点，针孔投影在 z=0 处无
     定义 → `uncertain/depth_span_crosses_origin`，不猜可见性。
3. 闭区间比较（含边界）：
   - `z_near >= d-T` 且 `z_far <= d+T` → **`visible`（确定可见）**
   - `z_near > d+T` → **`occluded`（确定遮挡）**
   - `z_far < d-T` → **`in_front`（确定在表面之前）**
   - 其余一切仅端点接触或部分重叠的情形 →
     **`uncertain/interval_overlap`**。

**重叠/接触不判可见**：只要区间有任何一段伸出容差带且方向指向表面
后方，就存在遮挡可能，只能给 `uncertain`。零宽度区间 `[z,z]` 的
裁定与标量路径完全一致（闭区间边界仍 `visible`）。采样失败的
unknown 原因（越界/缺邻点/禁用）优先于跨原点原因。

## 输入上限与错误语义

- 深度图尺寸：高、宽均在 **[1, 32]**，必须矩形；32×32 是接口边界。
- 所有数值输入（点、内参、外参、深度、容差、区间端点、查询像素）
  拒绝 **NaN / ±Infinity**。
- 整数参数（`k`、`view_index`）拒绝 **bool**（`True` 不当作 1）；
  区间端点同样拒绝 bool。
- 容差必须 `>= 0`；`k` 必须是 `[1, 视图数]` 内的整数。
- `DepthInterval` 要求有限且 `z_near <= z_far`（**允许 ≤ 0**，跨原点
  由裁定处理）；`IntervalQuery` 的 `u/v` 必须有限，`interval` 必须是
  `DepthInterval` 实例。
- 非法输入抛出可定位的异常，信息含字段/下标路径
  （如 `values[3][7]`、`point[1]`、`K[1][0]`、
  `depth_interval.z_near`），不静默纠正：
  `InvalidCameraError` / `InvalidDepthMapError` /
  `InvalidPointError` / `InvalidParameterError`，均继承
  `VisibilityError`（`ValueError`）。
- 批次原子性：`classify_points` 先整体校验全部点、视图与参数（含混合
  批次中的区间查询），再计算；函数不修改输入、不持有可变状态，
  校验失败时不产生部分结果。
- 本问题不存在“搜索预算耗尽”的裁定路径；裁定只有确定的四态加
  `uncertain`，因此无“预算耗尽 vs 证明无解”的区分需要。

## 浮点误差口径

- 投影、刚体变换与双线性插值使用 IEEE-754 双精度（Python `float`），
  常规数量级下误差为机器精度量级（约 1e-15 的相对舍入）。
- 可见性判据是 **闭区间** `|cz-d| <= abs_tol + rel_tol*d`；
  默认 `abs_tol=rel_tol=1e-6`。恰好在边界上的点判 `visible`，
  超出边界（哪怕 1e-9）即判 `occluded/in_front`（测试
  `test_tolerance_band` 固定了这一行为），不用宽容差掩盖边界错误。
- 区间裁定沿用同一闭区间口径：点区间被表面容差带 `[d-T,d+T]`
  **完全包含**才确定可见；仅端点接触（如 `z_far == d-T` 且近端在带
  之外）或部分重叠判 `uncertain`；零宽度 `[z,z]` 在带边界上仍可见。
- 相对项以**采样（插值）深度 d** 为基准，不以点深度为基准。
- 双线性只对四邻点齐全的像素执行；像素中心、图像边缘不做外推。

## 设计取舍

- 深度口径固定为相机 z（非欧氏距离），与题目一致。
- 像素中心位于整数坐标；投影在最后一行/列（缺少 +1 邻点）一律
  unknown，而非退化到单点/双点插值，避免歧义。
- 内参仅支持“针孔 + 可选斜切”的严格子集（`K10==0`、`K[2]==(0,0,1)`），
  不宣称与完整行业标准（畸变模型、任意 K 结构等）兼容。
- 区间查询以**像素射线 + z 区间**为输入而非世界点包围盒：同一条
  针孔光线上 z>0 的点投影像素恒定，射线+区间即完整描述，且天然
  覆盖跨原点情形；多视图结果逐视图保留状态与原因，只提供计数聚合，
  不强制融合成二值、不做概率/多数票。
- 表面不确定性只取声明的深度容差带（双线性表面值 ± T），不把像素
  角点深度跨度额外并入表面区间——深度图在亚像素处的表面由双线性
  插值确定；这保证零宽度区间与标量裁定逐位一致。
- 纯函数式、无隐藏状态；输入对象不可变（frozen dataclass + 元组）。

## 已知限制

- 不支持镜头畸变、正交/全景相机、非矩形深度图。
- 不做亚像素外推与图像边界外的任何采样。
- 深度图只携带表面单层深度，不处理透明/多表面。
- 区间裁定假定查询覆盖同一条针孔视线（像素固定）；不做像素足迹
  积分、跨视图概率证据优化或世界坐标包围盒的多像素投影（本轮明确
  不在范围内）。表面不确定性只含声明容差，不额外建模像素角点的
  深度跨度。
- 批量规模按 Python 列表/元组传入；32×32 是单图边界，库不额外限制
  视图数或点数，但演示不穷举上限规模。

## 测试与独立性说明

- `tests/test_visibility.py`：像素中心手算常量、容差边界、遮挡边缘、
  相机后点、缺邻域、mask 禁用、多视图聚合与输入校验（旧有 27 例，
  本轮全部保留未改）。
- `tests/test_interval.py`：区间裁定的手算常量（确定可见/遮挡/在前、
  接触与重叠不判可见、容差带包含与边界）、**原点深度退化**
  （跨原点 uncertain、全在后方 unknown、采样原因优先级）、多视图
  逐原因保留与混合批次、零宽度区间与标量路径一致性，以及非法区间/
  查询/容差与批次原子性失败用例（22 例）。
- `tests/reference.py`：**独立参考实现**，不导入被测核心；标量参考
  之外新增独立书写的 `interval_classify`，在解析斜面深度图上对网格
  枚举的各类区间做交叉核对。手工常量仍是期望值主来源，投影↔逆投影
  的往返一致仅作补充检查。
- `demo.py`：固定确定性输入，真实计算并打印标量四态、区间确定态/
  uncertain/原点退化、多视图逐视图结果与 k 聚合，以及两次实际触发
  的输入拒绝（NaN 点、反向区间）；无 sleep、无预录结果。
