# 深度遮挡的多视图可见性裁定（visibility）

离线视觉重建用的可复用小库：已知针孔相机、世界点与每相机深度图，
按“深度为相机 z、四邻域双线性、严格 unknown 语义”裁定每个世界点
在每个视图下是 **visible / occluded / in_front / unknown**，并聚合
出至少在 k 个视图中可见的点集。

- Python：**3.14.7**（仅用标准库：`math`、`dataclasses`、`enum`、`unittest` 等）
- 依赖：**无第三方依赖**，装好 Python 即可运行；不联网、不需要账号或密钥
- 平台：Linux 原生；核心、测试、演示全部离线、确定性

## 运行命令

```bash
# 必要测试（40 个用例：原回归 + 增量缓存）
python -m unittest discover -s tests -v

# 固定输入演示（真实计算，实测 < 0.1 秒，远低于约 90 秒要求）
python demo.py
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

## 版本化增量缓存（VisibilityCache）

固定相机集合、世界点集合与 `k` 后，可用缓存避免每次更新都全量重算：

```python
from visibility import (
    VisibilityCache, DepthUpdate, MaskUpdate,
)

cache = VisibilityCache(cameras, depth_maps, points, k=2,
                        abs_tol=1e-6, rel_tol=1e-6)
cache.status(point_index, view_index)      # ViewStatus（实时）
cache.visible_count(point_index)           # 该点可见相机数
cache.selected_indices()                   # 可见数 >= k 的点下标
snap = cache.snapshot()                    # 不可变快照
snap.status(point_index, view_index)       # 旧快照在更新后仍可查询

# 批量更新：每相机至多一个操作，携带 expected_version
report = cache.apply_updates([
    DepthUpdate(view_index=0, depth_map=new_dm, expected_version=0),
    MaskUpdate(view_index=1, pixels=[(row, col, enabled), ...],
               expected_version=0),
])
report.recomputed    # frozenset[(point_index, view_index)]，实际重算对
report.versions      # 发布后各相机版本
report.changed_pixels  # 每相机实际取值/mask 发生变化的像素
```

- `DepthUpdate`：整体替换某相机深度图（含 mask），新图尺寸必须与原图
  一致（相机/点固定，不改变 32×32 上限与矩形性约束）。
- `MaskUpdate`：批量修改像素启用状态；`(r, c, False)` 禁用、
  `(r, c, True)` 恢复；同一批内同一像素只能出现一次，坐标必须在界内。
- 版本：每相机独立整数版本，初值 0；每次成功的更新操作都使该相机
  版本 +1。`expected_version` 与当前版本不符（过期或超前）抛
  `StaleVersionError`。
- 原子性：一次 `apply_updates` 先在旧状态上验证**全部**操作的类型、
  坐标、版本与数值，再构造新数据、重算、一次性发布；任一校验失败时
  深度图、状态、每点可见计数与入选集合都保持原样，不留半份结果。
- 局部性：缓存为每个像素维护“实际查询它的 (点, 相机)”反向依赖。
  更新时只重算变化像素反向依赖命中的点相机对；`report.recomputed`
  暴露该集合供核对。实现**不会**清空全部缓存掩盖漏依赖。
- 依赖登记规则（与裁定顺序一致）：相机后方（`cz <= 0`）不投影、
  不登记；投影整体出界不登记不存在的像素；stencil 落在最后一行/列时
  只登记界内实际存在的角点；**因像素被 mask 禁用而 unknown 的查询也
  登记四个角点**，因此像素恢复有效后该对会被重算（unknown → visible）。
- 新增错误：`InvalidUpdateError`（操作/坐标/形状非法、同相机多操作）、
  `StaleVersionError`（版本不符），均继承 `VisibilityError`。

`ViewStatus` 字段：`view_index`、`visibility`、`reason`、
`projected_z`、`sampled_depth`、`u`、`v`。

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
- `tests/reference.py`：**独立参考实现**，不导入被测核心；测试用它
  交叉核对，另以手工常量作为期望值主来源，投影↔逆投影的往返一致
  仅作补充检查。
- `demo.py`：固定确定性输入，真实计算并打印全部状态类别及一次实际
  触发的输入拒绝（NaN）；无 sleep、无预录结果。
