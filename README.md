# 深度遮挡的多视图可见性裁定（visibility）

离线视觉重建用的可复用小库：已知针孔相机、世界点与每相机深度图，
按“深度为相机 z、四邻域双线性、严格 unknown 语义”裁定每个世界点
在每个视图下的状态，并聚合出至少在 k 个视图中可见的点集。

**深度按区间裁定**：表面可占据逐像素闭区间 `[near, far]`，点的投影
深度也可携带不确定半径 `depth_eps`（区间 `[cz-eps, cz+eps]`）。
裁定分别返回 **visible（确定可见）/ occluded（确定遮挡）/
in_front（确定在表面前）/ uncertain（证据不足）/ unknown（无法
投影或采样）**；区间重叠或仅端点接触一律 uncertain，**不直接判可见**。
多视图结果逐视图保留原因，不强制融合成二值。

- Python：**3.14.7**（仅用标准库：`math`、`dataclasses`、`enum`、`unittest` 等）
- 依赖：**无第三方依赖**，装好 Python 即可运行；不联网、不需要账号或密钥
- 平台：Linux 原生；核心、测试、演示全部离线、确定性

## 运行命令

```bash
# 全部测试（62 个用例）
python -m unittest discover -s tests -v

# 固定输入演示（真实计算，实测约 0.04 秒，远低于 90 秒要求）
python demo.py
```

## 接口与数据结构

```python
from visibility import (
    Camera, DepthMap, View,                          # 输入对象（不可变）
    Visibility, Reason, ViewStatus,                  # 输出
    classify_view, classify_points, classify_intervals,  # 裁定
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
- `DepthMap(values, disabled=None, far=None)`：行主序二维深度图。
  - `values[row][col]` 为该像素**近端**表面相机 z（必须 >0 且有限）；
  - `far[row][col]` 为可选的**远端**表面相机 z，必须 `far >= values`；
    缺省时与 `values` 相同，即旧版单点深度语义；
  - `disabled[row][col]` 为 `True` 表示显式禁用；缺省全 False；
  - 构造时做防御性拷贝，事后改外部列表不影响已构造对象。
  - `DepthMap.from_intervals(near, far, disabled=None)`：以近/远两层
    构造区间深度图的显式入口。
  - 属性 `is_interval`：是否存在 `near < far` 的像素。
- `View(camera, depth_map)`：相机与深度图的绑定。
- `classify_view(view, point, abs_tol=1e-6, rel_tol=1e-6, *, view_index=0, depth_eps=0.0) -> ViewStatus`。
  - `depth_eps`（>=0）：投影深度的不确定半径；点以区间
    `[cz-depth_eps, cz+depth_eps]` 参与裁定。缺省 0 时**旧版单点
    行为逐位保持**。
- `classify_intervals(z_lo, z_hi, d_lo, d_hi, abs_tol=1e-6, rel_tol=1e-6) -> Visibility`：
  不依赖相机/深度图的纯函数区间裁定，便于单独核对与复用。
- `classify_points(views, points, abs_tol=1e-6, rel_tol=1e-6, *, depth_eps=0.0)`：
  返回 `results[point_index][view_index]` 的 `ViewStatus` 矩阵；
  同一点在不同相机的状态与原因各自独立保留。
- `count_visible_views(statuses) -> int`。
- `indices_visible_in_at_least_k(results, k) -> list[int]`：只计
  `visible`，返回可见视图数 `>= k` 的点下标。**不做多数票**，
  `occluded` / `uncertain` 永远不会被其他视图翻转为 `visible`。

`ViewStatus` 字段：

| 字段 | 含义 |
| --- | --- |
| `view_index` / `visibility` / `reason` | 视图下标、裁定、细分原因 |
| `projected_z` | 相机坐标 z（点区间中点） |
| `sampled_depth` | 双线性采样的表面**近端**深度；无法采样为 None |
| `u` / `v` | 投影像素；无法计算为 None |
| `projected_z_low` / `projected_z_high` | 点投影区间 `[cz-eps, cz+eps]` |
| `sampled_depth_far` | 双线性采样的表面**远端**深度（退化时等于近端） |
| `depth_interval` / `projected_interval` | 上述两层区间的只读属性视图 |

新增字段全部带默认值，按旧字段位置构造/解包的既有调用不受影响。

## 裁定规则（区间语义，顺序即契约）

设点区间 `Z=[z_lo,z_hi]`（`z_lo=cz-depth_eps` 等），四邻域双线性对
near/far 两层分别插值得表面区间 `D=[d_lo,d_hi]`。

1. **先判深度区间与相机原点的关系**：
   - `z_hi <= 0` → `unknown/behind_camera`，不继续投影；
   - `z_lo <= 0 < z_hi`（区间跨过或触及原点，`1/z` 无定义）→
     `unknown/depth_interval_at_origin`。
2. 以区间中点 cz 投影到像素 `(u, v)`；落在图覆盖范围之外
   （`u<0 或 v<0 或 u>=宽 或 v>=高`）→ `unknown/out_of_bounds`。
3. 取 `(floor u, floor v)` 的四邻域 stencil；stencil 任一像素越界
   （图像最后一行/列）→ `unknown/missing_neighbor`。
4. 四个邻点任一在 `disabled` mask 中为 True → `unknown/masked_pixel`。
   **缺邻点绝不以 0 或默认值补齐。**
5. 四点全有效才对 near、far **分别双线性插值**，得 `[d_lo, d_hi]`。
   双线性权重非负且和为 1，端点分别插值的结果是混合表面真实深度
   范围的包络（对仿射深度斜面精确），不会用某个厚角点的远端
   污染整段。
6. 容差在两端分别取：`T_lo = abs_tol + rel_tol*d_lo`、
   `T_hi = abs_tol + rel_tol*d_hi`，容差板为
   `B=[d_lo-T_lo, d_hi+T_hi]`。
   - `Z ⊆ B`（闭区间，含端面）→ `visible`；
   - `z_lo > d_hi+T_hi`（严格分离于板后方）→ `occluded`；
   - `z_hi < d_lo-T_lo`（严格分离于板前方）→ `in_front`；
   - 其余——部分重叠，或仅端点接触于板外边界
     （`z_lo == d_hi+T_hi` / `z_hi == d_lo-T_lo`）→
     `uncertain`，原因 `depth_interval_overlap`。

退化情形复现旧规则：`depth_eps=0` 且 `far=near` 时，板即
`[d-T, d+T]`，标量点落入板内（含边界）为 visible，严格大于/小于
为 occluded/in_front——与旧实现逐位一致。

## 输入上限与错误语义

- 深度图尺寸：高、宽均在 **[1, 32]**，必须矩形；near/far 两层同形，
  逐元素 `far >= near`；32×32 是接口边界。
- 所有数值输入（点、内参、外参、深度、容差、`depth_eps`、区间端点）
  拒绝 **NaN / ±Infinity**。
- 整数参数（`k`、`view_index`）拒绝 **bool**（`True` 不当作 1）。
- 容差与 `depth_eps` 必须 `>= 0`；`k` 必须是 `[1, 视图数]` 内的整数。
- 非法输入抛出可定位的异常，信息含字段/下标路径
  （如 `values[3][7]`、`far[1][2]`、`point[1]`、`K[1][0]`），不静默
  纠正：`InvalidCameraError` / `InvalidDepthMapError` /
  `InvalidPointError` / `InvalidParameterError`，均继承
  `VisibilityError`（`ValueError`）。
- 批次原子性：`classify_points` 先整体校验全部点、视图、容差与
  `depth_eps`，再计算；函数不修改输入、不持有可变状态，校验失败时
  不产生部分结果。
- 本问题不存在“搜索预算耗尽”的裁定路径。

## 浮点误差口径

- 投影、刚体变换与双线性插值使用 IEEE-754 双精度（Python `float`），
  常规数量级下误差为机器精度量级（约 1e-15 的相对舍入）。
- 可见性判据是**闭区间**：点区间端点恰好在容差板边界上时，落在板内
  算 visible；仅与板在板外边界接触算 uncertain（不是可见）。
  超出边界哪怕 1e-9 也立刻判 occluded/in_front（测试
  `test_tolerance_band` 固定了这一行为），不用宽容差掩盖边界错误。
- 相对容差项分别以**采样得到的 near/far 端点深度**为基准。
- 双线性只对四邻点齐全的像素执行；像素中心、图像边缘不做外推。

## 设计取舍

- 深度口径固定为相机 z（非欧氏距离）。
- 表面厚度与点深度不确定度都用闭区间表达；区间不做概率分布假设，
  证据不足只给 uncertain，不输出置信度。
- 像素位置以点区间**中点** cz 投影：同一束光线上的深度不确定度不改
  变像素落点；这与“针孔方向固定、仅深度有误差”的重建场景一致。
- 像素中心位于整数坐标；投影在最后一行/列（缺少 +1 邻点）一律
  unknown，而非退化到单点/双点插值，避免歧义。
- 内参仅支持“针孔 + 可选斜切”的严格子集（`K10==0`、`K[2]==(0,0,1)`），
  不宣称与完整行业标准（畸变模型、任意 K 结构等）兼容。
- 纯函数式、无隐藏状态；输入对象不可变（frozen dataclass + 元组）。

## 已知限制

- 不支持镜头畸变、正交/全景相机、非矩形深度图。
- 不做亚像素外推与图像边界外的任何采样。
- 深度区间是各向同性的深度不确定度（单个 `depth_eps`），不支持
  逐点不同 eps 或协方差形状的不确定域。
- 深度图只携带单层表面区间，不处理透明/多表面；区间只表达
  “某处可能有表面”，不做多表面层间的穿越计数。
- 不做像素足迹积分、跨视图概率证据优化（本轮迭代明确不交付）。
- 批量规模按 Python 列表/元组传入；32×32 是单图边界，库不额外限制
  视图数或点数，但演示不穷举上限规模。

## 测试与独立性说明

- `tests/test_visibility.py`：原有 27 个用例——像素中心手算常量、
  容差边界、遮挡边缘、相机后点、缺邻域、mask 禁用、多视图聚合与
  输入校验（全部保留，未放宽任何断言）。
- `tests/test_intervals.py`：本轮新增 35 个用例——表面区间/点区间
  的四类确定裁定与 uncertain、**区间接触与重叠**、两端分别容差、
  **原点深度退化**、near/far 分层双线性（解析斜面 + 变厚度手算）、
  多视图逐视图原因、构造与参数校验、旧行为兼容；核对方式包括
  手算常量、纯函数的整数格点集合预言全枚举、以及独立参考交叉核对。
- `tests/reference.py`：**独立参考实现**，不导入被测核心；含原单点
  裁定与新增的区间裁定（`classify_interval`），测试用它交叉核对，
  手工常量仍是期望值主来源，投影↔逆投影的往返一致仅作补充检查。
- `demo.py`：固定确定性输入，真实计算并打印全部状态类别（含
  uncertain 接触/重叠、原点退化）与两次实际触发的输入拒绝
  （NaN 世界点、负 `depth_eps`）；无 sleep、无预录结果。
