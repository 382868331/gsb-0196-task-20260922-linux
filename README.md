# 深度遮挡的多视图可见性裁定（visibility）

离线视觉重建工程的可复用模块：给定已知针孔相机、世界空间 3D 点和每相机一张
深度图（相机系 z 值），裁定每个 (点, 相机) 对的可见性，并聚合"至少 k 个视图
可见"的点集。

## 环境

- Python 3.14.7，仅标准库，**无需安装任何第三方依赖**。
- Linux 原生运行；核心、测试、演示全部离线，不需要网络、账号、硬件或 Docker。

## 运行

```bash
python -m unittest discover -s tests -v   # 单元测试（25 个）
python demo.py                            # 确定性演示，约 0.1 秒内完成
```

## 接口

```python
from visibility import (
    PinholeCamera, DepthMap, Status, Verdict, VisibilityError,
    classify_point, classify_points, points_visible_in_at_least,
    MAX_DEPTH_MAP_SIZE,   # 32
)
```

- `PinholeCamera(fx, fy, cx, cy, rotation, translation)`
  已知针孔相机。`rotation` 为 3×3 嵌套序列，`translation` 为 3 向量，约定
  `p_cam = rotation @ p_world + translation`（即平移是世界→相机系的，相机中心
  在世界系位于 `-rotationᵀ @ translation`）。`fx`、`fy` 必须 > 0。
- `DepthMap(width, height, depths, disabled=None)`
  每相机深度图，行主序 `width*height` 个**严格正的有限**深度值（相机系 z）。
  无效像素必须通过 `disabled`（等长 bool 序列，True=禁用）标记；0 不是合法
  占位深度，会被拒绝。
- `classify_point(point, camera, depth_map, abs_tol=0.0, rel_tol=0.0) -> Verdict`
  单点单相机裁定。
- `classify_points(points, views, abs_tol=0.0, rel_tol=0.0) -> list[list[Verdict]]`
  批量裁定，`views` 为 `(camera, depth_map)` 对序列；返回
  `result[点序号][视图序号]`。先校验全部输入再计算，失败时抛出异常、不产生
  任何部分结果（函数纯、无内部状态、不改写入参）。
- `points_visible_in_at_least(per_view_statuses, k) -> set[int]`
  聚合：返回至少在 `k` 个视图中为 `VISIBLE` 的点下标集合。只做选择，从不改
  写任何逐视图状态（不会用多数票把 occluded 改成 visible）。

`Verdict` 字段：`status`（`Status.VISIBLE / OCCLUDED / IN_FRONT / UNKNOWN`）、
`reason`（人读原因，主要用于 UNKNOWN）、`z`（相机系 z）、`depth`（双线性采样
深度）、`u`/`v`（投影像素坐标；无完整邻域时为 `None`）。

## 裁定规则（核心契约）

1. 先变换到相机系并判 `z > 0`；`z <= 0`（相机后/相机平面上）直接 `UNKNOWN`，
   不做投影。
2. 投影 `u = fx*x/z + cx`，`v = fy*y/z + cy`。
3. 取四邻域像素做双线性采样：**四个邻点全部在图内、全部未被禁用、深度均有
   效**才插值；缺任一点（出界、被 mask 禁用）即 `UNKNOWN`，**绝不用 0 补**。
4. 容差 `tol = abs_tol + rel_tol * depth_sampled`：
   - `z < depth - tol` → `IN_FRONT`（点在记录表面之前）
   - `depth - tol <= z <= depth + tol` → `VISIBLE`（边界含端点）
   - `z > depth + tol` → `OCCLUDED`
5. 同一点在不同相机下的状态分别保留在结果矩阵中；聚合只产出点下标集合。

## 输入上限与错误语义

- 深度图宽高均在 `[1, 32]`（`MAX_DEPTH_MAP_SIZE`）。
- 数值输入拒绝 NaN/±Infinity；整数参数（`width`/`height`/`k`）拒绝 bool；
  布尔 mask 元素必须是 bool；容差必须 ≥ 0；点必须是 3 个有限数。
- 非法输入抛出 `VisibilityError`（`ValueError` 子类），消息中点名出错参数
  （如 `depths[1]: must be finite, got nan`），不静默纠正、不返回部分结果。
- 本算法无搜索过程，因此不存在"搜索预算耗尽"；`k` 大于视图数时聚合结果为
  空集，这是可证明的确定结果，与错误区分开（不抛异常）。

## 浮点误差口径

- 全部计算在 IEEE 754 双精度下进行；投影与双线性插值不做额外舍入控制。
- 判定边界为含端点比较（见上），测试用 `abs_tol=1e-9` 验证了解析平面上像素
  中心处 `z` 与采样深度在 1e-9 内一致（此时双线性退化为单像素值，误差为 0）。
- 投影落在非整数像素处时，双线性是对真实深度函数的近似：对光滑曲面误差为
  像素间距的二阶小量。调用方应通过 `rel_tol` 覆盖该离散化误差（demo 中
  16×16 平面场景用 `rel_tol=0.01`，实际误差约 4e-5 相对量级）。不要用过大
  容差掩盖场景本身的遮挡边界——容差语义是"表面厚度"，不是纠错机制。

## 设计取舍

- `UNKNOWN` 不再细分为枚举子状态，原因放在 `Verdict.reason` 字符串中（相机
  后、出界、缺邻域、mask 禁用），保持状态机简单且可机读。
- 旋转矩阵只校验形状与有限性，**不校验正交性**；给定"已知相机"，标定正确性
  是调用方责任。
- 深度必须 > 0：相机系 z ≤ 0 的深度值在针孔模型下无意义，强制用 mask 表达
  无效像素，避免"0 即无效"的隐式约定。
- 逐视图状态矩阵与聚合分离：聚合函数只读状态、只回下标，从结构上保证多数
  票不会改写单视图裁定。

## 已知限制

- 仅实现本题契约的针孔模型（无畸变、无滚动快门等），不宣称兼容任何完整
  行业格式或标准。
- 纯 Python 单线程；深度图 ≤32×32 时单点裁定约微秒级，未做大规模性能优化。
- `k > 视图数` 返回空集而非报错（见上）；`k < 1` 或非 int 会被拒绝。
- 双线性足印完全落在图内才采样，图像最外一圈像素投影恒为 `UNKNOWN`
  （契约要求的"缺邻域即 unknown"，不做边缘钳制）。

## 文件

- `visibility/core.py` — 核心实现（数据结构、校验、裁定、聚合）
- `visibility/__init__.py` — 公开接口再导出
- `tests/test_visibility.py` — 25 个单元测试；期望值由测试内独立参考公式
  （手写投影、射线-平面求交、手工双线性权重）给出，不调用被测核心
- `demo.py` — 确定性演示：解析平面深度图 + 直接投影参考对照，展示全部四种
  状态、双视图状态分离、k=2 聚合及实际触发的输入拒绝
