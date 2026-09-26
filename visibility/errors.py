"""可定位的错误类型。

所有输入校验错误都继承 :class:`VisibilityError`（同时是 ``ValueError``），
错误信息包含出错的对象与下标路径，例如 ``depth_map.values[3][7]``，
不做静默纠正。
"""


class VisibilityError(ValueError):
    """本库所有输入/参数错误的基类。"""


class InvalidCameraError(VisibilityError):
    """相机内外参非法（形状、有限性、针孔结构）。"""


class InvalidDepthMapError(VisibilityError):
    """深度图或其像素 mask 非法（尺寸、矩形性、有限性、布尔性）。"""


class InvalidPointError(VisibilityError):
    """世界点非法（长度、元素有限性）。"""


class InvalidParameterError(VisibilityError):
    """标量参数非法（k、容差、批次对象类型等）。"""


class InvalidSelectionError(VisibilityError):
    """纹理相机选择输入非法（面/候选/边结构、代价有限性等）。"""


class InfeasibleSelectionError(VisibilityError):
    """某个面没有任何可见候选视图，无法为其选择纹理相机。"""
