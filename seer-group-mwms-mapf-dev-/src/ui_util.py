from PySide6.QtGui import QColor


def color_to_css_rgba(color: QColor):
    r = color.red()
    g = color.green()
    b = color.blue()
    a = color.alpha() / 255.0  # 将透明度从 0 - 255 范围转换到 0 - 1 范围
    return f"rgba({r}, {g}, {b}, {a})"