from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QTransform, QFont
from PySide6.QtWidgets import QWidget


class RobotWidget(QWidget):
    def __init__(self, name: str, size: int, x: int, y: int, head: int, color: QColor, parent=None):
        super().__init__(parent)
        self.name = name
        self.color = color
        self.x = x
        self.y = y
        self.head = head
        self.setFixedSize(size, size)
        self.move(x, y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = min(self.width(), self.height())

        center_x = round(self.rect().x() + size / 2)
        center_y = round(self.rect().y() + size / 2)

        transform = QTransform()
        transform.translate(center_x, center_y)
        transform.rotate(self.head + 90)
        transform.translate(-center_x, -center_y)
        painter.setTransform(transform)

        # 画履带（左右深色矩形）
        tread_w = size * 0.18
        tread_h = size * 0.85
        tread_y = center_y - tread_h / 2
        tread_color = QColor("#444a3f")
        painter.setBrush(tread_color)
        painter.setPen(QColor("#222222"))
        # 左履带
        painter.drawRect(center_x - size * 0.35 - tread_w / 2, tread_y, tread_w, tread_h)
        # 右履带
        painter.drawRect(center_x + size * 0.35 - tread_w / 2, tread_y, tread_w, tread_h)

        # 画底盘（坦克车身）
        body_w = size * 0.56
        body_h = size * 0.7
        body_x = center_x - body_w / 2
        body_y = center_y - body_h / 2
        body_color = QColor(self.color)
        painter.setBrush(body_color)
        painter.setPen(QColor("#333333"))
        painter.drawRoundedRect(body_x, body_y, body_w, body_h, 4, 4)

        # 画炮塔（小圆）
        turret_r = size * 0.18
        painter.setBrush(QColor("#222222"))
        painter.setPen(QColor("#222222"))
        painter.drawEllipse(center_x - turret_r, center_y - turret_r, turret_r * 2, turret_r * 2)

        # 画炮管（正面方向，head方向）
        gun_len = size * 0.38
        gun_w = size * 0.09
        painter.setBrush(QColor("#222222"))
        painter.setPen(QColor("#222222"))
        painter.drawRect(center_x - gun_w/2, center_y - gun_len - turret_r, gun_w, gun_len)

        # 画编号
        small_font = QFont()
        small_font.setPointSize(8)
        painter.setFont(small_font)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.name)
