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
        # print(f"head: {self.head}")
        # 实现中心变换，比较麻烦
        transform.translate(center_x, center_y)
        transform.rotate(self.head)
        transform.translate(-center_x, -center_y)

        painter.setTransform(transform)
        painter.setBrush(QColor(self.color))
        painter.setPen(QColor("#333333"))
        offset = 4
        painter.drawEllipse(round(self.rect().x() + offset), round(self.rect().y() + offset),
                            size - offset * 2, size - offset * 2)

        painter.drawLine(center_x, center_y, round(self.rect().x() + size), center_y)

        small_font = QFont()
        small_font.setPointSize(8)
        painter.setFont(small_font)
        painter.setPen(QColor("#333333"))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.name)
