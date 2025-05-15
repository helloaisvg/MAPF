from typing import Callable

from PySide6.QtGui import QMouseEvent, QPainter, QColor, QFont, Qt
from PySide6.QtWidgets import QWidget


class CellUi(QWidget):
    def __init__(self, cell_size: int, cell_index: int, x: int, y: int, fill: QColor, label: str, tool_tip: str,
                 click_callback):
        super().__init__()

        self.cell_size = cell_size
        self.cell_index = cell_index
        self.cell_x = x
        self.cell_y = y
        self.fill = fill
        self.label = label
        self.click_callback = click_callback

        self.setFixedSize(cell_size, cell_size)
        self.move(x * (cell_size + 1), y * (cell_size + 1))

        self.setToolTip(tool_tip)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.fill)
        painter.drawRect(self.rect().x(), self.rect().y(), self.cell_size, self.cell_size)

        small_font = QFont()
        small_font.setPointSize(10)
        painter.setFont(small_font)
        painter.setPen(QColor("#333333"))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignLeading, str(self.cell_index))

        if self.label:
            painter.setPen(QColor("#000"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.label)

    def mousePressEvent(self, event: QMouseEvent):
        print("cell clicked", self.cell_x, self.cell_y, event.button())
        if event.button() == 2:  # 右键点击
            if self.click_callback:
                self.click_callback(self.cell_x, self.cell_y, event)
        if event.button() == 1:  # 左键点击
            if self.click_callback:
                self.click_callback(self.cell_x, self.cell_y, event)
