import logging
import math
import random
import re
import sys
import time
from dataclasses import dataclass, replace
from typing import Optional

from PySide6.QtCore import QSize, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QLineEdit, \
    QPushButton, QMessageBox, QFileDialog, QComboBox, QCheckBox
from pydash import find_index

from src.adg import build_adg, to_adg_key
from src.common import x_y_to_index, distance_of_two_points
from src.domain import MapConfig, MapReq, State, MapfResult, RobotTaskReq, AgentTask, Cell, Location
from src.ecbs_resolver import ECBSResolver
from src.low_common import LowOp
from src.ui_cell import CellUi
from src.ui_robot import RobotWidget

log_file = 'mapf.log'
# noinspection PyBroadException
try:
    # 以读写模式打开文件
    with open(log_file, 'r+') as file:
        # 截断文件为 0 字节
        file.truncate(0)
except:
    pass

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file
)


def add_widget_with_label(layout: QHBoxLayout, widget, label_text):
    """
    Add a widget with a label to the layout.
    """
    label = QLabel(label_text)
    layout.addWidget(label)
    layout.addWidget(widget)


@dataclass
class Node:
    """
    A* node used by ui
    """
    state: State
    parent: Optional['Node']
    g: float
    h: float
    f: float


@dataclass
class RobotExePath:
    """
    Robot state during simulation.
    """
    s2Index: int  # the index of s2(target) state in the path
    adgKey: str  # s2 index to adg key
    timeStart: int  # the time start of the path
    timeEnd: int  # the time end of the path
    startOn: int  # the time point in mm of simulation start
    s1: State  # the start state
    s2: State  # the goal state
    x: int  # current position x
    y: int  # current position y
    head: int  # current robot head (degree)
    rotateDuration: int  # the time step after rotation done
    moveDuration: int  # the time step after movement done
    waitDuration: int  # the time step after waiting done
    p: float  # step progress 0 ~ 1
    holding: bool  # waiting for other robot according to ADG


@dataclass
class RobotPosition:
    x: int
    y: int
    head: int


@dataclass
class LowSearchCell:
    color: QColor
    order: int
    tool_tip: str = ""


@dataclass(frozen=True)
class LowSearchNode(Location):
    id: int
    parentId: int
    expandedIndex: int
    cellIndex: int
    log: str
    timeStart: int
    timeEnd: int


@dataclass
class PathCell:
    label: str
    log: str


class MapGridWidget(QWidget):
    def __init__(self, mapf_ui):
        super().__init__()
        self.mapf_ui = mapf_ui
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)

    def wheelEvent(self, event):
        # 缩放cell_size，限制范围
        delta = event.angleDelta().y()
        cell_size = self.mapf_ui.map_config.cellSize
        if delta > 0:
            cell_size = min(cell_size + 2, 80)
        else:
            cell_size = max(cell_size - 2, 10)
        self.mapf_ui.map_config.cellSize = cell_size
        self.mapf_ui.rebuild_map_cells()


class MapfUi:
    def __init__(self):
        self.map_config = MapConfig()
        self.map_req = MapReq(self.map_config, {})

        self.robot_colors: dict[str, QColor] = {}

        self.target_to_robot: dict[int, str] = {}

        # the plan result
        self.plan: Optional[MapfResult] = None

        self.robot_widgets: dict[str, RobotWidget] = {}  # by robot name

        self.adg_nodes: dict[str, list[str]] = {}
        self.finished_adg_nodes: set[str] = set()

        self.simulation = False
        self.sim_speed = 4
        self.current_time = 0  # in mm
        self.stepDurationVar = .1
        self.sim_robots: dict[str, RobotExePath] = {}

        self.low_search_index: dict[int, LowSearchCell] = {}
        self.low_search_nodes: dict[int, LowSearchNode] = {}
        self.low_search_my_path: dict[int, PathCell] = {}

        # 创建一个垂直布局
        main_layout = QVBoxLayout()
        self.main_layout = main_layout

        # 创建一个滚动区域
        scroll_area = QScrollArea()

        # 创建一个包含大量内容的部件
        content_widget = QWidget()
        content_layout = QVBoxLayout()

        # 地图操作按钮分组
        button_layout = QHBoxLayout()
        save_map_btn = QPushButton('保存地图')
        button_layout.addWidget(save_map_btn)
        save_map_btn.clicked.connect(self.save_map)

        open_map_btn = QPushButton('打开地图')
        button_layout.addWidget(open_map_btn)
        open_map_btn.clicked.connect(self.open_map)

        rebuild_map_btn = QPushButton('绘制地图')
        button_layout.addWidget(rebuild_map_btn)
        rebuild_map_btn.clicked.connect(self.draw_maps)

        # 新增清除障碍物按钮
        clear_obstacle_btn = QPushButton('清除障碍物')
        button_layout.addWidget(clear_obstacle_btn)
        clear_obstacle_btn.clicked.connect(self.clear_obstacles)

        # 新增随机障碍物按钮
        random_obstacle_btn = QPushButton('随机障碍物')
        button_layout.addWidget(random_obstacle_btn)
        random_obstacle_btn.clicked.connect(self.random_obstacles)

        content_layout.addLayout(button_layout, 0)

        # 参数输入分组
        line1 = QHBoxLayout()
        self.robot_num_edit = QLineEdit(str(self.map_config.robotNum))
        add_widget_with_label(line1, self.robot_num_edit, '机器人数量:')

        self.map_dim_x_edit = QLineEdit(str(self.map_config.mapDimX))
        add_widget_with_label(line1, self.map_dim_x_edit, '地图宽度:')

        self.map_dim_y_edit = QLineEdit(str(self.map_config.mapDimY))
        add_widget_with_label(line1, self.map_dim_y_edit, '地图高度:')

        self.cell_size_edit = QLineEdit(str(self.map_config.cellSize))
        add_widget_with_label(line1, self.cell_size_edit, '单元格大小:')

        self.obstacle_ratio_edit = QLineEdit(str(self.map_config.obstacleRatio))
        add_widget_with_label(line1, self.obstacle_ratio_edit, '障碍比例:')

        self.toggle_obstacle_cb = QCheckBox("障碍编辑")
        add_widget_with_label(line1, self.toggle_obstacle_cb, '（勾选后点击单元格可切换障碍）')

        content_layout.addLayout(line1, 0)

        # 目标与任务分组
        line2 = QHBoxLayout()
        self.target_num_edit = QLineEdit(str(self.map_config.targetNum))
        self.target_num_edit.setMaximumWidth(50)
        add_widget_with_label(line2, self.target_num_edit, '目标数量:')

        self.goal_stop_times_edit = QLineEdit(str(self.map_config.goalStopTimes))
        self.goal_stop_times_edit.setMaximumWidth(50)
        add_widget_with_label(line2, self.goal_stop_times_edit, '目标等待时间:')

        init_targets_btn = QPushButton('随机目标')
        line2.addWidget(init_targets_btn)
        init_targets_btn.clicked.connect(self.random_targets)

        self.tasks_edit = QLineEdit()
        add_widget_with_label(line2, self.tasks_edit, '任务:')

        content_layout.addLayout(line2, 0)

        # 算法与求解分组
        line3 = QHBoxLayout()
        self.low_resolver = QComboBox()
        self.low_resolver.addItems(["A*", "SIPP"])
        add_widget_with_label(line3, self.low_resolver, '低层算法:')

        self.w_edit = QLineEdit(str(self.map_config.w))
        self.w_edit.setMaximumWidth(50)
        add_widget_with_label(line3, self.w_edit, 'W:')

        resolve_btn = QPushButton('求解')
        line3.addWidget(resolve_btn)
        resolve_btn.clicked.connect(self.resolve)

        self.result_edit = QLineEdit()
        self.result_edit.setReadOnly(True)
        add_widget_with_label(line3, self.result_edit, '结果:')

        self.resolve_cost_edit = QLineEdit()
        self.resolve_cost_edit.setReadOnly(True)
        self.resolve_cost_edit.setMaximumWidth(60)
        add_widget_with_label(line3, self.resolve_cost_edit, '耗时:')

        content_layout.addLayout(line3, 0)

        # 仿真与路径分组
        line4 = QHBoxLayout()
        self.sim_speed_edit = QLineEdit(str(self.sim_speed))
        self.sim_speed_edit.setMaximumWidth(60)
        add_widget_with_label(line4, self.sim_speed_edit, '仿真速度:')

        sim_btn = QPushButton('开始仿真')
        self.sim_btn = sim_btn
        line4.addWidget(sim_btn)
        sim_btn.clicked.connect(self.toggle_sim)

        self.path_to_me_edit = QLineEdit()
        add_widget_with_label(line4, self.path_to_me_edit, '回溯路径:')
        self.path_to_me_edit.returnPressed.connect(self.set_path_to_me)

        self.load_low_search_btn = QPushButton('加载低层搜索')
        line4.addWidget(self.load_low_search_btn)
        self.load_low_search_btn.clicked.connect(self.load_low_search)

        self.max_timecost_edit = QLineEdit("60")
        add_widget_with_label(line4, self.max_timecost_edit, '最大耗时(秒):')

        content_layout.addLayout(line4, 0)

        self.init_obstacles()
        self.reset_robot_colors()

        self.map_grid = MapGridWidget(self)

        self.map_cells: list[CellUi] = []
        self.rebuild_map_cells()

        content_layout.addWidget(self.map_grid, 1)

        content_layout.addStretch(100000)

        content_widget.setLayout(content_layout)

        # 将内容部件设置到滚动区域中
        scroll_area.setWidget(content_widget)
        scroll_area.setWidgetResizable(True)
        self.scroll_area = scroll_area

        # 将滚动区域添加到主布局中
        main_layout.addWidget(scroll_area)

        # 创建主窗口
        main_window = QWidget()
        self.main_window = main_window

        main_window.setWindowTitle('多智能体路径规划(MAPF)')
        # main_window.setGeometry(100, 100, 400, 300)

        # 将布局设置到主窗口
        main_window.setLayout(main_layout)

        self.timer = QTimer(main_window)
        self.timer.timeout.connect(self.sim_loop)
        self.timer.start(int(1000 / 10))

        self.open_last_map()

    def do_inputs(self):
        self.map_config.robotNum = int(self.robot_num_edit.text())
        self.map_config.mapDimX = int(self.map_dim_x_edit.text())
        self.map_config.mapDimY = int(self.map_dim_y_edit.text())
        self.map_config.cellSize = int(self.cell_size_edit.text())
        self.map_config.obstacleRatio = float(self.obstacle_ratio_edit.text())
        self.map_config.w = float(self.w_edit.text())
        self.map_config.targetNum = int(self.target_num_edit.text())
        self.map_config.goalStopTimes = int(self.goal_stop_times_edit.text())

        tasks_str = self.tasks_edit.text()
        task_str_list = tasks_str.split(",")
        tasks: dict[str, RobotTaskReq] = {}
        for task_str in task_str_list:
            parts = task_str.split(":")
            if len(parts) != 3:
                continue
            robot_name = parts[0].strip()
            from_index = parts[1].strip()
            to_index = parts[2].strip()

            if not (robot_name and from_index and to_index):
                continue
            tasks[robot_name] = RobotTaskReq(int(from_index), int(to_index))
            self.target_to_robot[int(to_index)] = robot_name
        self.map_req.tasks = tasks

        self.sim_speed = int(self.sim_speed_edit.text())

        try:
            self.map_config.maxTimecost = float(self.max_timecost_edit.text())
        except Exception:
            self.map_config.maxTimecost = 60.0

        self.reset_robot_colors()

    def reset_robot_colors(self):
        self.robot_colors = {}
        for i in range(self.map_config.robotNum):
            hue = i * 137.508  # use golden angle approximation
            color = QColor()
            color.setHsl(int(hue), int(255 * .7), int(255 * .5))
            self.robot_colors[str(i)] = color

    def init_obstacles(self):
        """
        只产生数据，不更新 UI
        """

        self.do_inputs()

        map_dim_x: int = self.map_config.mapDimX
        map_dim_y: int = self.map_config.mapDimY
        obstacle_ratio: float = self.map_config.obstacleRatio

        cell_num = map_dim_x * map_dim_y
        obstacle_num = round(cell_num * obstacle_ratio)
        cells = [0] * cell_num
        for i in range(cell_num):
            cells[i] = 1 if i < obstacle_num else 0
        random.shuffle(cells)
        obstacles = set()
        for i in range(cell_num):
            if cells[i]:
                obstacles.add(i)

        self.map_config.obstacles = obstacles
        self.map_req.tasks = {}
        self.target_to_robot = {}
        self.plan = None

    def random_obstacles(self):
        print("random_obstacles")
        self.init_obstacles()
        self.rebuild_map_cells()

    def clear_obstacles(self):
        self.do_inputs()
        self.map_config.obstacles = set()
        self.map_req.tasks = {}
        self.target_to_robot = {}
        self.plan = None
        self.rebuild_map_cells()

    def rebuild_map_cells(self):
        for cell in self.map_cells:
            cell.setParent(None)
            cell.deleteLater()

        self.map_cells = []

        x_n = self.map_config.mapDimX
        y_n = self.map_config.mapDimY
        print(f"rebuild_map_cells, x={x_n}, y={y_n}")

        cell_size = self.map_config.cellSize
        self.map_grid.setFixedSize(QSize(x_n * (cell_size + 1), y_n * (cell_size + 1)))

        for x in range(x_n):
            for y in range(y_n):
                index = x_y_to_index(x, y, x_n)

                fill = QColor("#000000")  # 黑色障碍物
                label = ""
                tool_tip = ""
                obstacle = index in self.map_config.obstacles
                if obstacle:
                    fill = QColor("#000000")  # 黑色障碍物
                else:
                    # 优先显示低层搜索颜色
                    if self.low_search_index and index in self.low_search_index:
                        fill = self.low_search_index[index].color
                        label = self.low_search_index[index].tool_tip  # 可选：显示搜索信息
                    else:
                        robot = self.target_to_robot.get(index)
                        if robot is not None:
                            fill = self.robot_colors[robot]
                        else:
                            fill = QColor("#ffffff")  # 普通格子白色

                cell = CellUi(cell_size, index, x, y, fill, label, tool_tip,
                              lambda cx, cy, event=None: self.toggle_obstacle(cx, cy, event))
                self.map_cells.append(cell)
                cell.setParent(self.map_grid)
                cell.show()

        print("rebuild_map_cells, done")

    def random_targets(self):
        self.do_inputs()

        tasks: dict[str, RobotTaskReq] = {}
        target_to_robot: dict[int, str] = {}

        used_indexes: set[int] = set()
        config = self.map_config
        cell_num = config.mapDimX * config.mapDimY
        if config.robotNum * 2 >= cell_num - len(config.obstacles):
            QMessageBox.warning(self.main_window, "Warning", "No enough room.")

        for i in range(config.robotNum):
            from_index = -1
            retry_start = 0
            while from_index < 0 and retry_start < cell_num:
                retry_start += 1
                while from_index < 0:
                    n = random.randint(0, cell_num - 1)
                    if n in config.obstacles or n in used_indexes:
                        continue
                    used_indexes.add(n)
                    from_index = n

                s1 = State(
                    x=from_index % config.mapDimX, y=math.floor(from_index / config.mapDimX),
                    head=0, timeStart=-1, timeEnd=-1, type=0
                )
                task = RobotTaskReq(from_index, 0)
                tasks[str(i)] = task

                bad_indexes: set[int] = set()
                retry_target = 0
                for ti in range(config.targetNum):
                    to_index = -1
                    while to_index < 0 and retry_target < cell_num:
                        retry_target += 1
                        n = math.floor(random.random() * cell_num)
                        if n in self.map_config.obstacles or n in used_indexes or n in bad_indexes:
                            continue
                        if n == from_index:
                            continue

                        s2 = State(x=n % config.mapDimX, y=math.floor(n / config.mapDimX),
                                   head=0, timeStart=-1, timeEnd=-1, type=0, )
                        if not self.test_path(i, s1, s2):
                            bad_indexes.add(n)
                            continue

                        used_indexes.add(n)
                        to_index = n
                        task.toIndex = to_index
                        # task.toStates.append(state_to_cell(s2))

                        target_to_robot[n] = str(i)

                    if to_index < 0:
                        from_index = -1  # 重新选起点
                        break

            if from_index < 0:
                QMessageBox.warning(self.main_window, "Warning", "No good start.")
                return

        self.map_req.tasks = tasks
        self.target_to_robot = target_to_robot
        self.plan = None

        print(f"tasks: {tasks}")

        self.update_tasks_edit()

        self.rebuild_map_cells()
        self.update_robots_ui()

    def test_path(self, robot_name: int, from_state: State, to_state: State) -> bool:
        """
        Return true if there is a path from from_state to to_state.
        """
        print(f"find path: {robot_name}, {from_state} {to_state}")
        if from_state.x == to_state.x and from_state.y == to_state.y:
            return True
        open_set: list[Node] = [Node(state=from_state, g=0, h=0, f=0, parent=None)]
        close_set: list[Node] = []
        expanded_count = 0
        while open_set:
            expanded_count += 1
            top = open_set.pop(0)
            # print(f"expanded: [{expanded_count}]R={robot_name}|x={top.state.x}|y={top.state.y}|f={top.f}")
            if top.state.x == to_state.x and top.state.y == to_state.y:
                return True
            close_set.append(top)
            neighbors = self.get_neighbors(top.state)
            for n in neighbors:
                g = top.g + 1
                h = distance_of_two_points(n.x, n.y, to_state.x, to_state.y)
                f = g + h
                open_index = find_index(open_set, lambda node: node.state.x == n.x and node.state.y == n.y)
                if open_index >= 0 and g < open_set[open_index].g:
                    open_set.pop(open_index)
                    open_index = -1
                close_index = find_index(close_set, lambda node: node.state.x == n.x and node.state.y == n.y)
                if close_index >= 0 and g < close_set[close_index].g:
                    close_set.pop(close_index)
                    close_index = -1
                if open_index < 0 and close_index < 0:
                    new_node = Node(state=n, g=g, h=h, f=f, parent=top)
                    open_set.append(new_node)

            open_set = sorted(open_set, key=lambda node: node.f)
        return False

    def get_neighbors(self, state: State) -> list[State]:
        neighbors: list[State] = []
        self.add_neighbor(neighbors, state, 0, 1)
        self.add_neighbor(neighbors, state, 0, -1)
        self.add_neighbor(neighbors, state, 1, 0)
        self.add_neighbor(neighbors, state, -1, 0)

        return neighbors

    def add_neighbor(self, neighbors: list[State], state: State, dx: int, dy: int) -> None:
        if state.x + dx < 0 or state.x + dx >= self.map_req.config.mapDimX:
            return
        if state.y + dy < 0 or state.y + dy >= self.map_req.config.mapDimY:
            return
        if ((state.y + dy) * self.map_req.config.mapDimX + state.x + dx) in self.map_config.obstacles:
            return
        neighbors.append(
            State(x=state.x + dx, y=state.y + dy, head=0, timeStart=-1, timeEnd=-1, type=0)
        )

    def resolve(self):
        self.do_inputs()

        tasks: dict[str, AgentTask] = {}
        for robot_name, t in self.map_req.tasks.items():
            c1 = self.index_to_cell(t.fromIndex)
            c2 = self.index_to_cell(t.toIndex)
            tasks[robot_name] = AgentTask(robot_name,
                                          fromState=c1,
                                          toStates=[c2],
                                          stopTimes=self.map_config.goalStopTimes)

        high_resolver = ECBSResolver(
            w=self.map_config.w,
            map_dim_x=self.map_config.mapDimX,
            map_dim_y=self.map_config.mapDimY,
            obstacles=self.map_config.obstacles,
            tasks=tasks,
            low_resolver=self.low_resolver.currentIndex(),
            update_cost=self.update_cost,
            max_timecost=self.map_config.maxTimecost,  # 传递最大耗时
        )
        try:
            r = high_resolver.search()
            self.plan = r
        except Exception as e:
            logging.exception(f"Resolve exception: {e}")
            QMessageBox.warning(self.main_window, "Warning", str(e))
            return

        # noinspection PyUnresolvedReferences
        self.result_edit.setText(self.plan.to_json())
        print("Plan: " + str(r))

        msg = "Success" if r.ok else f"Fail：{r.msg}"
        QMessageBox.information(self.main_window, '完成', msg)

    def toggle_sim(self):
        self.do_inputs()

        if self.simulation:
            self.stop_sim()
        else:
            if not (self.plan and self.plan.ok):
                QMessageBox.warning(self.main_window, "Warning", "No ok plan.")

            self.sim_robots = {}
            for robot_name, plan in self.plan.plans.items():
                self.sim_robots[robot_name] = self.build_robot_exe_path(robot_name, 1, plan.path[0], plan.path[1])

            self.simulation = True
            self.sim_btn.setText('Stop Sim')
            self.adg_nodes = build_adg(self.plan)
            self.finished_adg_nodes.clear()

    def stop_sim(self):
        self.simulation = False
        self.sim_btn.setText('Start Sim')

    def sim_loop(self):
        if not self.simulation:
            return

        self.do_inputs()

        now = round(time.time() * 1000)
        robot_names = self.plan.plans.keys()

        # 第一轮循环，先推下进度
        for robot_name in robot_names:
            sim_robot = self.sim_robots[robot_name]
            duration = sim_robot.rotateDuration + sim_robot.moveDuration + sim_robot.waitDuration
            if duration <= 0:
                duration = 1000
            time_pass = now - sim_robot.startOn
            print(f"duration={duration}, time_pass={time_pass}")
            sim_robot.p = time_pass / duration * self.sim_speed
            if sim_robot.p >= 1:
                sim_robot.p = 1
                sim_robot.holding = True
                print(f"done ADG node {sim_robot.adgKey}")
                self.finished_adg_nodes.add(sim_robot.adgKey)
                if sim_robot.timeEnd > self.current_time:
                    self.current_time = sim_robot.timeEnd

            rp = self.get_position(sim_robot)
            sim_robot.x = rp.x
            sim_robot.y = rp.y
            sim_robot.head = rp.head

        # 分配下一步
        all_done = True
        for robot_name in robot_names:
            sim_robot = self.sim_robots[robot_name]
            if sim_robot.p < 1:
                all_done = False
                continue
            path = self.plan.plans[robot_name].path
            if not path:
                continue
            next_index = sim_robot.s2Index + 1
            s1 = path[sim_robot.s2Index] if sim_robot.s2Index < len(path) else None
            s2 = path[next_index] if next_index < len(path) else None
            if not s1 or not s2:
                continue
            all_done = False
            adg_key = to_adg_key(robot_name, next_index)
            dependents = self.adg_nodes.get(adg_key)
            dependents_all_pass = True
            if dependents:
                for d in dependents:
                    if d not in self.finished_adg_nodes:
                        dependents_all_pass = False
                        break
            if dependents_all_pass:
                self.sim_robots[robot_name] = self.build_robot_exe_path(robot_name, next_index, s1, s2)
                print(f"release ADG node {sim_robot.adgKey}")

        self.update_robots_ui()

        if all_done:
            print(f"Sim done")
            self.stop_sim()
            return

    def build_robot_exe_path(self, robot_name: str, s2_index: int, s1: State, s2: State) -> RobotExePath:
        dx = s2.x - s1.x
        dy = s2.y - s1.y
        def calc_direction_angle(dx, dy):
            if dx == 1 and dy == 0:
                return 0
            elif dx == 0 and dy == 1:
                return 90
            elif dx == -1 and dy == 0:
                return 180
            elif dx == 0 and dy == -1:
                return 270
            else:
                return s1.head  # 原地等待或特殊动作保持原朝向
        # 判断是否移动
        if dx != 0 or dy != 0:
            direction = calc_direction_angle(dx, dy)
            s1 = replace(s1, head=direction)
            s2 = replace(s2, head=direction)
        # 需要转的角度，初始，-270 ~ +270
        d_head = abs(s2.head - s1.head)
        if d_head > 180:
            d_head = 360 - d_head
        d_head /= 90
        rotate_time_num = math.ceil(d_head)
        move_time_num = abs(s1.x - s2.x + s1.y - s2.y)
        wait_time_num = s2.timeEnd - s2.timeStart + 1 - rotate_time_num - move_time_num
        rotate_duration = rotate_time_num * 1000 * (1 + random.random() * self.stepDurationVar)
        move_duration = move_time_num * 1000 * (1 + random.random() * self.stepDurationVar)
        cell_size = self.map_config.cellSize
        return RobotExePath(
            s2Index=s2_index,
            adgKey=to_adg_key(robot_name, s2_index),
            timeStart=s1.timeStart or 0,
            timeEnd=s2.timeEnd or 0,
            startOn=round(time.time() * 1000),
            s1=s1, s2=s2,
            rotateDuration=round(rotate_duration),
            moveDuration=round(move_duration),
            waitDuration=wait_time_num * 1000,
            p=0,
            x=s1.x * (cell_size + 1),
            y=s1.y * (cell_size + 1),
            head=s1.head,
            holding=False
        )

    def get_position(self, sim_robot: RobotExePath) -> RobotPosition:
        s1 = sim_robot.s1
        s2 = sim_robot.s2
        p_rotate = 1
        p_move = 1
        time_pass = round(time.time() * 1000) - sim_robot.startOn
        if sim_robot.rotateDuration > 0:
            p_rotate = time_pass / sim_robot.rotateDuration * self.sim_speed
            if p_rotate > 1:
                p_rotate = 1
        if sim_robot.moveDuration > 0:
            p_move = (time_pass - sim_robot.rotateDuration * self.sim_speed) / sim_robot.moveDuration * self.sim_speed
            if p_move > 1:
                p_move = 1
            if p_move < 0:
                p_move = 0
        cell_size = self.map_config.cellSize
        # 计算插值位置
        cur_x = s1.x + p_move * (s2.x - s1.x)
        cur_y = s1.y + p_move * (s2.y - s1.y)
        # 计算插值方向
        dx = s2.x - s1.x
        dy = s2.y - s1.y
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            # 有移动，计算当前插值方向
            angle = math.degrees(math.atan2(dy, dx))
            head = (angle + 360) % 360
        else:
            # 没有移动，保持原朝向
            head = s2.head
        return RobotPosition(
            x=round(cur_x * (cell_size + 1)),
            y=round(cur_y * (cell_size + 1)),
            head=head
        )

    def update_robots_ui(self):
        for (robot_name, r_ui) in self.robot_widgets.items():
            r_ui.setParent(None)
            r_ui.deleteLater()

        self.robot_widgets = {}

        robot_positions: dict[str, RobotPosition] = {}
        cell_size = self.map_config.cellSize
        if self.simulation and self.plan and self.plan.ok:
            for robot_name, sim_robot in self.sim_robots.items():
                robot_positions[robot_name] = RobotPosition(x=sim_robot.x,
                                                            y=sim_robot.y,
                                                            head=sim_robot.head)
        else:
            for robot_name, task in self.map_req.tasks.items():
                from_cell = self.index_to_cell(task.fromIndex)
                robot_positions[robot_name] = RobotPosition(
                    x=from_cell.x * (cell_size + 1),
                    y=from_cell.y * (cell_size + 1),
                    head=0)

        ri = 0
        for (robot_name, p) in robot_positions.items():
            print(f"robot {robot_name} position: {p}")

            color = self.robot_colors[robot_name]
            r_ui = RobotWidget(robot_name, cell_size, p.x, p.y, p.head, color, self.map_grid)
            self.robot_widgets[robot_name] = r_ui
            r_ui.show()

            ri += 1

    def toggle_obstacle(self, x: int, y: int, event=None):
        if not self.toggle_obstacle_cb.isChecked():
            return
        print(f"toggle obstacle: {x}, {y}, event={event}")
        print("before:", self.map_config.obstacles)
        index = x_y_to_index(x, y, self.map_config.mapDimX)
        if index in self.map_config.obstacles:
            self.map_config.obstacles.remove(index)
        else:
            self.map_config.obstacles.add(index)
        print("after:", self.map_config.obstacles)
        self.rebuild_map_cells()
        self.map_grid.update()

    def save_map(self):
        self.do_inputs()

        file_path, _ = QFileDialog.getSaveFileName(self.main_window, '保存地图', '', 'JSON 文件 (*.map.json)')
        # noinspection PyUnresolvedReferences
        txt = self.map_req.to_json()
        try:
            with open(file_path, 'w', encoding='utf-8') as map_file:
                map_file.write(txt)
            print('写入成功')
        except Exception as e:
            print(f'写入文件时出错: {e}')

        file_path = "last.map.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as map_file:
                map_file.write(txt)
            print('写入成功')
        except Exception as e:
            print(f'写入文件时出错: {e}')

    def open_map(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, '选择文件', '', 'JSON 文件 (*.map.json)')
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as map_file:
                    content = map_file.read()
                    self.load_map(content)
            except Exception as e:
                print(f'打开文件时出错: {e}')

    def open_last_map(self):
        file_path = "last.map.json"
        try:
            with open(file_path, 'r', encoding='utf-8') as map_file:
                content = map_file.read()
                self.load_map(content)
        except Exception as e:
            print(f'打开文件时出错: {e}')

    def load_map(self, content):
        req: MapReq = MapReq.from_json(content)
        self.map_req = req
        self.map_config = self.map_req.config

        self.target_to_robot = {}
        for (robot_name, t) in req.tasks.items():
            self.target_to_robot[t.toIndex] = robot_name

        self.stop_sim()

        self.reset_robot_colors()

        self.update_tasks_edit()

        self.rebuild_map_cells()

        self.update_robots_ui()

        self.robot_num_edit.setText(str(self.map_config.robotNum))
        self.map_dim_x_edit.setText(str(self.map_config.mapDimX))
        self.map_dim_y_edit.setText(str(self.map_config.mapDimY))
        self.obstacle_ratio_edit.setText(str(self.map_config.obstacleRatio))
        self.w_edit.setText(str(self.map_config.w))
        self.target_num_edit.setText(str(self.map_config.targetNum))
        self.goal_stop_times_edit.setText(str(self.map_config.goalStopTimes))
        self.sim_speed_edit.setText(str(self.sim_speed))

    def update_tasks_edit(self):
        tasks = [f"{robot_name}:{task.fromIndex}:{task.toIndex}" for robot_name, task in self.map_req.tasks.items()]
        tasks_str = ", ".join(tasks)
        self.tasks_edit.setText(tasks_str)

    def index_to_cell(self, index: int) -> Cell:
        x = index % self.map_config.mapDimX
        y = math.floor(index / self.map_config.mapDimX)
        return Cell(x, y)

    def update_cost(self, cost: float):
        self.resolve_cost_edit.setText('%.1f' % cost)

    def load_low_search(self):
        if self.low_search_index:
            self.load_low_search_btn.setText('Load Low Search')
            self.low_search_index = None
            self.low_search_nodes = None
            self.low_search_my_path = None
        else:
            self.load_low_search_btn.setText('Clear Low Search')
            self.stop_sim()

            file_path, _ = QFileDialog.getOpenFileName(self.main_window, 'Select log file', '', 'JSON file (*.json)')
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as record_file:
                        content = record_file.read()
                        op: LowOp = LowOp.from_json(content)
                        self.low_search_index = {}
                        self.low_search_nodes = {}
                        for es in op.expandedList:
                            # 1|9,2@90|2:2|3|g=2.0|f=17.0|f2=0|id=2|p=1| <- 9,1@90|1:1|4
                            # 1|1:3|327|27,10|False|g=3.0|f=22.0|f2=0|id=1|p=0| <- 0:0|328|28,10|False
                            g = re.match(r"(\d+)\|(\d+),(\d+)@(\d+)\|(\d+):(\d+)\|(\d+)\|.+\|id=(\w+)\|p=(\w+)", es)
                            if not g:
                                continue
                            expanded_index = int(g.group(1))
                            x = int(g.group(2))
                            y = int(g.group(3))
                            # head = int(g.group(4))
                            time_start = int(g.group(5))
                            time_end = int(g.group(6))
                            # state_type = int(g.group(7))
                            node_id = int(g.group(8))
                            parent_id = int(g.group(9))

                            cell_index = x + y * self.map_config.mapDimX

                            lsn = LowSearchNode(id=node_id, parentId=parent_id, expandedIndex=expanded_index,
                                                timeStart=time_start, timeEnd=time_end, x=x, y=y, cellIndex=cell_index,
                                                log=es)
                            self.low_search_nodes[node_id] = lsn

                            color = QColor("red")
                            alpha = float(expanded_index) / op.expandedNum * .8 + .2
                            color.setAlpha(round(255 * alpha))
                            lsc = self.low_search_index.get(cell_index)
                            if lsc:
                                lsc.tool_tip += f"\n{es}"
                            else:
                                self.low_search_index[cell_index] = LowSearchCell(color, expanded_index, es)
                        self.low_search_index[op.startIndex] = LowSearchCell(QColor("blue"), 0)
                        self.low_search_index[op.goalIndex] = LowSearchCell(QColor("green"), 0)
                except Exception as e:
                    print(f'打开文件时出错: {e}')
                    return
        self.rebuild_map_cells()

    def set_path_to_me(self):
        self.low_search_my_path = {}
        node_id = int(self.path_to_me_edit.text())
        if node_id != -1:
            node = self.low_search_nodes.get(node_id)
            if not node:
                QMessageBox.warning(self.main_window, "Warning", f"No node {node_id}")
                return
            while node:
                self.low_search_my_path[node.cellIndex] = PathCell(label=str(node.id), log=node.log)
                node = self.low_search_nodes.get(node.parentId)
            logging.debug(f"Path to me: {self.low_search_my_path}")

        self.rebuild_map_cells()

    def draw_maps(self):
        self.do_inputs()
        self.rebuild_map_cells()


def main():
    app = QApplication(sys.argv)

    mapf_ui = MapfUi()
    mapf_ui.main_window.resize(960, 600)
    mapf_ui.main_window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
