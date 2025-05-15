import heapq
import logging
import time
from dataclasses import replace, dataclass
from math import ceil
from typing import Optional

from dataclasses_json import dataclass_json
from pydash import find_index

from src.common import is_time_overlay, is_same_location_of_state, vertex_key, edge_key
from src.conflicts import count_transition_conflicts
from src.domain import State, TargetOnePlanResult
from src.low_common import LowContext, LowOp


@dataclass_json
@dataclass(frozen=True)
class LowNode:
    id: int
    state: State
    parent: Optional['LowNode'] = None
    g: float = 0.0  # 到这个节点的成本
    f: float = 0.0
    focalValue: float = 0.0  # 注意，这个不是一定是 f 值

    def desc(self):
        parent: Optional[LowNode] = self.parent
        parent_desc = ""
        parent_id = -1
        if parent:
            parent_desc = str(parent.state)
            parent_id = parent.id
        return (f"{self.state}|g={self.g}|f={self.f}|f2={self.focalValue}|id={self.id}|p={parent_id}|"
                f" <- {parent_desc}")


class OpenLowNode:
    """
    排序：lowest fScore，highest gScore
    """

    def __init__(self, n: LowNode):
        self.n = n

    def __lt__(self, other: 'OpenLowNode'):
        if self.n.f != other.n.f:
            return self.n.f < other.n.f
        elif self.n.g != other.n.g:
            return self.n.g > other.n.g
        else:
            return self.n.id < other.n.id


class FocalLowNode:
    """
    # Sort order (see "Improved Solvers for Bounded-Suboptimal Multi-Agent Path Finding" by Cohen et. al.)
    # 1. lowest focalHeuristic
    # 2. lowest fScore
    # 3. highest gScore
    """

    def __init__(self, n: LowNode):
        self.n = n

    def __lt__(self, other: 'FocalLowNode'):
        if self.n.focalValue != other.n.focalValue:
            return self.n.focalValue < other.n.focalValue
        elif self.n.f != other.n.f:
            return self.n.f < other.n.f
        elif self.n.g != other.n.g:
            return self.n.g > other.n.g
        else:
            return self.n.id < other.n.id


class LowResolver:
    """
    单目标
    有界最优，Focal Search A*
    f, g, h 单位是时间，表示时间成本
    """

    def __init__(self, ctx: LowContext, time_offset: int,
                 start_state: State, goal_state: State,
                 last_goal_constraint: int = -1,
                 max_timecost: float = 60.0):
        """
        :param time_offset: 起始时刻
        :param start_state:
        :param goal_state:
        :param last_goal_constraint: 如果目标位置被约束了，last_goal_constraint 是被约束的最后一个时刻
        """
        print(f"Search one: robot={ctx.robotName}, offset={time_offset}, start={start_state}, goal={goal_state}, "
              f"constraints={ctx.constraints}")
        self.ctx = ctx
        self.time_offset = time_offset
        self.start_state = start_state
        self.goal_state = goal_state
        self.last_goal_constraint = last_goal_constraint
        self.max_timecost = max_timecost
        self._start_time = None

        self.node_id = 0

        # 计算最晚约束的时间
        self.constraint_time_end = -1
        if ctx.constraints:
            for cs in ctx.constraints.values():
                for c in cs:
                    if c.timeEnd > self.constraint_time_end:
                        self.constraint_time_end = c.timeEnd

        self.op = LowOp(
            robotName=ctx.robotName,
            highId=ctx.highId,
            w=ctx.w,
            mapDimX=ctx.mapDimX,
            mapDimY=ctx.mapDimY,
            # obstacles=ctx.obstacles,
            moveUnitCost=ctx.moveUnitCost,
            rotateUnitCost=ctx.rotateUnitCost,
            goalStopTimeNum=ctx.goalStopTimeNum,
            startCell=f"{start_state.x},{start_state.y}",
            goalCell=f"{goal_state.x},{goal_state.y}",
            startIndex=start_state.x + ctx.mapDimX * start_state.y,
            goalIndex=goal_state.x + ctx.mapDimX * goal_state.y,
            constraints=ctx.constraints,
            ok=False,
            errMsg="",
            path=None,
            expandedNum=0,
            expandedList=[],
            openSize=[],
            focalSize=[],
            logs=[],
            warnings=[],
            startedOn=time.time()
        )

        self.open_set: list[OpenLowNode] = []
        self.open_dict: dict[str, LowNode] = {}  # open查重dict
        self.focal_set: list[FocalLowNode] = []
        self.closed_set: dict[str, LowNode] = {}  # by key

    def resolve_one(self):
        r = self.do_resolve_one()

        # 求解过程写入日志
        op = self.op
        op.expandedNum = len(op.expandedList)
        op.endedOn = time.time()
        op.timeCost = op.endedOn - op.startedOn
        op.ok = r.ok
        op.errMsg = r.reason  # 修正：使用 r.reason 而不是 r.msg 或 r.errMsg
        op.path = [str(s) for s in r.path] if r.ok else None

        # noinspection PyUnresolvedReferences
        txt = op.to_json(indent=2)
        file = f"op/high-{op.highId}-r-{op.robotName}-{op.ok}-{op.expandedNum}.json"
        try:
            with open(file, 'w', encoding='utf-8') as file:
                file.write(txt)
        except Exception as e:
            logging.error(f'Failed to write out low resolver op: {e}')

        return r

    def do_resolve_one(self) -> TargetOnePlanResult:
        self._start_time = time.time()
        # 初始节点
        start_node = LowNode(
            self.node_id,
            replace(self.start_state, timeStart=self.time_offset, timeEnd=self.time_offset, type=1),
            None,
            g=0.0,
            f=self.admissible_heuristic(self.start_state),
            focalValue=0.0,
        )
        heapq.heappush(self.open_set, OpenLowNode(start_node))
        self.open_dict[start_node.state.state_time_key()] = start_node  # 新增dict
        heapq.heappush(self.focal_set, FocalLowNode(start_node))

        while self.open_set:
            if time.time() - self._start_time > self.max_timecost:
                return TargetOnePlanResult(
                    self.ctx.robotName,
                    False,
                    f"Timeout: 超过最大耗时{self.max_timecost}s",
                    planCost=time.time() - self.op.startedOn,
                    fromState=self.start_state,
                    toState=self.goal_state,
                )
            # 每次进来重建 focal set，因此后续都不处理向 focal set 添加元素
            self.rebuild_focal_set()

            # 取出下一个有界最优启发值最低的节点
            min_focal_n: LowNode = heapq.heappop(self.focal_set).n
            min_focal_s = min_focal_n.state

            # 记录展开过程
            self.op.expandedList.append(str(self.op.expandedNum) + "|" + min_focal_n.desc())
            self.op.expandedNum += 1
            self.op.openSize.append(len(self.open_set))
            self.op.focalSize.append(len(self.focal_set))

            # 从 open 中删除
            self.remove_open_node(min_focal_n)

            # 已展开的节点加入到 close
            self.closed_set[min_focal_s.state_time_key()] = min_focal_n

            # 如果限制最大展开次数，超过则结束
            expanded_max = self.get_expanded_max()
            if 0 < expanded_max < self.op.expandedNum:
                return TargetOnePlanResult(
                    self.ctx.robotName,
                    False,
                    "Expanded nodes too many",  # 这里使用 reason
                    planCost=time.time() - self.op.startedOn,
                    fromState=self.start_state,
                    toState=self.goal_state,
                )

            # 找到解：到达目标位置
            if (is_same_location_of_state(min_focal_s, self.goal_state)
                    and min_focal_s.timeEnd > self.last_goal_constraint):
                return self.build_ok_result(min_focal_n)

            # 扩展节点
            neighbors = self.get_neighbors(min_focal_s)
            for neighbor in neighbors:
                self.new_open_node(min_focal_n, neighbor)

        return TargetOnePlanResult(
            self.ctx.robotName,
            False,
            "All states checked",  # 这里使用 reason
            planCost=time.time() - self.op.startedOn,
            expandedCount=self.op.expandedNum,
            fromState=self.start_state,
            toState=self.goal_state,
        )

    def rebuild_focal_set(self):
        """
        整体重建 focal 集合
        """
        min_f = self.open_set[0].n.f  # 读取 open 最小的节点，但不删除
        self.focal_set = []
        bound = min_f * self.ctx.w
        for node in self.open_set:
            if node.n.f <= bound:
                heapq.heappush(self.focal_set, FocalLowNode(node.n))

    def get_expanded_max(self):
        """
        限制最多展开次数。-1 表示不限制。
        """
        return -1
        # return self.ctx.mapDimX * self.ctx.mapDimY

    def new_open_node(self, from_node: LowNode, neighbor: State):
        """
        构造新节点并添加到 open 集合
        """

        if neighbor.type != 5:
            if self.detect_loop(from_node, neighbor):
                self.op.logs.append(f"Loop, return, {neighbor}")
                return

        g = from_node.g + neighbor.timeEnd - neighbor.timeStart + 1
        f = g + self.admissible_heuristic(neighbor)
        focal_value = self.calc_focal_value(from_node, neighbor, g, f)
        self.node_id += 1
        node = LowNode(self.node_id, neighbor, from_node, g=g, f=f, focalValue=focal_value)

        cell_index = self.state_to_index(neighbor.x, neighbor.y)
        state_time_key = neighbor.state_time_key()
        old_closed = self.closed_set.get(state_time_key)
        if old_closed:
            if f < old_closed.f:
                self.op.warnings.append(f"SmallerFInClosed|{cell_index}|{neighbor.x},{neighbor.y}|"
                                        f"{neighbor.timeStart}:{neighbor.timeEnd}|"
                                        f"{old_closed.state.timeStart}:{old_closed.state.timeEnd}|"
                                        f"{f}|{old_closed.f}")
                del self.closed_set[state_time_key]
            else:
                return
        old_open = self.open_dict.get(state_time_key)
        if old_open:
            if f >= old_open.f:
                return
            node = replace(old_open, g=g, f=f, focalValue=focal_value)
            self.replace_open_node(OpenLowNode(node))
        else:
            heapq.heappush(self.open_set, OpenLowNode(node))
            self.open_dict[state_time_key] = node  # 新增

    def admissible_heuristic(self, state: State) -> float:
        """
        改进的启发式函数，考虑距离与旋转
        """
        # h_val: 曼哈顿距离+旋转启发式
        dx = abs(state.x - self.goal_state.x)
        dy = abs(state.y - self.goal_state.y)
        h = dx + dy
        # 加上旋转代价
        if hasattr(state, 'head') and hasattr(self.goal_state, 'head'):
            dh = abs(state.head - self.goal_state.head) // 90
            h += dh
        return h

    def calc_focal_value(self, from_node: LowNode, to_state: State, to_state_g: float, to_state_f: float) -> float:
        """
        计算 focal 值。
        """

        # case 1
        # return to_state_g

        # case 2
        # return to_state_f

        # case 3 计算这个状态转移与其他机器人的冲突数
        # 假设约束当前机器人的其他机器人有 B、C、D，因为前面检查状态转移有效性时，已保证不与这些机器人冲突，
        # 因此这里计算的是与 B、C、D 之外的其他机器人的冲突
        return count_transition_conflicts(self.ctx.robotName, from_node.state, to_state, self.ctx.oldAllPaths)

    def get_neighbors(self, from_state: State) -> list[State]:
        neighbors = []

        # 如果时间点已经晚于最后的约束，没必要等了
        if from_state.timeEnd <= self.constraint_time_end:
            self.add_valid_neighbor(neighbors, from_state, 0, 0, from_state.head)  # waiting

        self.add_valid_neighbor(neighbors, from_state, 1, 0, 0)
        self.add_valid_neighbor(neighbors, from_state, -1, 0, 180)
        self.add_valid_neighbor(neighbors, from_state, 0, 1, 90)
        self.add_valid_neighbor(neighbors, from_state, 0, -1, 270)
        return neighbors

    def add_valid_neighbor(self, neighbors: list[State], from_state: State, dx: int, dy: int, to_head: int):
        """
        如果移动有效，则添加到 neighbors
        toHead 目标车头朝向
        """
        x = from_state.x + dx
        y = from_state.y + dy

        # 在地图范围内且不是障碍物
        if (x < 0 or x >= self.ctx.mapDimX or y < 0 or y >= self.ctx.mapDimY
                or self.state_to_index(x, y) in self.ctx.obstacles):
            return

        # 需要转的角度，初始，-270 ~ +270
        d_head = abs(to_head - from_state.head)
        # 270 改成 90
        if d_head > 180:
            d_head = 360 - d_head

        d_head /= 90

        # 耗时，也作为 g 的增量
        # 假设 dx/dy 1 是 1 米
        time_num = ceil(float(abs(dx + dy)) / self.ctx.moveUnitCost + float(d_head) / self.ctx.rotateUnitCost)

        if time_num < 1:
            time_num = 1  # 原地等待

        new_state = State(
            x=x,
            y=y,
            head=to_head,
            timeStart=from_state.timeEnd + 1,
            timeEnd=from_state.timeEnd + time_num,
            type=5 if dx == 0 and dy == 0 else 3,
        )

        # 最后一步要等待
        test_state = new_state
        if is_same_location_of_state(self.goal_state, new_state):
            # 增加等待时间
            test_state = replace(new_state, timeEnd=new_state.timeEnd + self.ctx.goalStopTimeNum)

        # 额外检查是否有效，主要是检查约束
        if not (self.state_valid(test_state) and self.transition_valid(from_state, test_state)):
            return

        neighbors.append(new_state)

    def state_valid(self, to_state: State) -> bool:
        cs = self.ctx.constraints
        if not cs:
            return True

        v_key = vertex_key(to_state.x, to_state.y)

        constraints = cs.get(v_key)
        if not constraints:
            return True

        for c in constraints:
            if is_time_overlay(to_state.timeStart, to_state.timeEnd, c.timeStart, c.timeEnd):
                return False
        return True

    def transition_valid(self, from_state: State, to_state: State) -> bool:
        if is_same_location_of_state(from_state, to_state):
            return True

        cs = self.ctx.constraints
        if not cs:
            return True

        e_key = edge_key(from_state.x, from_state.y, to_state.x, to_state.y)

        e_constraints = cs.get(e_key)
        if not e_constraints:
            return True

        for c in e_constraints:
            # !!!! from_state.timeEnd
            if is_time_overlay(from_state.timeEnd, to_state.timeEnd, c.timeStart, c.timeEnd):
                return False
        return True

    def state_to_index(self, x: int, y: int):
        return x + y * self.ctx.mapDimX

    def find_open_node_by_key(self, key: str) -> Optional[LowNode]:
        for node in self.open_set:
            n = node.n
            if n.state.state_time_key() == key:
                return n
        return None

    def remove_open_node(self, node: LowNode):
        index = find_index(self.open_set, lambda n: n.n.id == node.id)
        if index < 0:
            return
        else:
            self.open_set[index] = self.open_set[-1]
            self.open_set.pop()
            heapq.heapify(self.open_set)
            # 同步删除dict
            k = node.state.state_time_key()
            if k in self.open_dict:
                del self.open_dict[k]

    def replace_open_node(self, new_one: OpenLowNode):
        index = find_index(self.open_set, lambda n: n.n.id == new_one.n.id)
        if index < 0:
            heapq.heappush(self.open_set, new_one)
        else:
            self.open_set[index] = new_one
            heapq.heapify(self.open_set)
        # 同步更新dict
        k = new_one.n.state.state_time_key()
        self.open_dict[k] = new_one.n

    def build_ok_result(self, min_focal_n: LowNode):
        # 达到时间在最后一次目标点被约束的时刻后
        # 到达后等待一段时间
        last_state = min_focal_n.state
        path = [last_state]
        curr_node = min_focal_n.parent
        while curr_node:
            path.append(curr_node.state)
            curr_node = curr_node.parent
        path.reverse()

        goal_stop_time_num = self.ctx.goalStopTimeNum if self.ctx.goalStopTimeNum > 0 else 1

        # 最后追加一个原地等待的，模拟动作时间
        # noinspection PyTypeChecker
        action_state = replace(last_state,
                               timeStart=last_state.timeEnd + 1,
                               timeEnd=last_state.timeEnd + goal_stop_time_num)
        path.append(action_state)

        return TargetOnePlanResult(
            self.ctx.robotName,
            True,
            cost=min_focal_n.g + goal_stop_time_num,
            planCost=time.time() - self.op.startedOn,
            expandedCount=self.op.expandedNum,
            timeNum=action_state.timeEnd,
            timeStart=self.time_offset,
            timeEnd=action_state.timeEnd,
            fromState=self.start_state,
            toState=self.goal_state,
            path=path
        )

    @staticmethod
    def detect_loop(from_node: LowNode, neighbor: State):
        """
        检查路径中是否出现过此位置
        """
        n = from_node
        while n:
            if is_same_location_of_state(n.state, neighbor):
                return True
            n = n.parent
        return False
