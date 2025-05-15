import heapq
import logging
import time
from dataclasses import replace, dataclass
from typing import Optional

from dataclasses_json import dataclass_json

from src.common import is_time_overlay, vertex_key, edge_key, is_same_location_of_state, is_same_location
from src.conflicts import Constraint
from src.domain import State, TargetOnePlanResult
from src.low_common import LowContext, LowOp
from src.safe_interval import Interval, MAX_TIME, constraints_to_safe_intervals


@dataclass_json
@dataclass(frozen=True)
class SippNode:
    id: int
    state: State
    vertexSafeInterval: Interval
    parent: Optional['SippNode'] = None
    g: float = 0.0  # 到这个节点的成本
    f: float = 0.0

    def __lt__(self, other: 'SippNode'):
        if self.f != other.f:
            return self.f < other.f
        elif self.g != other.g:
            return self.g > other.g
        else:
            return self.id < other.id

    def desc(self):
        parent: Optional[SippNode] = self.parent
        parent_desc = ""
        parent_id = -1
        if parent:
            parent_desc = str(parent.state)
            parent_id = parent.id
        return (f"{self.state}|g={self.g}|f={self.f}|f2=0|id={self.id}|p={parent_id}|"
                f" <- {parent_desc}")


class SippResolver:

    def __init__(self, ctx: LowContext, time_offset: int,
                 start_state: State, goal_state: State,
                 last_goal_constraint: int = -1):
        self.ctx = ctx
        self.time_offset = time_offset
        self.start_state = start_state
        self.goal_state = goal_state
        self.last_goal_constraint = last_goal_constraint

        logging.debug(f"SIPP, h={self.ctx.highId}, r={self.ctx.robotName}")

        self.node_id = 0

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

        self.open_set: list[SippNode] = []
        self.closed_set: dict[str, SippNode] = {}  # by key

    def resolve_one(self):
        r = self.do_resolve_one()

        # 求解过程写入日志
        op = self.op
        op.expandedNum = len(op.expandedList)
        op.endedOn = time.time()
        op.timeCost = op.endedOn - op.startedOn
        op.ok = r.ok
        op.errMsg = r.reason
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
        # 先计算起点约束
        start_v_key = vertex_key(self.start_state.x, self.start_state.y)
        start_v_cs = self.ctx.constraints.get(start_v_key)
        start_vertex_safe_intervals = constraints_to_safe_intervals(start_v_cs) or [Interval(0, MAX_TIME)]

        # 初始节点
        start_node = SippNode(
            id=self.node_id,
            state=replace(self.start_state, timeStart=self.time_offset, timeEnd=self.time_offset),
            vertexSafeInterval=start_vertex_safe_intervals[0],
            parent=None,
            g=0.0,
            f=self.admissible_heuristic(self.start_state),
        )

        heapq.heappush(self.open_set, start_node)

        while self.open_set:
            if (time.time() - self.op.startedOn) > 1:
                return TargetOnePlanResult(
                    self.ctx.robotName,
                    False,
                    "Low timeout",  # 这里使用 reason
                    planCost=time.time() - self.op.startedOn,
                    expandedCount=self.op.expandedNum,
                    fromState=self.start_state,
                    toState=self.goal_state,
                )

            node: SippNode = heapq.heappop(self.open_set)
            state = node.state

            # 记录展开过程
            self.op.expandedList.append(str(self.op.expandedNum) + "|" + node.desc())
            self.op.expandedNum += 1
            self.op.openSize.append(len(self.open_set))

            # 已展开的节点加入到 close
            self.closed_set[state.state_time_key()] = node

            if is_same_location_of_state(state, self.goal_state) and state.timeEnd > self.last_goal_constraint:
                return self.build_ok_result(node)

            self.build_successors(node)

        return TargetOnePlanResult(
            self.ctx.robotName,
            False,
            "All states checked",  # 这里使用 reason
            planCost=time.time() - self.op.startedOn,
            expandedCount=self.op.expandedNum,
            fromState=self.start_state,
            toState=self.goal_state,
        )

    def admissible_heuristic(self, state: State) -> float:
        """
        改进的启发式函数，考虑距离与旋转
        """
        # 计算曼哈顿距离
        manhattan_dist = abs(state.x - self.goal_state.x) + abs(state.y - self.goal_state.y)

        # 计算旋转成本
        d_head = abs(state.head - self.goal_state.head)
        if d_head > 180:
            d_head = 360 - d_head
        rotation_cost = (d_head / 90.0) * self.ctx.rotateUnitCost

        # 综合距离和旋转
        return (float(manhattan_dist) / self.ctx.moveUnitCost) + (float(rotation_cost) / self.ctx.rotateUnitCost)

    def build_ok_result(self, node: SippNode):
        # 达到时间在最后一次目标点被约束的时刻后
        # 到达后等待一段时间
        last_state = node.state
        path = [last_state]
        curr_node = node.parent
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
            cost=node.g + goal_stop_time_num,
            planCost=time.time() - self.op.startedOn,
            expandedCount=self.op.expandedNum,
            timeNum=action_state.timeEnd,
            timeStart=self.time_offset,
            timeEnd=action_state.timeEnd,
            fromState=self.start_state,
            toState=self.goal_state,
            path=path
        )

    def build_successors(self, from_node: SippNode):
        moves = [(1, 0, 0), (0, 1, 90), (0, -1, 270), (-1, 0, 180)]  # dx, dy, target direction

        for dx, dy, to_head in moves:
            to_x = from_node.state.x + dx
            to_y = from_node.state.y + dy
            to_cell_index = to_x + to_y * self.ctx.mapDimX

            # if self.detect_loop(from_node, State(x=to_x, y=to_y, head=0, timeStart=-1, timeEnd=-1, type=-1)):
            #     continue

            # 在地图范围内且不是障碍物
            if (to_x < 0 or to_x >= self.ctx.mapDimX or to_y < 0 or to_y >= self.ctx.mapDimY
                    or to_cell_index in self.ctx.obstacles):
                continue

            rotate_time_num = 0
            if to_head != from_node.state.head:
                # 需要转的角度，初始，-270 ~ +270
                d_head = abs(to_head - from_node.state.head)
                # 270 改成 90
                if d_head > 180:
                    d_head = 360 - d_head

                d_head /= 90

                # 旋转需要的时间步
                rotate_time_num = round(float(d_head) / self.ctx.rotateUnitCost)

            if rotate_time_num > 0:
                if from_node.state.timeEnd + rotate_time_num > from_node.vertexSafeInterval.timeEnd:
                    # 不能在当前单元格呆这么久
                    return

            move_time_num = round(float(abs(dx + dy)) / self.ctx.moveUnitCost)

            if move_time_num < 1:
                move_time_num = 1

            # 寻找与 [start_t, end_t] 都相交的顶点安全区间
            vertex_safe_interval: Optional[Interval] = None

            # 下一个状态安全区间
            v_key = vertex_key(to_x, to_y)
            v_cs = self.ctx.constraints.get(v_key)
            vertex_safe_intervals = constraints_to_safe_intervals(v_cs) or [Interval(0, MAX_TIME)]

            # 到下一个状态的边约束
            e_key = edge_key(from_node.state.x, from_node.state.y, to_x, to_y)
            e_cs = self.ctx.constraints.get(e_key)

            # 到下一个状态的时间
            t = MAX_TIME

            # 允许到达下一个顶点的最早时间和最晚时间
            next_v_t_start = from_node.state.timeEnd + rotate_time_num + move_time_num
            next_v_t_end = from_node.vertexSafeInterval.timeEnd + move_time_num
            if next_v_t_end > MAX_TIME:
                next_v_t_end = MAX_TIME

            for (vi, v_interval) in enumerate(vertex_safe_intervals):
                if not is_time_overlay(v_interval.timeStart, v_interval.timeEnd, next_v_t_start, next_v_t_end):
                    continue

                vertex_safe_interval = v_interval

                safe_interval_start = max(v_interval.timeStart, next_v_t_start)
                safe_interval_end = min(v_interval.timeEnd, next_v_t_end)

                # 检查边约束
                tt = self.check_edge_constraints(safe_interval_start, safe_interval_end, move_time_num, e_cs)
                if tt < MAX_TIME:
                    # 如果是终点，追加等待时间
                    if is_same_location(to_x, to_y, self.goal_state.x, self.goal_state.y):
                        if tt + self.ctx.goalStopTimeNum > v_interval.timeEnd:
                            self.op.logs.append(
                                f"Added goal stop time {self.ctx.goalStopTimeNum} exceeds safe interval {v_interval}")
                            continue

                    t = tt
                    break

            if t >= MAX_TIME or not vertex_safe_interval:
                return

            parent_node = from_node

            # 先旋转再等待
            if rotate_time_num > 0:
                time_start = parent_node.state.timeEnd + 1  # 上一个状态结束 +1
                time_end = parent_node.state.timeEnd + rotate_time_num  # 上一个状态结束 + 旋转时间
                rotate_state = replace(parent_node.state, head=to_head, timeStart=time_start, timeEnd=time_end, type=4)
                g = parent_node.g + rotate_time_num
                self.node_id += 1
                rotate_node = SippNode(id=self.node_id,
                                       state=rotate_state,
                                       vertexSafeInterval=parent_node.vertexSafeInterval,
                                       parent=parent_node,
                                       g=g,
                                       f=g + self.admissible_heuristic(rotate_state),
                                       )
                self.closed_set[rotate_state.state_time_key()] = rotate_node
                parent_node = rotate_node

            # 是否等待
            wait_time_num = t - next_v_t_start  # 等待时间
            if wait_time_num > 0:
                time_start = parent_node.state.timeEnd + 1  # 旋转完了再 +1
                time_end = time_start + wait_time_num - 1
                wait_state = replace(parent_node.state, timeStart=time_start, timeEnd=time_end, type=5)
                g = parent_node.g + t - next_v_t_start
                self.node_id += 1
                wait_node = SippNode(id=self.node_id,
                                     state=wait_state,
                                     vertexSafeInterval=parent_node.vertexSafeInterval,
                                     parent=parent_node,
                                     g=g,
                                     f=g + self.admissible_heuristic(wait_state),
                                     )
                self.closed_set[wait_state.state_time_key()] = wait_node
                parent_node = wait_node

            next_state = State(x=to_x, y=to_y, head=to_head, timeStart=t, timeEnd=t + move_time_num - 1, type=3)
            g = parent_node.g + move_time_num
            self.node_id += 1
            next_node = SippNode(id=self.node_id,
                                 state=next_state,
                                 vertexSafeInterval=vertex_safe_interval,
                                 parent=parent_node,
                                 g=g,
                                 f=g + self.admissible_heuristic(next_state),
                                 )

            if self.closed_set.get(next_state.state_time_key()):
                return

            heapq.heappush(self.open_set, next_node)

    @staticmethod
    def check_edge_constraints(safe_interval_start: int, safe_interval_end: int, move_time_num: int,
                               e_cs: list[Constraint]):
        if not e_cs:
            return safe_interval_start

        for tt in range(safe_interval_start, safe_interval_end + 1):
            for c in e_cs:
                if is_time_overlay(tt - 1, tt - 1 + move_time_num, c.timeStart, c.timeEnd):
                    return MAX_TIME
            return tt
        return MAX_TIME

    @staticmethod
    def detect_loop(from_node: SippNode, neighbor: State):
        """
        检查路径中是否出现过此位置
        """
        n = from_node
        while n:
            # 如果当前位置与目标位置相同，则返回True
            if is_same_location_of_state(n.state, neighbor):
                return True
            # 否则，继续向上查找父节点
            n = n.parent
        # 如果没有找到相同的位置，则返回False
        return False
