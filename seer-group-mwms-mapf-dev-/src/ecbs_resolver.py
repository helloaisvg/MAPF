import heapq
import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json
from pydash import find_index

from src.common import agents_solution_to_paths
from src.conflicts import count_conflicts_find_first_constraints, AgentConstraints, AllConstraints, clean_constraints
from src.domain import AgentTask, MapfResult, TargetManyPlanResult, State, TargetOnePlanResult
from src.low_common import LowContext
from src.low_sipp import SippResolver
from src.low_state_time_a_start import LowResolver


@dataclass_json
@dataclass(frozen=True)
class HighNode:
    id: int  # 节点 ID
    parentId: int
    solution: dict[str, TargetManyPlanResult]  # agent name ->
    constraints: dict[str, AgentConstraints]  # agent name ->
    cost: float  # 最终的所有机器人的 f 值相加
    conflictsCount: int  # 这个高层节点中的冲突总数
    firstConstraints: Optional[AllConstraints] = None  # 这个高层节点中的第一个冲突


class OpenHighNode:

    def __init__(self, n: HighNode):
        self.n = n

    def __lt__(self, other: 'OpenHighNode'):
        # if (cost != n.cost)
        return self.n.cost < other.n.cost


class FocalHighNode:

    def __init__(self, n: HighNode):
        self.n = n

    def __lt__(self, other: 'FocalHighNode'):
        if self.n.conflictsCount != other.n.conflictsCount:
            return self.n.conflictsCount < other.n.conflictsCount
        return self.n.cost < other.n.cost


class ECBSResolver:

    def __init__(self, w: float, map_dim_x: int, map_dim_y: int, obstacles: set[int], tasks: dict[str, AgentTask],
                 low_resolver: int, update_cost: Callable[[float], None]):
        """
        :param w:
        :param map_dim_x:
        :param map_dim_y: y 向下为正
        :param obstacles:
        :param tasks: by robot
        """
        self.cancelled = False

        self.w = w
        self.map_dim_x = map_dim_x
        self.map_dim_y = map_dim_y
        self.obstacles = obstacles
        self.tasks = tasks
        self.low_resolver = low_resolver
        self.update_cost = update_cost

        # TODO 它们会被并发访问
        self.high_id = 0
        self.high_node_expanded = 0
        self.low_node_expanded = 0
        self.start_on = time.time()

    def search(self) -> MapfResult:
        logging.info(f"High resolver start: {self.tasks}")

        op_dir = "op"
        abs_op_dir = os.path.abspath(op_dir)
        logging.info(f"High resolver on dir: {abs_op_dir}")
        # 删除 on 目录
        shutil.rmtree(op_dir, ignore_errors=True)
        # 重新创建目录
        os.makedirs(op_dir)

        root_node = self.build_root_hl_node()
        if root_node is None:
            return MapfResult(
                ok=False,
                plans={},
                timeCost=time.time() - self.start_on,
            )

        open_set: list[OpenHighNode] = []
        focal_set: list[FocalHighNode] = []

        heapq.heappush(open_set, OpenHighNode(root_node))
        heapq.heappush(focal_set, FocalHighNode(root_node))

        while open_set:
            time_pass = time.time() - self.start_on
            self.update_cost(time_pass)
            if self.cancelled:
                return MapfResult(
                    ok=False,
                    msg="Cancelled",
                    plans={},
                    timeCost=time.time() - self.start_on,
                )

            # 检查超时
            if time_pass > 5:
                return MapfResult(
                    ok=False,
                    msg="Timeout",
                    plans={},
                    timeCost=time.time() - self.start_on,
                )

            # 重建 focal 集合
            best_cost = open_set[0].n.cost
            focal_set = []
            bound = best_cost * self.w
            for on in open_set:
                if on.n.cost <= bound:  # 机器人执行成本:
                    heapq.heappush(focal_set, FocalHighNode(on.n))

            high_node = heapq.heappop(focal_set).n
            self.remove_open_node(open_set, high_node)
            self.high_node_expanded += 1
            logging.info(f"High node expanded: {self.high_node_expanded}: {high_node}")

            if not high_node.firstConstraints:
                return MapfResult(
                    ok=True,
                    plans=high_node.solution,
                    timeCost=time.time() - self.start_on)

            for agent_name, c in high_node.firstConstraints.constraints.items():
                if self.cancelled:
                    return MapfResult(
                        ok=False,
                        msg="Cancelled",
                        plans={},
                        timeCost=time.time() - self.start_on,
                    )
                child_node = self.build_child_hl_node(high_node, agent_name, c)
                if not child_node:
                    continue
                heapq.heappush(open_set, OpenHighNode(child_node))

        return MapfResult(
            ok=False,
            msg="All high states expanded",
            plans={},
            timeCost=time.time() - self.start_on,
        )

    def build_root_hl_node(self) -> Optional[HighNode]:
        solution: dict[str, TargetManyPlanResult] = {}
        cost = 0.0

        for (robot_name, task) in self.tasks.items():
            fa_ctx = LowContext(
                robotName=robot_name,
                highId=self.high_id,
                w=self.w,
                mapDimX=self.map_dim_x,
                mapDimY=self.map_dim_y,
                obstacles=self.obstacles,
                moveUnitCost=1.0,
                rotateUnitCost=1.0,
                goalStopTimeNum=task.stopTimes,
                constraints={},
                oldAllPaths={}
            )
            rs = search_low_many(fa_ctx, self.low_resolver, task)
            self.low_node_expanded += rs.expandedCount

            if not rs.ok:
                print(f"Agent {rs.agentName} no init solution")
                return None

            solution[robot_name] = rs
            cost += rs.cost

        conflicts_count, first_constraints = count_conflicts_find_first_constraints(agents_solution_to_paths(solution))

        # 确保每个机器人都有一个空约束
        constraints_map: dict[str, AgentConstraints] = {}
        for robot_name in self.tasks.keys():
            constraints_map[robot_name] = {}

        return HighNode(
            id=self.high_id,
            parentId=-1,
            solution=solution,
            constraints=constraints_map,
            cost=cost,
            conflictsCount=conflicts_count,
            firstConstraints=first_constraints,
        )

    def build_child_hl_node(self, parent: HighNode, agent_name: str, constraint: AgentConstraints) \
            -> Optional[HighNode]:
        child_constraints = self.build_child_constraints(parent, agent_name, constraint)
        agent_constraints = child_constraints[agent_name]
        solution = parent.solution.copy()

        all_paths = agents_solution_to_paths(solution)

        cost = parent.cost - solution[agent_name].cost

        self.high_id += 1
        child_node_id = self.high_id

        task = self.tasks[agent_name]
        fa_ctx = LowContext(
            robotName=agent_name,
            highId=child_node_id,
            w=self.w,
            mapDimX=self.map_dim_x,
            mapDimY=self.map_dim_y,
            obstacles=self.obstacles,
            moveUnitCost=1.0,
            rotateUnitCost=1.0,
            goalStopTimeNum=task.stopTimes,
            constraints=agent_constraints,
            oldAllPaths=all_paths,
        )

        rs = search_low_many(fa_ctx, self.low_resolver, task)
        self.low_node_expanded += rs.expandedCount
        if not rs.ok:
            logging.info(f"Failed to build child high node, no solution for agent {agent_name}")
            return None
        solution[agent_name] = rs

        cost += rs.cost

        conflicts_count, first_constraints = count_conflicts_find_first_constraints(
            agents_solution_to_paths(solution))

        # 检查是否遵守了约束
        new_agent_constraints: Optional[AgentConstraints] = (first_constraints and
                                                             first_constraints.constraints.get(agent_name))
        if new_agent_constraints:
            for key, cs in new_agent_constraints.items():
                if key in agent_constraints:
                    cs2 = agent_constraints[key]
                    for c in cs:
                        if c in cs2:
                            raise Exception(f"Constraints already exist for agent {agent_name}, high={child_node_id}: "
                                            f"{constraint}")

        return HighNode(
            id=child_node_id,
            parentId=parent.id,
            constraints=child_constraints,
            solution=solution,
            cost=cost,
            conflictsCount=conflicts_count,
            firstConstraints=first_constraints
        )

    @staticmethod
    def build_child_constraints(node: HighNode, agent_name: str, acs: AgentConstraints) -> dict[str, AgentConstraints]:
        """
        产生子节点用的约束
        """
        constraints = node.constraints.copy()  # 继承父节点的约束
        old_agent_constraints: AgentConstraints = constraints[agent_name]

        for key, cs in acs.items():
            if not key in old_agent_constraints:
                old_agent_constraints[key] = []
            old_agent_constraints[key].extend(cs)
            old_agent_constraints[key] = clean_constraints(old_agent_constraints[key])

        return constraints

    @staticmethod
    def remove_open_node(open_set: list[OpenHighNode], node: HighNode):
        index = find_index(open_set, lambda n: n.n.id == node.id)
        if index < 0:
            return
        else:
            # 把最后一个填充过来
            open_set[index] = open_set[-1]
            open_set.pop()
            # 必须重建！o(n)
            heapq.heapify(open_set)


def search_low_many(ctx: LowContext, low_resolver_index: int, task: AgentTask) -> TargetManyPlanResult:
    """
    多目标搜索
    """
    start_state = State(x=task.fromState.x, y=task.fromState.y, head=0, timeStart=0, timeEnd=0, type=1)
    goal_states = [State(x=s.x, y=s.y, head=0, timeStart=-1, timeEnd=-1, type=2) for s in task.toStates]

    start_on = time.time()

    from_state = start_state
    ok = True
    reason = ""

    steps: list[TargetOnePlanResult] = []
    path: list[State] = []

    expanded_count = 0
    cost = 0.0
    time_num = 0

    for ti, goal_state in enumerate(goal_states):
        sr = search_low_one(ctx, low_resolver_index, time_num, from_state, goal_state)
        expanded_count += sr.expandedCount

        if not sr.ok:
            ok = False
            reason = sr.reason
            break

        steps.append(sr)
        path.extend(sr.path)

        cost += sr.cost
        time_num += sr.timeNum

        from_state = goal_state

    return TargetManyPlanResult(
        ctx.robotName,
        ok,
        reason,
        cost=cost,
        expandedCount=expanded_count,
        planCost=time.time() - start_on,
        timeNum=time_num, timeStart=0, timeEnd=time_num,
        steps=steps, path=path,
    )


def search_low_one(ctx: LowContext, low_resolver_index: int, time_num: int, from_state: State, goal_state: State) \
        -> TargetOnePlanResult:
    low_resolver = None
    if low_resolver_index == 1:
        low_resolver = SippResolver(ctx, time_num, from_state, goal_state)
    else:
        low_resolver = LowResolver(ctx, time_num, from_state, goal_state)
    sr = low_resolver.resolve_one()
    return sr
