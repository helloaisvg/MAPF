from dataclasses import dataclass, replace
from typing import Optional

from typing_extensions import TypeAlias

from src.common import is_time_overlay, vertex_key, is_same_location_of_state, edge_key
from src.domain import State


@dataclass
class Constraint:
    """
    一项约束
    """
    key: str  # 顶点/边
    type: int  # 1 vertex, 2 edge
    masterAgent: str  # 施加约束的智能体
    timeStart: int  # 约束开始时间
    timeEnd: int  # 约束结束时间

    def __str__(self):
        return f"{self.key}|{self.timeStart}:{self.timeEnd}|t={self.type}a=|{self.masterAgent}"


# 一个智能体的所有约束
AgentConstraints: TypeAlias = dict[str, list[Constraint]]  # key ->


@dataclass
class AllConstraints:
    """
    一个冲突产生的约束
    """
    timeStart: int  # 最早时间
    constraints: dict[str, AgentConstraints]  # agent name ->


def count_conflicts_find_first_constraints(paths: dict[str, list[State]]) -> (int, Optional[AllConstraints]):
    """
    计算所有智能体的总冲突数，并返回时间上最早的一个冲突产生的约束
    实际返回的约束，含有冲突的两个智能体的最早的约束
    """
    conflicts_count = 0
    first_conflict: Optional[AllConstraints] = None

    agent_names = list(paths.keys())

    for a1i in range(len(agent_names) - 1):
        a1 = agent_names[a1i]
        path1 = paths[a1]
        for a2i in range(a1i + 1, len(agent_names)):
            a2 = agent_names[a2i]
            path2 = paths[a2]
            # 顶点约束
            # 遍历两个智能体的每个状态，如果位置相同且时间交叠
            for s1 in path1:
                for s2 in path2:
                    if (is_same_location_of_state(s1, s2) and
                            is_time_overlay(s1.timeStart, s1.timeEnd, s2.timeStart, s2.timeEnd)):
                        conflicts_count += 1
                        if (first_conflict is None
                                or s1.timeStart < first_conflict.timeStart
                                or s2.timeStart < first_conflict.timeStart):
                            v_key = vertex_key(s1.x, s1.y)
                            # 注意用另一个机器人的时间
                            first_conflict = AllConstraints(
                                min(s1.timeStart, s2.timeStart),
                                {
                                    a1: {v_key: [Constraint(v_key, 1, a2, s2.timeStart, s2.timeEnd)]},
                                    a2: {v_key: [Constraint(v_key, 1, a1, s1.timeStart, s1.timeEnd)]}
                                }
                            )
            # 边约束
            # 遍历两个智能体的所有移动 i -> i+1
            for s1i in range(len(path1) - 1):
                a1a = path1[s1i]
                a1b = path1[s1i + 1]
                if is_same_location_of_state(a1a, a1b): continue  # 原地动作
                for s2i in range(len(path2) - 1):
                    a2a = path2[s2i]
                    a2b = path2[s2i + 1]
                    if is_same_location_of_state(a2a, a2b): continue  # 原地动作
                    if (is_same_location_of_state(a1a, a2b) and is_same_location_of_state(a1b, a2a) and
                            # 起始时间取上一状态的结束时间 TODO 是否合理
                            is_time_overlay(a1a.timeEnd, a1b.timeEnd, a2a.timeEnd, a2b.timeEnd)):
                        conflicts_count += 1
                        if (first_conflict is None
                                or a1a.timeStart < first_conflict.timeStart
                                or a2a.timeStart < first_conflict.timeStart):
                            e_key_1 = edge_key(a1a.x, a1a.y, a1b.x, a1b.y)
                            e_key_2 = edge_key(a2a.x, a2a.y, a2b.x, a2b.y)
                            # 注意用另一个机器人的时间，但边的起点和终点还是选自己的（起点终点不需要反转）
                            first_conflict = AllConstraints(
                                min(a1a.timeStart, a2a.timeStart),
                                {
                                    a1: {e_key_1: [Constraint(e_key_1, 2, a2, a2a.timeEnd, a2b.timeEnd)]},
                                    a2: {e_key_2: [Constraint(e_key_2, 2, a1, a1a.timeEnd, a1b.timeEnd)]}
                                }
                            )

    return conflicts_count, first_conflict


def count_transition_conflicts(agent1: str, s1a: State, s1b: State, all_paths: dict[str, list[State]]) -> int:
    """
    智能体从 s1a 转移到新状态 s1b 与其他智能体的冲突
    """
    conflicts_count = 0
    for agent2, path2 in all_paths.items():
        if agent1 != agent2 and path2 is not None:
            for s2i, s2a in enumerate(path2):
                # 顶点
                if (is_same_location_of_state(s1b, s2a) and
                        is_time_overlay(s1b.timeStart, s1b.timeEnd, s2a.timeStart, s2a.timeEnd)):
                    conflicts_count += 1
                if s2i < len(path2) - 1:
                    s2b = path2[s2i + 1]
                    if (is_same_location_of_state(s1a, s2b) and is_same_location_of_state(s1b, s2a) and
                            # 起始时间取上一状态的结束时间 TODO 是否合理
                            is_time_overlay(s1a.timeEnd, s1b.timeEnd, s2a.timeEnd, s2b.timeEnd)):
                        conflicts_count += 1

    return conflicts_count


def check_constraints_intersect(robot_name: str, cs1: AgentConstraints, cs2: AgentConstraints):
    """
    两个约束集有交集
    """
    for key, list1 in cs1.items():
        list2 = cs2.get(key)
        if not list2:
            continue
        for c1 in list1:
            if c1 in list2:  # Constraint 完整相等
                raise Exception(f"Constraints already exist for robot {robot_name}: {c1}")


def clean_constraints(cs: list[Constraint]) -> list[Constraint]:
    """去重、合并"""

    if len(cs) <= 1:
        return cs

    # 按 timeStart 排序
    cs.sort(key=lambda c: c.timeStart)

    new_cs: list[Constraint] = []
    last_c = None
    for c in cs:
        if last_c:
            if c.timeStart > last_c.timeEnd + 1: # 如果 start = end + 1 也是相交的，如 [2, 2], [3, 3]
                # 不相交
                new_cs.append(c)
            else:
                # 相交及包含
                c = replace(c, timeStart=last_c.timeStart, timeEnd=max(last_c.timeEnd, c.timeEnd))
                new_cs.pop()
                new_cs.append(c)
        else:
            new_cs.append(c)
        last_c = c

    return new_cs
