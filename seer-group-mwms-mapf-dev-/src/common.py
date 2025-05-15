import math

from src.domain import State, TargetManyPlanResult


def vertex_key(x: int, y: int) -> str:
    return f"{x},{y}"


def edge_key(x1: int, y1: int, x2: int, y2: int) -> str:
    return f"{x1},{y1}->{x2},{y2}"


def x_y_to_index(x: int, y: int, map_dim_x: int) -> int:
    return y * map_dim_x + x


def distance_of_two_points(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def is_same_location(x1: int, y1: int, x2: int, y2: int) -> bool:
    return x1 == x2 and y1 == y2

def is_same_location_of_state(s1: State, s2: State) -> bool:
    return is_same_location(s1.x, s1.y, s2.x, s2.y)


def is_time_overlay(start1: int, end1: int, start2: int, end2: int):
    return start1 <= end2 and start2 <= end1


def agents_solution_to_paths(solution: dict[str, TargetManyPlanResult]) -> dict[str, list[State]]:
    all_paths: dict[str, list[State]] = {}
    for robot_name, s in solution.items():
        if s.ok:
            all_paths[robot_name] = s.path
    return all_paths
