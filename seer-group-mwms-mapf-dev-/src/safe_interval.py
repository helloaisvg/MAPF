import logging
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json

from src.conflicts import Constraint, clean_constraints

# 安全区间最大时间
MAX_TIME = 1000 * 60 * 60


@dataclass_json
@dataclass(frozen=True)
class Interval:
    timeStart: int
    timeEnd: int


def constraints_to_safe_intervals(constraints: Optional[list[Constraint]]):
    intervals: list[Interval] = [Interval(0, MAX_TIME)]
    if not constraints:
        return intervals
    # 先清理
    constraints = clean_constraints(constraints)
    for c in constraints:
        insert_safe_interval(intervals, c)

    logging.debug(f"constraints: {constraints}, intervals: {intervals}")
    return intervals


def insert_safe_interval(intervals: list[Interval], c: Constraint):
    i = 0
    while i < len(intervals):
        interval = intervals[i]
        if c.timeStart >= interval.timeEnd:
            # 不相交
            i += 1
        elif c.timeEnd <= interval.timeStart:
            # 后续不再相交
            return
        elif interval.timeStart < c.timeStart and interval.timeEnd <= c.timeEnd:
            # 切掉后面不安全的部分
            intervals[i] = Interval(timeStart=interval.timeStart, timeEnd=c.timeStart - 1)
            i += 1
        elif c.timeStart <= interval.timeStart and c.timeEnd < interval.timeEnd:
            # 切掉前面不安全的部分
            intervals[i] = Interval(timeStart=c.timeEnd + 1, timeEnd=interval.timeEnd)
            return  # 后续不再相交
        elif interval.timeStart < c.timeStart and c.timeEnd < interval.timeEnd:
            # 去掉中间不安全的部分
            intervals[i] = Interval(timeStart=c.timeEnd + 1, timeEnd=interval.timeEnd)
            intervals.insert(i, Interval(timeStart=interval.timeStart, timeEnd=c.timeStart - 1))
            return  # 后续不再相交
        else:  # start <= interval.timeStart and interval.timeEnd <= end
            # 完全覆盖，删除此安全区间
            intervals.pop(i)
