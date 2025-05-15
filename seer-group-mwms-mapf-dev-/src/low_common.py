from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json

from src.conflicts import AgentConstraints
from src.domain import State


@dataclass
class LowContext:
    robotName: str
    highId: int
    w: float
    mapDimX: int
    mapDimY: int
    obstacles: set[int]
    moveUnitCost: float
    rotateUnitCost: float
    goalStopTimeNum: int
    constraints: Optional[AgentConstraints]  # 约束可能为空，初始求解时
    oldAllPaths: dict[str, list[State]]  # 在进行此次底层求解所有机器人的已知路径


@dataclass_json
@dataclass
class LowOp:
    """
    记录一次底层求解过程
    """
    robotName: str
    highId: int
    w: float
    mapDimX: int
    mapDimY: int
    # obstacles: set[int]
    moveUnitCost: float
    rotateUnitCost: float
    goalStopTimeNum: int
    startCell: str
    goalCell: str
    startIndex: int
    goalIndex: int
    constraints: Optional[AgentConstraints]
    ok: bool
    errMsg: str
    path: Optional[list[str]]
    expandedNum: int
    expandedList: list[str]
    openSize: list[int]  # 每次展开后 open 集合的大小
    focalSize: list[int]  # 每次展开后 focal 集合的大小
    logs: list[str]
    warnings: list[str]
    startedOn: float = 0.0
    endedOn: float = 0.0
    timeCost: float = 0.0
