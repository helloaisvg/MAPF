from dataclasses import dataclass, field

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class Location:
    """
    frozen 才是 hashable 的，可以放入 set。
    """
    x: int
    y: int


@dataclass_json
@dataclass(frozen=True)
class Cell(Location):
    pass


@dataclass_json
@dataclass(frozen=True)
class Duration:
    timeStart: int
    timeEnd: int


@dataclass_json
@dataclass(frozen=True)
class State:
    """
    允许跨多个时间步
    """
    x: int
    y: int
    head: int
    timeStart: int  # 开始进入这个状态的时间步 = 上一个状态的 timeEnd+1
    timeEnd: int  # 到达这个位置的时间步
    type: int  # 1: start, 2: target, 3: move, 4: rotate, 5: wait

    def __str__(self):
        return f"{self.x},{self.y}@{self.head}|{self.timeStart}:{self.timeEnd}|{self.type}"

    def state_time_key(self):
        return f"{self.x},{self.y}@{self.head}°|{self.timeStart}:{self.timeEnd}"


@dataclass_json
@dataclass
class TargetOnePlanResult:
    """
    单目标
    """
    agentName: str
    ok: bool = True
    reason: str = None
    cost: float = 0.0  # 机器人执行成本
    expandedCount: int = 0
    planCost: float = 0  # 秒
    timeNum: int = 0
    timeStart: int = -1
    timeEnd: int = -1
    fromState: State = None
    toState: State = None
    path: list[State] = None
    extra: any = None


@dataclass_json
@dataclass
class TargetManyPlanResult:
    """
    多目标
    """
    agentName: str
    ok: bool = True
    reason: str = None
    cost: float = 0.0  # 机器人执行成本
    expandedCount: int = 0
    planCost: float = 0  # 秒
    timeNum: int = 0
    timeStart: int = -1
    timeEnd: int = -1
    steps: list[TargetOnePlanResult] = None
    path: list[State] = None  # 总路径
    extra: any = None


@dataclass_json
@dataclass
class AgentTask:
    name: str
    fromState: Cell
    toStates: list[Cell]
    stopTimes: int = 1  # 停多久


@dataclass_json
@dataclass
class MapfReq:
    w: float
    mapDimX: int
    mapDimY: int  # y 向下为正
    obstacles: set[int]
    tasks: dict[str, AgentTask]  # robot name ->
    goalStops: int = 0


@dataclass_json
@dataclass
class MapfResult:
    ok: bool = True
    msg: str = ""
    plans: dict[str, TargetManyPlanResult] = None
    timeCost: float = 0.0


@dataclass_json
@dataclass
class MapConfig:
    robotNum: int = 10
    mapDimX: int = 30
    mapDimY: int = 20
    obstacleRatio: float = .3
    w: float = 1.5
    targetNum: int = 1
    goalStopTimes: int = 5
    cellSize: int = 28
    obstacles: set[int] = field(default_factory=set)


@dataclass_json
@dataclass
class RobotTaskReq:
    fromIndex: int
    toIndex: int


@dataclass_json
@dataclass
class MapReq:
    config: MapConfig
    tasks: dict[str, RobotTaskReq]
