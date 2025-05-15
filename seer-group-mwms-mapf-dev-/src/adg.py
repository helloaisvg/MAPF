from src.domain import MapfResult


def build_adg(mapf_result: MapfResult):
    """
    机器人 1 要从 A 到 C，然后从 C 到 D；到达 C 的时刻是 t。
    机器人 2 要从 B 到 C，到达 C 的时刻是 t + M，即比机器人 1 晚。
    那么一定要机器人 1 先到 D（即确保已离开 C），机器人 2 才能开始从 B 到 C 的移动，比较保险。
    否则机器人 1 一边离开 C，机器人 2 同时去 C，如果速度不同步，2 可能撞 1。
    """

    # 每个节点的前序节点
    nodes: dict[str, list[str]] = {}

    if mapf_result.ok:
        for rr in mapf_result.plans.values():
            path = rr.path
            for i in range(len(path) - 2):
                # 前一段路径
                n1key = to_adg_key(rr.agentName, i)
                # 后一段路径
                n2key = to_adg_key(rr.agentName, i + 1)
                nodes.setdefault(n2key, []).append(n1key)

        # 跨机器人路径点存在共享位置且时间不冲突时，添加依赖边
        for rr1 in mapf_result.plans.values():
            path1 = rr1.path
            for i in range(len(path1) - 1):
                for rr2 in mapf_result.plans.values():
                    if rr1.agentName == rr2.agentName:
                        continue
                    path2 = rr2.path
                    for j in range(len(path2) - 1):
                        s = path1[i]
                        g = path2[j]
                        if s.x == g.x and s.y == g.y and s.timeEnd <= g.timeEnd:
                            n1key = to_adg_key(rr1.agentName, i + 1)
                            n2key = to_adg_key(rr2.agentName, j)
                            nodes.setdefault(n2key, []).append(n1key)

    return nodes


def to_adg_key(robot_name: str, path_index: int) -> str:
    return f"{robot_name}#{path_index}"
