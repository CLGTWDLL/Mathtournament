"""把本地反馈策略接入 question2 选点流程。

所有场景都现场调用本目录 Q2Policy，不读取冻结中间点或外部目录。
"""


def select_second_point(first_position, bearing_deg, max_extra_measurements=5, weight=1.0):
    """返回可执行策略所需的第二点信息。"""
    from model import Problem

    from q2_policy import Q2Policy
    policy = Q2Policy(
        first_position=list(first_position),
        first_bearing_deg=float(bearing_deg),
        max_extra_measurements=max_extra_measurements,
        weight=weight,
    )
    action = policy.next_action()
    if action is None or action["action"] != "measure":
        raise RuntimeError("多步策略未能给出第二个检测点")
    world = (float(action["position"]["x"]), float(action["position"]["y"]))
    local = Problem(x=first_position[0], y=first_position[1], angle_deg=bearing_deg).local(world)
    return {
        "method": "Q2Policy半解析/后验规划选点",
        "local_point": list(local),
        "world_point": list(world),
        "policy_ready": True,
        "source": "代码/q2_policy.py",
    }


def make_policy(first_position, bearing_deg, **kwargs):
    """返回本地 Q2Policy 实例，供主流程继续执行多步反馈。"""
    from q2_policy import Q2Policy
    return Q2Policy(
        first_position=list(first_position),
        first_bearing_deg=float(bearing_deg),
        **kwargs,
    )
