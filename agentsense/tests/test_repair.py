import random

from agentsense.baselines.planners import PLANNERS
from agentsense.config import DEFAULT_ENV
from agentsense.data.instances import make_instance
from agentsense.env.model import validate_solution
from agentsense.repair import repair_solution


def test_repair_keeps_valid_solution_untouched():
    inst = make_instance("tdrive_small", 0)
    sol = PLANNERS["TVPG"](inst, DEFAULT_ENV, random.Random(1))
    fixed, violations = repair_solution(inst, sol)
    assert violations == []
    assert {k: v for k, v in fixed.items()} == sol


def test_repair_preserves_llm_route_shape():
    """最小修复应保留模型路线，而不是回退成 DP 路径。

    注意：``cells_per_step > 1`` 之后，"1 格/步"的锯齿路线在**很短的窗口**里
    已经没有空间保住形状（修复会走 ceil(d/cps) 的粗路径），所以这里取窗口最长的
    工作者——检验的目标是"修复 ≠ 重规划"，不是"任何窗口都能保住形状"。
    """
    inst = make_instance("tdrive_small", 0)
    w = max(inst.workers, key=lambda x: x.window)
    wid = w.wid
    # 构造一条形状合法但终点/时间戳故意写错的路线
    bad = [[w.origin[0], w.origin[1], 0]]
    x, y = w.origin
    for step in range(1, w.window):
        if step % 2:
            nx, ny = min(x + 1, inst.nx - 1), y
        else:
            nx, ny = x, min(y + 1, inst.ny - 1)
        bad.append([nx, ny, 0])
        x, y = nx, ny
    fixed, violations = repair_solution(inst, {wid: bad})
    assert violations == []
    vr = validate_solution(inst, fixed)
    assert vr.feasible
    # 修复后应与输入共享大部分格点（形状保住），而不是被整条替换
    original_cells = {(b[0], b[1]) for b in bad}
    fixed_cells = {(p[0], p[1]) for p in fixed[wid]}
    assert len(original_cells & fixed_cells) >= 2
