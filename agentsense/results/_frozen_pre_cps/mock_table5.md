# 表 5 复现（Mock LLM，small，n=2，planner=TVPG）

| 扰动 | SR | AIR | ANI | ACS | 用时(s) |
|---|---|---|---|---|---|
| bad_weather | 100.0% | +1.195% | 1.0 | +0.00 | 1.0 |
| area_blocked | 100.0% | +0.204% | 1.0 | -1.00 | 0.7 |
| worker_unavailable | 100.0% | -1.727% | 1.0 | +3.00 | 0.6 |
| traffic_congestion | 100.0% | -10.044% | 1.0 | +18.00 | 1.2 |
| time_window_shift | 100.0% | +0.361% | 1.0 | -1.00 | 2.6 |
| budget_cut | 100.0% | -6.888% | 1.0 | +14.50 | 2.6 |
| task_priority_change | 100.0% | +0.569% | 1.0 | -1.00 | 2.7 |
