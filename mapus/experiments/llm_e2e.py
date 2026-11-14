"""单个实例的真实 LLM 端到端验证：看解析成功率、耗时与输出质量。"""
import sys, time
sys.path.insert(0, r"D:\composition\mapus")
from agentsense.config import DEFAULT_ENV
from mapus.config import MAPUSConfig
from mapus.data.instances import make_mapus_instance
from mapus.llm_backend import make_llm
from mapus.pipeline import run_mapus

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-72B-Instruct"
inst = make_mapus_instance("tdrive_small", 0)
w = inst.workers[0]
prof = inst.profiles[w.wid]
print("worker", w, "\nprofile", prof.to_text())
llm = make_llm("siliconflow", model, use_cache=True, max_tokens=4096)
cfg = MAPUSConfig()
t0 = time.time()
run = run_mapus(inst, inst.urban, inst.profiles, cfg, DEFAULT_ENV, llm, seed=1000, parallel=4)
dt = time.time() - t0
s = run.stats
print(f"\n{model}: {dt:.1f}s")
for k in ("coverage", "pss", "n_accepted", "n_selected", "llm_calls_gen", "llm_invalid",
          "llm_repaired", "refines_accepted", "neg_attempts", "neg_accepted", "neg_rejected",
          "base_pss", "final_pss", "avg_overlap", "pre_neg_overlap"):
    print(f"  {k:18s} {s[k]}")
print("usage", llm.usage)
