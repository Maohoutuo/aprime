# -*- coding: utf-8 -*-
# 真实 LLM 复跑：FROZEN.md「还没跑的两项」里的第一项
#   ① 图 12 案例研究  experiments\run_case_study.py
#   ② 图 11 基座模型  experiments\run_base_llm.py
#
#   powershell -ExecutionPolicy Bypass -File experiments\run_llm_item1.ps1
#
# 前置：mapus\.env 的 SILICONFLOW_API_KEY 必须可用（欠费会返回 402 / code 30001）。
# 说明：硅基流动在并发突发时会偶发 402/限流，脚本按「断点续跑」处理——
#       results\llm_cache\ 里命中过的调用不再计费，重跑只补没跑完的部分。

$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)

function Invoke-Resume {
    param([string]$Label, [string]$Log, [scriptblock]$Body)
    for ($i = 1; $i -le 6; $i++) {
        Write-Host "[$Label] 第 $i 次尝试 ..." -ForegroundColor Cyan
        & $Body 2>&1 | Tee-Object -FilePath $Log -Append
        if ($LASTEXITCODE -eq 0) { Write-Host "[$Label] 完成" -ForegroundColor Green; return }
        Write-Warning "[$Label] 中断（exit=$LASTEXITCODE），10s 后从缓存处续跑"
        Start-Sleep -Seconds 10
    }
    throw "[$Label] 连续 6 次失败，检查 key/余额/限流"
}

# 1) 先验：只做一次最小调用，key 无效或欠费会立刻失败。
Write-Host '[preflight] siliconflow 单次调用 ...' -ForegroundColor Cyan
py -3.14 ..\agentsense\experiments\siliconflow_smoke.py
if ($LASTEXITCODE -ne 0) { throw 'preflight 失败：检查 mapus\.env 的 SILICONFLOW_API_KEY 与账户余额' }

# 2) 图 12 案例研究（2 个 small 实例，最便宜，先跑）
Invoke-Resume -Label '1/2 图12' -Log 'results\case_study_siliconflow.log' -Body {
    py -3.14 experiments\run_case_study.py --backend siliconflow `
        --model Qwen/Qwen2.5-72B-Instruct --llm-workers 2
}

# 3) 图 11 基座模型对比（默认模型表里的 Qwen3-32B 是思考模型，长提示下极慢，这里去掉）
Invoke-Resume -Label '2/2 图11' -Log 'results\base_llm_n2.log' -Body {
    py -3.14 experiments\run_base_llm.py --n 2 --datasets tdrive,grab `
        --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-72B-Instruct,deepseek-ai/DeepSeek-V3.2 `
        --parallel 1 --llm-workers 3
}

Write-Host '完成：results\case_study_siliconflow.json / results\base_llm_n2.json' -ForegroundColor Green
