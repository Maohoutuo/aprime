$ErrorActionPreference = 'Continue'
Set-Location -Path 'd:\composition\mapus\_rerun_speed_v2'
$py = 'py'
$log = 'run.log'

function Step($name, $argline) {
    "[$(Get-Date -Format 'HH:mm:ss')] === $name ===" | Tee-Object -FilePath $log -Append
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $cmd = "$py -3.14 $argline"
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath $log -Append
    $sw.Stop()
    "[$(Get-Date -Format 'HH:mm:ss')] --- $name done in $([int]$sw.Elapsed.TotalSeconds)s (exit $LASTEXITCODE) ---" |
        Tee-Object -FilePath $log -Append
}

"" | Tee-Object -FilePath $log -Append
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] rerun speed_v2 start" | Tee-Object -FilePath $log -Append

Step 'main tdrive'    '..\experiments\run_main.py --dataset tdrive --n 20 --parallel 8'
Step 'main grab'      '..\experiments\run_main.py --dataset grab --n 20 --parallel 8'
Step 'ablation tdrive' '..\experiments\run_ablation.py --dataset tdrive --n 20 --parallel 8'
Step 'ablation grab'   '..\experiments\run_ablation.py --dataset grab --n 20 --parallel 8'
Step 'negotiation'     '..\experiments\run_negotiation.py --scale medium --n 20 --parallel 8'
Step 'fairness'        '..\experiments\run_fairness.py --n-tasks 60 --route-workers 6'
Step 'runtime'         '..\experiments\run_runtime.py --dataset tdrive --n 5'
Step 'figures'         '..\experiments\make_figures.py'
Step 'report'          '..\experiments\build_report.py'

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] rerun speed_v2 ALL DONE" | Tee-Object -FilePath $log -Append
