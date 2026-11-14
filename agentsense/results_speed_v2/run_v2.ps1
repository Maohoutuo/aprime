$ErrorActionPreference = "Continue"
Set-Location "d:\composition\agentsense"
function Step($name, $argline) {
    "[$(Get-Date -Format HH:mm:ss)] === $name ===" | Tee-Object -FilePath _rerun_speed_v2\run.log -Append
    cmd /c "py -3.14 $argline 2>&1" | Tee-Object -FilePath _rerun_speed_v2\run.log -Append
    "[$(Get-Date -Format HH:mm:ss)] --- $name exit $LASTEXITCODE ---" | Tee-Object -FilePath _rerun_speed_v2\run.log -Append
}
Step "calibrate e_scale (speed v2)" "experiments\calibrate.py"
Step "table4 tdrive n20" "experiments\run_table4.py --dataset tdrive --n 20 --out _rerun_speed_v2\table4_tdrive_n20.md --no-report"
Step "table4 grab n20"   "experiments\run_table4.py --dataset grab --n 20 --out _rerun_speed_v2\table4_grab_n20.md --no-report"
"[$(Get-Date -Format yyyy-MM-dd HH:mm:ss)] agentsense v2 DONE" | Tee-Object -FilePath _rerun_speed_v2\run.log -Append
