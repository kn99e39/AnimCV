# Installs (or removes) the AnimCV autonomous Batch dispatch worker on Main.
#
# What it sets up, all per-user, nothing system-wide:
#   %USERPROFILE%\animcv-automation\repo   dedicated clone of arch/single_frame_first (hardlinked from the source
#                                          checkout, origin = GitHub)
#   every poll first pulls (fast-forward only) the dedicated clone and the source checkout this script lives in;
#   a checkout that is on another branch, has tracked modifications or has diverged is skipped, never repaired
#   %USERPROFILE%\animcv-automation\bin    frozen copy of scripts/batch_automation.py (reinstall to update)
#   %USERPROFILE%\animcv-automation\state  config.json, lock, journal, status, logs, blockers, PAUSE flag
#   repo\.git\hooks\pre-commit, pre-push   refuse next_batch.md edits, other-branch/deletion/non-fast-forward pushes
#   Task Scheduler task "AnimCV Batch Automation": pythonw poll every 15 min (+ at logon), current user,
#                                          interactive logon (runs while the user is logged on), no overlap
#
# Usage (PowerShell, any directory):
#   windows\install_batch_automation.ps1              install / update
#   windows\install_batch_automation.ps1 -Uninstall   remove the scheduled task (clone and state are kept)
# Pause:  New-Item "$env:USERPROFILE\animcv-automation\state\PAUSE"   (or Disable-ScheduledTask "AnimCV Batch Automation")
# Resume: Remove-Item that file                                          (or Enable-ScheduledTask ...)
# Status: python "$env:USERPROFILE\animcv-automation\bin\batch_automation.py" status --config "...\state\config.json"
param(
    [switch]$Uninstall,
    [string]$Root = "$env:USERPROFILE\animcv-automation",
    [string]$Python = "C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe",
    [int]$IntervalMinutes = 15
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Source = (Get-Location).Path
$TaskName = "AnimCV Batch Automation"
$Branch = "arch/single_frame_first"
$Url = "https://github.com/kn99e39/AnimCV.git"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Unregistered '$TaskName'. Clone and state kept at $Root."
    return
}

$Repo = Join-Path $Root "repo"; $State = Join-Path $Root "state"; $Bin = Join-Path $Root "bin"
New-Item -ItemType Directory -Force $State, $Bin | Out-Null
$Utf8 = New-Object System.Text.UTF8Encoding $false

function Invoke-Git { & git.exe @args; if ($LASTEXITCODE) { throw "git $args failed" } }

if (-not (Test-Path (Join-Path $Repo ".git"))) {
    Invoke-Git clone --quiet --branch $Branch $Source $Repo
    Invoke-Git -C $Repo remote set-url origin $Url
}
Invoke-Git -C $Repo fetch --quiet origin "+refs/heads/${Branch}:refs/remotes/origin/$Branch"
if ((git -C $Repo rev-parse --abbrev-ref HEAD) -ne $Branch) { throw "dedicated clone is not on $Branch; not changing it" }
if (git -C $Repo status --porcelain) { throw "dedicated clone is dirty; not touching it" }
Invoke-Git -C $Repo merge --ff-only --quiet "origin/$Branch"

Copy-Item (Join-Path $Source "scripts\batch_automation.py") $Bin -Force
$Worker = (Join-Path $Bin "batch_automation.py")
$Config = (Join-Path $State "config.json")
$cfg = [ordered]@{ branch = $Branch; remote = "origin"; repo = $Repo; state = $State; host = $env:COMPUTERNAME
                   python = $Python; claude = "auto"; timeout_hours = 48; gpu_busy_check = $true; notify = $true
                   pull_paths = @($Source) }
[IO.File]::WriteAllText($Config, ($cfg | ConvertTo-Json), $Utf8)

$py = $Python -replace "\\", "/"; $wk = $Worker -replace "\\", "/"; $cf = $Config -replace "\\", "/"
$hooks = Join-Path $Repo ".git\hooks"
[IO.File]::WriteAllText((Join-Path $hooks "pre-commit"), "#!/bin/sh`nexec `"$py`" `"$wk`" hook-pre-commit --config `"$cf`"`n", $Utf8)
[IO.File]::WriteAllText((Join-Path $hooks "pre-push"), "#!/bin/sh`nexec `"$py`" `"$wk`" hook-pre-push --config `"$cf`" `"`$@`"`n", $Utf8)

$Pythonw = Join-Path (Split-Path $Python) "pythonw.exe"
# USERDOMAIN can be "WORKGROUP", which does not map to an account; use the real Windows identity.
$User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $Pythonw -Argument "`"$Worker`" poll --config `"$Config`"" -WorkingDirectory $State
$every = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
$logon = New-ScheduledTaskTrigger -AtLogOn -User $User
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$Arguments = "`"$Worker`" poll --config `"$Config`""
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$same = $existing -and $existing.Actions.Execute -eq $Pythonw -and $existing.Actions.Arguments -eq $Arguments -and
        ($existing.Triggers | Where-Object { $_.Repetition.Interval -eq "PT$($IntervalMinutes)M" }) -and
        $existing.Settings.MultipleInstances -eq "IgnoreNew" -and $existing.Principal.LogonType -eq "Interactive"
if ($same) {
    # Worker code and config are updated in place above; an identical registration is left untouched (overwriting an
    # existing task can be denied to a non-elevated user).
    Write-Output "Task '$TaskName' already registered with the identical action and schedule; not re-registered."
} else {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $every, $logon -Settings $settings -Principal $principal `
        -Description "AnimCV: poll docs/next_batch.md on $Branch and execute each new ChatGPT Batch exactly once (Main only)." `
        -Force -ErrorAction Stop | Out-Null
}
# Never report success without reading the registration back.
$check = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
if ($check.Actions.Execute -ne $Pythonw) { throw "registered task does not match the requested action" }
Write-Output "Installed '$TaskName' for $User (every $IntervalMinutes min + at logon) -> $Worker"
