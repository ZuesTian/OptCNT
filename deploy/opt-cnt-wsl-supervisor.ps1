# OptCNT WSL2 Worker Supervisor
# 守护三件事:
#   1) Windows SSH 隧道: 127.0.0.1:8870 -> 远端 127.0.0.1:8769 (worker 轮询走 loopback,不依赖 WSL2 出网)
#   2) WSL2 worker 进程: wsl.exe 启动 run_candidate_worker_foreground.sh,退出后自动拉起
#   3) WSL2 出网探测: 断了先软恢复(nmcli + systemd-resolved),连续失败后 wsl --shutdown 兜底(自动拉起)
# 用法: 由计划任务 'OptCNT WSL2 Worker Supervisor' 登录自启; 放 stop 文件到 StateDir 可停止。
$ErrorActionPreference = 'Stop'

# ---------- 配置 ----------
$SshKey       = 'C:\Users\AMD\.codex\mcp\.secrets\ali_mcp.pem'
$KnownHosts   = 'C:\Users\AMD\.codex\mcp\.secrets\known_hosts_ali_mcp'
$RemoteHost   = '47.236.76.214'
$TunnelPort   = 8870
$RemotePort   = 8769
$HealthUrl    = "http://127.0.0.1:$TunnelPort/api/v1/health"
$WorkerScript = '/mnt/e/Trae_Project/Analyzer/OPT/tmp/run_candidate_worker_foreground.sh'
$WslDistro    = 'Ubuntu-24.04'
$ProbeTarget  = '223.5.5.5'
$StateDir     = Join-Path $env:LOCALAPPDATA 'OptCNT\worker-supervisor'
$LogPath      = Join-Path $StateDir 'supervisor.log'
$StopFile     = Join-Path $StateDir 'stop'
$SoftRepairThreshold = 2     # 连续软恢复失败次数达到后允许兜底
$ShutdownCooldownMin  = 15   # wsl --shutdown 兜底最小间隔(分钟)

New-Item -ItemType Directory -Path $StateDir -Force | Out-Null
Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue

function Write-SupervisorLog([string]$Message) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding utf8
}

function Stop-Child([System.Diagnostics.Process]$Process) {
    if ($null -eq $Process -or $Process.HasExited) { return }
    try { $Process.Kill($true) } catch { Write-SupervisorLog "stop failed: $($_.Exception.Message)" }
}

function Start-Tunnel {
    $arguments = @(
        '-N', '-T',
        '-o', 'BatchMode=yes',
        '-o', 'ExitOnForwardFailure=yes',
        '-o', 'ServerAliveInterval=20',
        '-o', 'ServerAliveCountMax=3',
        '-o', 'ConnectTimeout=10',
        '-o', 'StrictHostKeyChecking=yes',
        '-o', "UserKnownHostsFile=$KnownHosts",
        '-i', $SshKey,
        '-L', "127.0.0.1:$($TunnelPort):127.0.0.1:$($RemotePort)",
        "root@$RemoteHost"
    )
    Start-Process -FilePath 'ssh.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru
}

function Start-Worker {
    Start-Process -FilePath 'wsl.exe' -ArgumentList @(
        '-d', $WslDistro, '--', 'sudo', 'bash', $WorkerScript
    ) -WindowStyle Hidden -PassThru
}

function Test-WslOutbound {
    & wsl.exe -d $WslDistro -- ping -c 1 -W 2 $ProbeTarget 2>&1 | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Repair-WslNet {
    Write-SupervisorLog 'soft repair: resolvectl flush + systemd-resolved restart + nmcli off/on'
    & wsl.exe -d $WslDistro -- sudo resolvectl flush-caches 2>&1 | Out-Null
    & wsl.exe -d $WslDistro -- sudo systemctl restart systemd-resolved 2>&1 | Out-Null
    & wsl.exe -d $WslDistro -- sudo nmcli networking off 2>&1 | Out-Null
    Start-Sleep -Seconds 2
    & wsl.exe -d $WslDistro -- sudo nmcli networking on 2>&1 | Out-Null
    Start-Sleep -Seconds 6
    return (Test-WslOutbound)
}

function Test-WindowsOnline {
    try {
        return (Test-NetConnection -ComputerName $RemoteHost -Port 22 -WarningAction SilentlyContinue -InformationLevel Quiet)
    } catch { return $false }
}

$tunnel = $null
$worker = $null
$lastProbe = [DateTime]::MinValue
$softRepairCount = 0
$lastShutdown = [DateTime]::MinValue
Write-SupervisorLog 'supervisor started'

try {
    while (-not (Test-Path -LiteralPath $StopFile)) {
        # ---------- 1. 隧道守护 ----------
        if ($null -eq $tunnel -or $tunnel.HasExited) {
            $tunnel = Start-Tunnel
            Write-SupervisorLog "tunnel started pid=$($tunnel.Id)"
            Start-Sleep -Seconds 3
            if ($tunnel.HasExited) {
                Write-SupervisorLog 'tunnel exited immediately (ssh unreachable?) backoff 25s'
                Start-Sleep -Seconds 25
                continue
            }
        }

        # ---------- 2. 健康检查 ----------
        $healthy = $false
        try {
            $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 8
            $healthy = ($health.status -eq 'ok')
        } catch {
            Write-SupervisorLog "health check failed: $($_.Exception.Message)"
        }

        if (-not $healthy) {
            Write-SupervisorLog 'tunnel unhealthy, restarting tunnel'
            Stop-Child $tunnel
            $tunnel = $null
            Start-Sleep -Seconds 5
            continue
        }

        # ---------- 3. worker 守护 ----------
        if ($null -eq $worker -or $worker.HasExited) {
            $worker = Start-Worker
            Write-SupervisorLog "worker started pid=$($worker.Id)"
        }

        # ---------- 4. WSL2 出网探测与恢复(每 20 秒一次) ----------
        if ($null -ne $worker -and -not $worker.HasExited -and ((Get-Date) - $lastProbe).TotalSeconds -ge 20) {
            $lastProbe = Get-Date
            if (Test-WslOutbound) {
                if ($softRepairCount -gt 0) {
                    Write-SupervisorLog "WSL2 outbound recovered (failed soft repairs: $softRepairCount)"
                    $softRepairCount = 0
                }
            } else {
                if (Test-WindowsOnline) {
                    $softRepairCount++
                    Write-SupervisorLog "WSL2 outbound DOWN, soft repair #$softRepairCount"
                    if (Repair-WslNet) {
                        Write-SupervisorLog 'WSL2 outbound recovered by soft repair'
                        $softRepairCount = 0
                    } elseif ($softRepairCount -ge $SoftRepairThreshold -and ((Get-Date) - $lastShutdown).TotalMinutes -ge $ShutdownCooldownMin) {
                        Write-SupervisorLog 'soft repair failed -> wsl --shutdown fallback (worker will auto-restart)'
                        Stop-Child $worker
                        $worker = $null
                        & wsl.exe --shutdown 2>&1 | Out-Null
                        $lastShutdown = Get-Date
                        $softRepairCount = 0
                        Start-Sleep -Seconds 12
                    }
                } else {
                    Write-SupervisorLog "Windows cannot reach $RemoteHost, skip WSL2 repair"
                }
            }
        }

        Start-Sleep -Seconds 10
    }
} finally {
    Stop-Child $worker
    Stop-Child $tunnel
    Write-SupervisorLog 'supervisor stopped'
}