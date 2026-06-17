param(
    [int]$IntervalSeconds = 1800,
    [int]$ContextSize = 8192,
    [int]$MinFreeMb = 1024
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Clean-Ram {
    Write-Host "[runner] Cleaning RAM to free up 5+ GB for model..."
    $RamMapPaths = @("C:\Users\sao\Desktop\RAMMap64.exe", "C:\Users\sao\Desktop\RAMMap.exe")
    $RamMap = $null
    foreach ($path in $RamMapPaths) {
        if (Test-Path $path) {
            $RamMap = $path
            break
        }
    }

    if ($RamMap) {
        Write-Host "[runner] Found RAMMap at $RamMap. Executing all cleaning options..."
        # Run RAMMap with all emptying options. We use -Wait to ensure memory is cleared before proceeding.
        Start-Process -FilePath $RamMap -ArgumentList "-Ew", "/accepteula" -Wait -WindowStyle Hidden
        Start-Process -FilePath $RamMap -ArgumentList "-Es", "/accepteula" -Wait -WindowStyle Hidden
        Start-Process -FilePath $RamMap -ArgumentList "-Em", "/accepteula" -Wait -WindowStyle Hidden
        Start-Process -FilePath $RamMap -ArgumentList "-E0", "/accepteula" -Wait -WindowStyle Hidden
        Write-Host "[runner] RAM cleaned successfully using RAMMap."
    } else {
        Write-Host "[runner] RAMMap not found on Desktop, skipping RAM cleaning."
    }
    
    [System.GC]::Collect()
}

Clean-Ram
Write-Host "[runner] Starting local worker using llama-cpp-python natively..."
python "$Root\local_llama_worker.py" --interval $IntervalSeconds --min-free-mb $MinFreeMb
