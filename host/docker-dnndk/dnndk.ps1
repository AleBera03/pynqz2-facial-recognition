param (
    [Alias('r')]
    [Switch]$Restart
)

if ($args.Count -gt 0) {
    Write-Host "USAGE: .\dnndk.ps1 [-Restart|-r]" -ForegroundColor Yellow
    Exit 1
}

if ($Restart) {
    docker restart dnndk-container
}

docker exec -it dnndk-container bash