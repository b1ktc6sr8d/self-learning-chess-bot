$url = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
$out = Join-Path $env:TEMP 'vc_redist.x64.exe'
Write-Host "Downloading VC++ Redistributable..."
Invoke-WebRequest -Uri $url -OutFile $out
Write-Host "Installing..."
Start-Process -FilePath $out -ArgumentList '/install','/quiet','/norestart' -Wait
Write-Host "Done."
