# EviVN Studio for Windows

- [Download version 0.2.2](https://github.com/lingxiaolu7qinglong/Evi-VN/releases/download/v0.2.2/EviVNStudio-WebSetup-0.2.2.exe)
- [SHA-256 file](EviVNStudio-WebSetup-0.2.2.sha256)

Run the installer, then launch EviVN Studio. The first launch downloads the
Python runtime dependencies and TabICL v2 checkpoint.

Verify the downloaded installer in PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 .\EviVNStudio-WebSetup-0.2.2.exe
```

Expected SHA-256:

```text
4535D0F0E5DD200CD2EF5C8472C897A996B340301669A191CDBB2DAFC52CA29F
```
