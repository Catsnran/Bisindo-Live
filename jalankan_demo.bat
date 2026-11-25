@echo off
setlocal
title BISINDO Live - Demo (server + Cloudflare Tunnel)
cd /d "%~dp0"

echo ============================================================
echo   BISINDO Live - menjalankan server + link publik Cloudflare
echo ============================================================
echo.

rem ---- 1. Cek Python venv & model -------------------------------
set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" goto :no_venv
if not exist "models\best_model.joblib" goto :no_model

rem ---- 2. Cari cloudflared (pasang lewat winget bila belum ada) --
call :find_cf
if defined CF goto :start_server
echo [INFO] cloudflared belum terpasang. Memasang lewat winget...
winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements
call :find_cf
if not defined CF goto :no_cf

rem ---- 3. Jalankan server di jendela terpisah -------------------
:start_server
echo [1/3] Menyalakan server FastAPI di jendela "BISINDO Server"...
start "BISINDO Server" cmd /k ""%PY%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo [2/3] Menunggu server siap (memuat model bisa 10-30 detik)...
set /a TRIES=0
:wait_loop
set /a TRIES+=1
curl -s -o nul http://127.0.0.1:8000/health && goto :server_ready
if %TRIES% geq 60 goto :server_timeout
timeout /t 2 /nobreak >nul
goto :wait_loop

:server_ready
echo        Server siap: http://localhost:8000
start "" http://localhost:8000
echo.
echo [3/3] Membuka Cloudflare Tunnel...
echo ------------------------------------------------------------
echo   Cari baris berisi  https://....trycloudflare.com  di bawah.
echo   Itu LINK PUBLIK yang bisa dibuka dari HP / laptop lain.
echo.
echo   Berhenti: tekan Ctrl+C di jendela ini, lalu tutup
echo   jendela "BISINDO Server".
echo ------------------------------------------------------------
echo.
"%CF%" tunnel --url http://127.0.0.1:8000
goto :eof

rem ---- Subrutin & pesan error ----------------------------------
:find_cf
set "CF="
where cloudflared >nul 2>nul && set "CF=cloudflared"
if not defined CF if exist "%~dp0cloudflared.exe" set "CF=%~dp0cloudflared.exe"
if not defined CF if exist "%ProgramFiles(x86)%\cloudflared\cloudflared.exe" set "CF=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
if not defined CF if exist "%ProgramFiles%\cloudflared\cloudflared.exe" set "CF=%ProgramFiles%\cloudflared\cloudflared.exe"
exit /b 0

:no_venv
echo [ERROR] Python .venv tidak ditemukan: "%PY%"
echo         Dari folder PROJECT BISINDO jalankan:
echo           python -m venv .venv
echo           .venv\Scripts\pip install -r bisindo_ml\requirements.txt
pause
exit /b 1

:no_model
echo [ERROR] File models\best_model.joblib belum ada.
echo         Jalankan dulu BISINDO_Training.ipynb sampai bagian 9 (Simpan Model).
pause
exit /b 1

:no_cf
echo [ERROR] cloudflared tidak ditemukan.
echo         Unduh cloudflared-windows-amd64.exe dari
echo           https://github.com/cloudflare/cloudflared/releases/latest
echo         ganti namanya menjadi cloudflared.exe, taruh di folder bisindo_ml,
echo         lalu jalankan file ini lagi.
pause
exit /b 1

:server_timeout
echo [ERROR] Server tidak merespons setelah 2 menit.
echo         Lihat pesan error di jendela "BISINDO Server".
pause
exit /b 1
