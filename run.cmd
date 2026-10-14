@echo off
rem  Shortcuts for the commands this project needs, so none of them has to be
rem  remembered. Double-click to open the Web UI; pass a word for anything else.
setlocal
cd /d "%~dp0"
chcp 65001 >nul

set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=web"

if /i "%ACTION%"=="web"   goto :web
if /i "%ACTION%"=="ui"    goto :web
if /i "%ACTION%"=="cli"   goto :cli
if /i "%ACTION%"=="test"  goto :test
if /i "%ACTION%"=="setup" goto :setup
if /i "%ACTION%"=="help"  goto :help
if /i "%ACTION%"=="-h"    goto :help
if /i "%ACTION%"=="--help" goto :help
echo.
echo   不明なコマンド: %ACTION%
goto :help

:help
echo.
echo   run.cmd            Web UI を開く ^(既定^)
echo   run.cmd cli ...    CLI を実行する   例: run.cmd cli --image a.png
echo   run.cmd test       テストを実行する
echo   run.cmd setup      依存関係を入れ直す
echo.
echo   Web UI は http://127.0.0.1:7860 で開きます。
echo   別のポートにしたいときは: run.cmd web --port 7861
echo.
goto :done

:setup
call :need_uv || goto :failed
echo   依存関係を入れています...
uv sync --extra web --group test
if errorlevel 1 goto :failed
echo   完了しました。
goto :done

:test
call :need_uv || goto :failed
uv run pytest -q
if errorlevel 1 goto :failed
goto :done

:cli
call :need_uv || goto :failed
call :need_ollama
rem  Everything after the word "cli", passed straight through.
set "REST=%*"
call set "REST=%%REST:*%1=%%"
uv run danbooru-prompt %REST%
if errorlevel 1 goto :failed
goto :done

:web
call :need_uv || goto :failed
call :need_ollama
rem  Everything after the word "web", so --port and --host still work.
set "REST="
if /i "%~1"=="web" set "REST=%*" & call set "REST=%%REST:*%1=%%"
if /i "%~1"=="ui"  set "REST=%*" & call set "REST=%%REST:*%1=%%"
echo.
echo   Web UI を開きます: http://127.0.0.1:7860
echo   止めるときはこのウィンドウで Ctrl+C を押してください。
echo.
uv run danbooru-prompt-web %REST%
if errorlevel 1 goto :failed
goto :done

rem --------------------------------------------------------------- helpers
:need_uv
where uv >nul 2>&1
if errorlevel 1 (
  echo.
  echo   uv が見つかりません。https://docs.astral.sh/uv/ を入れてください。
  exit /b 1
)
exit /b 0

:need_ollama
rem  Without Ollama every run fails with a bare connection error that reads
rem  like a bug in this project, so it is started here rather than left to
rem  surprise someone mid-run.
curl -s -m 2 http://127.0.0.1:11434/api/version >nul 2>&1
if not errorlevel 1 exit /b 0
where ollama >nul 2>&1
if errorlevel 1 (
  echo.
  echo   Ollama が起動しておらず、見つかりません。
  echo   先に 'ollama serve' を実行してください。
  exit /b 0
)
echo   Ollama が起動していないので起動します...
start "ollama serve" /min ollama serve
for /l %%i in (1,1,30) do (
  curl -s -m 2 http://127.0.0.1:11434/api/version >nul 2>&1
  if not errorlevel 1 goto :ollama_up
  rem  ping, not timeout: timeout refuses to run when stdin is redirected.
  ping -n 2 127.0.0.1 >nul
)
echo   Ollama の応答がありません。そのまま続けます。
exit /b 0
:ollama_up
echo   Ollama を起動しました。
exit /b 0

:failed
echo.
echo   失敗しました。上のメッセージを確認してください。
pause
exit /b 1

:done
endlocal
exit /b 0
