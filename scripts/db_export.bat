@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."

set CONTAINER=ai-stock-db
set DB_USER=ai_stock
set DB_NAME=ai_stock_assistant

set RUNNING=
for /f %%i in ('docker ps -q -f "name=%CONTAINER%" -f "status=running" 2^>nul') do set RUNNING=%%i
if not defined RUNNING goto :not_running

echo DB 내용을 db.dump 파일로 저장하는 중...
docker exec %CONTAINER% pg_dump -U %DB_USER% -d %DB_NAME% -Fc -f /tmp/db.dump || goto :fail
docker cp %CONTAINER%:/tmp/db.dump db.dump || goto :fail
docker exec %CONTAINER% rm -f /tmp/db.dump >nul

echo.
for %%A in (db.dump) do echo 완료: %%~fA  [%%~zA bytes]
echo 이 파일을 팀원에게 전달하면 됩니다. 깃에는 올라가지 않습니다.
goto :end

:not_running
echo DB가 켜져 있지 않습니다.
echo Docker Desktop을 켜고 레포 폴더에서 docker compose up -d 를 먼저 실행하세요.
goto :end

:fail
echo.
echo 실패했습니다. 위 에러 메시지를 확인하세요.

:end
echo.
pause
