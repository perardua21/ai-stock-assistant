@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."

rem db/migrations/*.sql 을 이름 순서대로 지금 DB에 적용한다.
rem 처음 docker compose up 할 때는 자동으로 적용되지만, 이미 DB가 있는 상태에서
rem 새 migration(예: 0004)이 추가되면 이걸 한 번 실행하면 된다. 여러 번 실행해도 안전하다.

set CONTAINER=ai-stock-db
set DB_USER=ai_stock
set DB_NAME=ai_stock_assistant

set RUNNING=
for /f %%i in ('docker ps -q -f "name=%CONTAINER%" -f "status=running" 2^>nul') do set RUNNING=%%i
if not defined RUNNING goto :not_running

for %%f in (db\migrations\*.sql) do (
    echo 적용: %%~nxf
    docker exec -i %CONTAINER% psql -q -U %DB_USER% -d %DB_NAME% -v ON_ERROR_STOP=1 < "%%f" || goto :fail
)

echo.
echo 완료했습니다. 현재 테이블:
docker exec %CONTAINER% psql -U %DB_USER% -d %DB_NAME% -c "\dt"
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
