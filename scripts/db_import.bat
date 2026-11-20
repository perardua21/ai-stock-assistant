@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."

set CONTAINER=ai-stock-db
set DB_USER=ai_stock
set DB_NAME=ai_stock_assistant

if not exist db.dump goto :no_file

set RUNNING=
for /f %%i in ('docker ps -q -f "name=%CONTAINER%" -f "status=running" 2^>nul') do set RUNNING=%%i
if not defined RUNNING goto :not_running

echo 주의: 지금 내 DB 데이터가 db.dump 내용으로 전부 덮어써집니다.
echo 받기 전에 git pull 로 코드와 마이그레이션을 최신으로 맞췄는지 확인하세요.
echo.
set ANSWER=
set /p ANSWER=계속할까요? [Y/N]: 
if /i not "%ANSWER%"=="Y" goto :cancel

echo.
echo db.dump 내용을 DB에 넣는 중...
docker cp db.dump %CONTAINER%:/tmp/db.dump || goto :fail
docker exec %CONTAINER% pg_restore -U %DB_USER% -d %DB_NAME% --clean --if-exists --no-owner /tmp/db.dump || goto :fail
docker exec %CONTAINER% rm -f /tmp/db.dump >nul

echo.
echo 완료했습니다. 들어간 데이터 확인:
docker exec %CONTAINER% psql -U %DB_USER% -d %DB_NAME% -c "select (select count(*) from stock_master) as stock_master, (select count(*) from price_daily) as price_daily, (select count(*) from index_daily) as index_daily, (select count(*) from label_crash) as label_crash, (select count(*) from forecast_feature) as forecast_feature;"
goto :end

:no_file
echo 레포 폴더에 db.dump 파일이 없습니다.
echo 받은 db.dump를 %CD% 에 넣고 다시 실행하세요.
goto :end

:not_running
echo DB가 켜져 있지 않습니다.
echo Docker Desktop을 켜고 레포 폴더에서 docker compose up -d 를 먼저 실행하세요.
goto :end

:cancel
echo 취소했습니다. DB는 그대로입니다.
goto :end

:fail
echo.
echo 실패했습니다. 위 에러 메시지를 확인하세요.

:end
echo.
pause
