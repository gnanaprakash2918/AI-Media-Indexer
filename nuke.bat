@echo off
echo ==================================================
echo      NUCLEAR WIPE - FORCE RESET
echo ==================================================
echo.

echo [1/5] Killing processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1
taskkill /F /IM uvicorn.exe >nul 2>&1

echo [2/5] FORCE stopping Docker containers...
docker compose down -v 2>nul
docker rm -f media_agent_qdrant >nul 2>&1
docker rm -f media_agent_postgres >nul 2>&1
docker rm -f media_agent_minio >nul 2>&1
docker rm -f media_agent_redis >nul 2>&1
docker rm -f media_agent_clickhouse >nul 2>&1
docker rm -f media_agent_langfuse >nul 2>&1
docker rm -f media_agent_langfuse_worker >nul 2>&1
docker rm -f media_agent_createbuckets >nul 2>&1

echo [3/5] Deleting data folders and databases...
if exist qdrant_data rmdir /s /q qdrant_data
if exist qdrant_data_embedded rmdir /s /q qdrant_data_embedded
if exist postgres_data rmdir /s /q postgres_data
if exist thumbnails rmdir /s /q thumbnails
if exist logs rmdir /s /q logs
if exist .cache rmdir /s /q .cache
if exist jobs.db del /f /q jobs.db
if exist identity.db del /f /q identity.db

echo [4/5] verifying...
if exist qdrant_data echo ERROR: qdrant_data still exists!
if exist postgres_data echo ERROR: postgres_data still exists!
if exist jobs.db echo ERROR: jobs.db still exists!

echo.
echo [5/5] WIPE COMPLETE. 
echo Now run: .\start.ps1 -SkipClean
echo.
pause
