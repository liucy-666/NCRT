@echo off
pushd "%~dp0"
python -m ncrt %*
set "NCRT_EXIT=%ERRORLEVEL%"
popd
exit /b %NCRT_EXIT%
