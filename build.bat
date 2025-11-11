@echo off
setlocal

set NUM_CORES=4

echo Cleaning the build folder...
if exist build (
    rmdir /s /q build
)

echo Creating new build folder...
mkdir build
cd build

echo Starting CMake configuration...
cmake ..

if errorlevel 1 goto error_handler

echo Building the project...
if defined NUM_CORES (
    cmake --build . --config Release -j%NUM_CORES%
) else (
    cmake --build . --config Release
)

if errorlevel 1 goto error_handler

echo.
echo =======================================================
echo Build completed successfully.
echo Application files are located in the build\Release\ directory.
echo =======================================================
goto end

:error_handler
echo.
echo =======================================================
echo ERROR: CMake configuration or build failed.
echo =======================================================

:end
endlocal
pause