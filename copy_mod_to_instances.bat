@echo off
echo ================================================
echo Copia SyntheticSoulMod a tutte le istanze
echo ================================================

set SOURCE_FOLDER="C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight\hollow_knight_Data\Managed\Mods\SyntheticSoulMod"

echo.
echo Copiando in Instance 1...
xcopy /E /I /Y %SOURCE_FOLDER% "C:\Users\plumi\Desktop\HK_Instance_1\hollow_knight_Data\Managed\Mods\SyntheticSoulMod"

echo.
echo Copiando in Instance 2...
xcopy /E /I /Y %SOURCE_FOLDER% "C:\Users\plumi\Desktop\HK_Instance_2\hollow_knight_Data\Managed\Mods\SyntheticSoulMod"

echo.
echo Copiando in Instance 3...
xcopy /E /I /Y %SOURCE_FOLDER% "C:\Users\plumi\Desktop\HK_Instance_3\hollow_knight_Data\Managed\Mods\SyntheticSoulMod"

echo.
echo ================================================
echo Copia completata!
echo ================================================
echo.
echo Ora:
echo 1. Chiudi tutte le istanze di Hollow Knight
echo 2. Riavviale (doppio click su hollow_knight.exe in ogni cartella)
echo 3. Avvia il training (script su discord canale codice)
echo.
pause
