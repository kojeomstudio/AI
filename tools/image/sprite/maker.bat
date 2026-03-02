@echo off

chcp 65001

pyinstaller --onefile --windowed --add-data "sprite_spliter.py;." -n SpriteSplitter sprite_spliter.py
echo Build Complete!
pause
