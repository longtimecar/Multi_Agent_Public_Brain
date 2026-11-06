' Launch (or replace) the membrain dashboard with the code currently on disk.
'
' --restart matters: without it a second launch only opened a browser tab, so a
' relaunch never picked up a code change and the old process kept serving.
Set sh = CreateObject("WScript.Shell")
Set env = sh.Environment("Process")
env("HTTP_PROXY") = ""
env("HTTPS_PROXY") = ""
sh.CurrentDirectory = "D:\MemBrain"
sh.Run """D:\MemBrain\.venv\Scripts\pythonw.exe"" -m membrain.cli dashboard --port 1966 --restart", 0, False
