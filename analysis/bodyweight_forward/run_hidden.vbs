' Launches collector.py without a visible console window (for the
' PyCaLiAI_BodyWeight scheduled task) and passes its exit code through.
Option Explicit
Dim sh, cmd
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "E:\PyCaLiAI"
cmd = """C:\Users\gutch\AppData\Local\Programs\Python\Python312-32\python.exe"" " & _
      """E:\PyCaLiAI\analysis\bodyweight_forward\collector.py"""
WScript.Quit sh.Run(cmd, 0, True)
