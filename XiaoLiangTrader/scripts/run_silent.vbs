' ============================================================================
' XiaoLiangTrader 静默运行包装脚本 (VBScript)
' 作用: 在后台完全静默执行 run_daily.bat，不弹出任何黑框命令行窗口
' ============================================================================

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' 获取当前脚本所在目录
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptDir & "\run_daily.bat"

' 0 表示隐藏窗口运行，False 表示不阻塞等待
WshShell.Run "cmd /c """ & batPath & """", 0, False
