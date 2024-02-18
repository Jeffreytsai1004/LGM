@CALL "%~dp0micromamba.exe" shell init --shell=cmd.exe --prefix="%~dp0\"
@CALL condabin\micromamba.bat activate lgm-venv-001
:: Launch the WebUI
:: 启动网页
@CALL cd LGM
@CALL python -B app.py big --resume pretrained/model_fp16.safetensors
@CALL PAUSE