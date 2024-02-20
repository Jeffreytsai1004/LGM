@CALL "%~dp0micromamba.exe" shell init --shell=cmd.exe --prefix="%~dp0\"
@CALL condabin\micromamba.bat activate lgm-venv-001
:: Set environment variables
@CALL set GDOWN_CACHE=cache\gdown
@CALL set TORCH_HOME=cache\torch
@CALL set HF_HOME=cache\huggingface
@CALL set PYTHONDONTWRITEBYTECODE=1
:: Launch the WebUI
:: 启动网页
@CALL cd LGM
:: Launch the WebUI
:: 启动网页
:: gradio app for both text/image to 3D
@CALL python -B app.py big --workspace workspace
@CALL PAUSE