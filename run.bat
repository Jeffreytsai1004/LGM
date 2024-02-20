@CALL "%~dp0micromamba.exe" shell init --shell=cmd.exe --prefix="%~dp0\"
@CALL condabin\micromamba.bat activate lgm-venv-001
:: Launch the WebUI
:: 启动网页
@CALL cd LGM
:: Launch the WebUI
:: 启动网页
:: gradio app for both text/image to 3D
python app.py
@CALL PAUSE