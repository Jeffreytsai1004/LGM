@CALL "%~dp0micromamba.exe" create -n lgm-venv-001 python=3.10.9 git git-lfs -c conda-forge -r "%~dp0\" -y
@CALL "%~dp0micromamba.exe" shell init --shell=cmd.exe --prefix="%~dp0\"
@CALL condabin\micromamba.bat activate lgm-venv-001

:: Set environment variables
@CALL set GDOWN_CACHE=cache\gdown
@CALL set TORCH_HOME=cache\torch
@CALL set HF_HOME=cache\huggingface
@CALL set PYTHONDONTWRITEBYTECODE=1

:: Install dependencies
:: 安装依赖
@CALL pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
@CALL pip install xformers==0.0.20 --no-cache-dir
@CALL pip install -r requirements.txt
@CALL pip install triton-2.0.0-cp310-cp310-win_amd64.whl

:: Install other dependencies
:: 安装其他依赖
@CALL pip install -r requirements.txt


:: Clone LGM
@CALL git clone https://github.com/3DTopia/LGM

@CALL cd LGM
:: a modified gaussian splatting (+ depth, alpha rendering)
:: 修改后的 gaussian splatting （+ depth、alpha 渲染）
@CALL git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
@CALL pip install ./diff-gaussian-rasterization
:: for mesh extraction
:: 用于网格提取
@CALL pip install git+https://github.com/NVlabs/nvdiffrast
:: download pretrained model
:: 下载预训练模型
@CALL mkdir pretrained
@CALL curl -o ./pretrained/model.safetensors https://huggingface.co/ashawkey/LGM/blob/main/model.safetensors
@CALL curl -o ./pretrained/model_fp16.safetensors https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16.safetensors

:: Launch the WebUI
:: 启动网页
@CALL python -B app.py big

@CALL PAUSE