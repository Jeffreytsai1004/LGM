@CALL "%~dp0micromamba.exe" create -n lgm-venv-001 python=3.10.9 git git-lfs -c conda-forge -r "%~dp0\" -y
@CALL "%~dp0micromamba.exe" shell init --shell=cmd.exe --prefix="%~dp0\"
@CALL condabin\micromamba.bat activate lgm-venv-001

:: xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
:: for example, we use torch 2.1.0 + cuda 18.1
@CALL pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
@CALL pip install xformers==0.0.24+cu118 --index-url https://download.pytorch.org/whl/cu118

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
python app.py big --workspace workspace --test_path data_test

@CALL PAUSE