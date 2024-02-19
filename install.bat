@CALL "%~dp0micromamba.exe" create -n lgm-venv-001 python=3.10.9 git git-lfs -c conda-forge -r "%~dp0\" -y
@CALL "%~dp0micromamba.exe" shell init --shell=cmd.exe --prefix="%~dp0\"
@CALL condabin\micromamba.bat activate lgm-venv-001

:: xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
:: for example, we use torch 2.1.0 + cuda 18.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
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
@CALL if not exist pretrained
@CALL mkdir pretrained
@CALL curl -o ./pretrained/model.safetensors https://huggingface.co/ashawkey/LGM/blob/main/model.safetensors
@CALL curl -o ./pretrained/model_fp16.safetensors https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16.safetensors

@CALL python -B app.py big --resume pretrained/model_fp16.safetensors

@CALL PAUSE