@CALL "%~dp0micromamba.exe" create -n lgm-venv-001 python=3.10.9 git git-lfs -c conda-forge -r "%~dp0\" -y
@CALL "%~dp0micromamba.exe" shell init --shell=cmd.exe --prefix="%~dp0\"
@CALL condabin\micromamba.bat activate lgm-venv-001

:: xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
:: 需要 xformers！ 详情请参阅 https://github.com/facebookresearch/xformers for details.
:: for example, we use torch 2.1.0 + cuda 18.1
:: 例如: 我们使用torch 2.1.0 + cuda 18.1
@CALL pip install torch==2.1.0+cu118 torchvision torchaudio xformer --index-url https://download.pytorch.org/whl/cu118

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

:: other dependencies
:: 其他依赖项
@CALL pip install -r requirements.txt

:: download pretrained model
:: 下载预训练模型
@CALL if not exist pretrained
@CALL mkdir pretrained
@CALL curl -o ./pretrained/model.safetensors https://huggingface.co/ashawkey/LGM/blob/main/model.safetensors
@CALL curl -o ./pretrained/model_fp16.safetensors https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16.safetensors

:: Launch the WebUI
:: 启动网页
@CALL python -B app.py big --resume pretrained/model_fp16.safetensors

@CALL PAUSE