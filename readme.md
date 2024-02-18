
## Large Multi-View Gaussian Model
## 大型多视图高斯模型

This is the official implementation of *LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation*.
这是*LGM：用于高分辨率 3D 内容创建的大型多视图高斯模型* 的官方实现。

### [Project Page](https://me.kiui.moe/lgm/) | [Arxiv](https://arxiv.org/abs/2402.05054) | [Weights](https://huggingface.co/ashawkey/LGM) | <a href="https://huggingface.co/spaces/ashawkey/LGM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>
### [项目页面](https://me.kiui.moe/lgm/) | [Arxiv](https://arxiv.org/abs/2402.05054) | [Weights](https://huggingface.co/ashawkey/LGM) | <a href="https://huggingface.co/spaces/ashawkey/LGM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>

https://github.com/3DTopia/LGM/assets/25863658/cf64e489-29f3-4935-adba-e393a24c26e8

### Replicate Demo:
### Demo 演示:
* gaussians: [demo](https://replicate.com/camenduru/lgm) | [code](https://github.com/camenduru/LGM-replicate)
* mesh: [demo](https://replicate.com/camenduru/lgm-ply-to-glb) | [code](https://github.com/camenduru/LGM-ply-to-glb-replicate)

Thanks [@camenduru](https://github.com/camenduru)!
谢谢 [@camenduru](https://github.com/camenduru)!

### Install
### 安装

```bash
# xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
# 需要 xformers！ 详情请参阅 https://github.com/facebookresearch/xformers for details.
# for example, we use torch 2.1.0 + cuda 18.1
# 例如我们使用torch 2.1.0 + cuda 18.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118

# a modified gaussian splatting (+ depth, alpha rendering)
# 修改后的 gaussian splatting （+ depth、alpha 渲染）
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# for mesh extraction
# 用于网格提取
pip install git+https://github.com/NVlabs/nvdiffrast

# other dependencies
# 其他依赖项
pip install -r requirements.txt
```

### Pretrained Weights
### 预训练权重

Our pretrained weight can be downloaded from [huggingface](https://huggingface.co/ashawkey/LGM).
我们的预训练权重可以从以下位置下载 [huggingface](https://huggingface.co/ashawkey/LGM).

For example, to download the fp16 model for inference:
例如，下载 fp16 模型进行推理：
```bash
mkdir pretrained && cd pretrained
wget https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16.safetensors
cd ..
```

For [MVDream](https://github.com/bytedance/MVDream) and [ImageDream](https://github.com/bytedance/ImageDream), we use a [diffusers implementation](https://github.com/ashawkey/mvdream_diffusers).
对于 [MVDream](https://github.com/bytedance/MVDream) 和 [ImageDream](https://github.com/bytedance/ImageDream)，我们使用 [diffusers implementation](https://github.com/ashawkey/mvdream_diffusers)。
Their weights will be downloaded automatically.
它们的权重将自动下载。

### Inference
### 推理

Inference takes about 10GB GPU memory (loading all imagedream, mvdream, and our LGM).
推理需要大约 10GB GPU 内存（加载所有 imagedream、mvdream 和我们的 LGM）。

```bash
### gradio app for both text/image to 3D
### 适用于文本/图像转 3D 的 gradio 应用程序
python app.py big --resume pretrained/model_fp16.safetensors

### test
### 测试
# --workspace: folder to save output (*.ply and *.mp4)
# --workspace：保存输出的文件夹（*.ply 和 *.mp4）
# --test_path: path to a folder containing images, or a single image
# --test_path：包含图像或单个图像的文件夹的路径
python infer.py big --resume pretrained/model_fp16.safetensors --workspace workspace_test --test_path data_test 

### local gui to visualize saved ply
### 本地 GUI 用于可视化保存的层
python gui.py big --output_size 800 --test_path workspace_test/saved.ply

### mesh conversion
### 网格转换
python convert.py big --test_path workspace_test/saved.ply
```

For more options, please check [options](./core/options.py).
更多选项请查看[options](./core/options.py)。

### Training
### 训练

**NOTE**: 
**注意**: 
Since the dataset used in our training is based on AWS, it cannot be directly used for training in a new environment.
We provide the necessary training code framework, please check and modify the [dataset](./core/provider_objaverse.py) implementation!
由于我们训练中使用的数据集是基于AWS的，所以不能直接用于新环境中的训练。
我们提供了必要的训练代码框架，请检查并修改[数据集](./core/provider_objaverse.py)实现！

We also provide the **~80K subset of [Objaverse](https://objaverse.allenai.org/objaverse-1.0)** used to train LGM in [objaverse_filter](https://github.com/ashawkey/objaverse_filter).
我们还提供了 ** [Objaverse](https://objaverse.allenai.org/objaverse-1.0) 的~80K子集 **，用于在  [objaverse_filter](https://github.com/ashawkey/objaverse_filter) 中训练 LGM) ）。

```bash
# debug training
# 调试训练
accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_debug

# training (use slurm for multi-nodes training)
# 训练（使用slurm进行多节点训练）
accelerate launch --config_file acc_configs/gpu8.yaml main.py big --workspace workspace
```

### Acknowledgement
### 致谢

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
这项工作建立在许多令人惊叹的研究工作和开源项目的基础上，非常感谢所有作者的分享！

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)
- [tyro](https://github.com/brentyi/tyro)

### Citation
### 引文

```
@article{tang2024lgm,
  title={LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation},
  author={Tang, Jiaxiang and Chen, Zhaoxi and Chen, Xiaokang and Wang, Tengfei and Zeng, Gang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2402.05054},
  year={2024}
}
```
