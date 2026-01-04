# 配置
- python 3.11.9 设置虚拟环境
- ffmpeg 7.1.1  安装 ffmpeg 工具，设置全局路径

# requirements package
```
gradio==6.2.0
requests==2.32.5
python-multipart==0.0.21
ffmpeg-python==0.2.0
accelerate==1.8.1
cn2an==0.5.22
cython==3.0.7
descript-audiotools==0.7.2
einops>=0.8.1
g2p-en==2.1.0
jieba==0.42.1
json5==0.10.0
keras==2.9.0
librosa==0.10.2.post1
matplotlib==3.8.2
modelscope==1.27.0
munch==4.0.0
numba==0.58.1
numpy==1.26.2
omegaconf>=2.3.0
opencv-python==4.9.0.80
pandas==2.3.2
torchvision==0.23.0
safetensors==0.5.2
sentencepiece>=0.2.1
tensorboard==2.9.1
textstat>=0.7.10
tokenizers==0.21.0
torch==2.8.*
torchaudio==2.8.*
tqdm>=4.67.1
transformers==4.52.1
wetext>=0.0.9
deepspeed=0.18.3
#bitsandbytes==0.49.0  # 可选，用于模型量化（降低显存占用）
#flash-attn  FlashAttention 仅支持 NVIDIA GPU（算力 ≥ 7.0，如 V100/A100/3090/4090），不支持 CPU/AMD GPU/Apple Silicon（M 系列）。

# infinitetalk
diffusers>=0.31.0
imageio==2.37.2
easydict==1.13
dashscope==1.25.5
imageio-ffmpeg==0.6.0
scikit-image==0.25.2
loguru==0.7.3
xfuser>=0.4.1
optimum-quanto==0.2.6
scenedetect==0.6.7.1
moviepy==2.2.1
#decord
ftfy==6.2.0
# 适配transformer==4.52.1, 所有英伟达 GPU（CUDA 11.8+/12.x）;如果适配适配最新 CUDA 12.4+， 版本升级为 0.44.1
bitsandbytes==0.41.1  

# 从音频获取文字
openai-whisper==20250625
pangu==4.0.6.1
zhon==2.1.1

```

# 模型文件下载

 ## 调用openai/whisper-small模型提取音频中的文字
     下载地址： {
               "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
               "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
               "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
               "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
               "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
               "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
               "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
               "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
               "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
               "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
               "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
               "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
           }
     下载到本地./checkpoints/whipser-small/small.pt

 ## 调用Qwen/Qwen2.5-1.5B-Instruct大模型实现文本增强
     下载地址： https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct

 ## 调用IndexTeam/IndexTTS-2模型实现语音克隆
      下载地址： https://hf-mirror.com/IndexTeam/IndexTTS-2
      调用源码下载：https://github.com/index-tts/index-tts
      Index-2用到的4个模型：
        - facebook/w2v-bert-2.0：基于Conformer的W2v-BERT 2.0语音编码器
            下载地址： https://hf-mirror.com/facebook/w2v-bert-2.0
        - amphion/MaskGCT：基于掩码生成编解码器转换器的零样本文本转语音 
            下载地址： https://hf-mirror.com/amphion/MaskGCT
        - funasr/campplus：FunASR 是一个基础的语音识别工具包，具备多种功能，包括语音识别（ASR）、
                         语音活动检测（VAD）、标点恢复、语言模型、说话人验证、说话人分离以及多说话人语音识别。
                         FunASR 提供了便捷的脚本和教程，支持预训练模型的推理和微调
             下载地址： https://hf-mirror.com/funasr/campplus
        - nvidia/bigvgan_v2_22khz_80band_256x：大规模训练的通用神经声码器
             下载地址： https://hf-mirror.com/nvidia/bigvgan_v2_22khz_80band_256x
           
 # 调用 MeiGen-AI/InfiniteTalk 模型生成视频
     下载地址： https://hf-mirror.com/MeiGen-AI/InfiniteTalk  single
     调用源码下载： https://github.com/MeiGen-AI/InfiniteTalk
               