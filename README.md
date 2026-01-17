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
# bitsandbytes==0.41.1  

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
               
# InfiniteTalk 数字人口型同步视频生成

使用以下组件：
- **Wan-AI/Wan2.1-T2V-14B**: 基础视频生成模型（可选，用于生成运动参考）
- **chinese-wav2vec2-base**: 中文语音识别和特征提取
- **InfiniteTalk**: 唇形同步模型

输入：图片 + 音频  
输出：数字人口型同步视频

```
# 安装必要的依赖包
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip3 install transformers accelerate librosa soundfile
!pip3 install diffusers opencv-python pillow numpy
!pip3 install imageio imageio-ffmpeg
!pip3 install einops xformers
```
## InfiniteTalk依赖说明

主要依赖包：
- `torch`: PyTorch 深度学习框架
- `transformers`: HuggingFace 模型库（用于 wav2vec2）
- `librosa`, `soundfile`: 音频处理
- `opencv-python`: 视频处理
- `Pillow`: 图像处理
- `diffusers`: 用于 Wan2.1 模型（如果使用）

### 系统要求：
- **GPU**: 建议 24GB+ VRAM（使用 Wan2.1 和 InfiniteTalk 时）
- **CPU**: 可以运行但速度较慢
- **内存**: 建议 16GB+ RAM               

## 集成 InfiniteTalk 模型说明

要使用完整的 InfiniteTalk 功能，请执行以下步骤：

1. **克隆 InfiniteTalk 仓库**:
   ```bash
   git clone https://github.com/bmwas/InfiniteTalk.git
   cd InfiniteTalk
   pip install -r requirements.txt
   ```

2. **下载模型权重**:
   - 根据 InfiniteTalk 仓库的说明下载预训练模型

3. **修改 lipsync_with_infinitetalk 函数**:
   - 导入 InfiniteTalk 的实际接口
   - 替换简化的实现为真实的模型调用

### 示例集成代码（需要根据实际API调整）：
```python
# 如果 InfiniteTalk 提供了 Python API
from infinitetalk import InfiniteTalkInference

def lipsync_with_infinitetalk_real(image_tensor, audio_features, audio_path, duration, fps=25):
    # 初始化 InfiniteTalk 模型
    model = InfiniteTalkInference(
        model_path="path/to/infinitetalk/model",
        device=device
    )
    
    # 调用推理
    video_frames = model.infer(
        image=image_tensor,
        audio=audio_path,  # 或使用音频特征
        duration=duration,
        fps=fps
    )
    
    return video_frames
```
## 调用参数解析

- Args:
        image_path: 输入图像路径
        audio_path: 音频文件路径
        output_path: 输出视频路径
        infinitetalk_repo_dir: InfiniteTalk 仓库目录
        ckpt_dir: Wan2.1 模型目录
        wav2vec_dir: wav2vec2 模型目录
        infinitetalk_weight: InfiniteTalk 权重路径
        size: 分辨率 "infinitetalk-480" 或 "infinitetalk-720"
        fps: 视频帧率
        sample_steps: 采样步数
        mode: "streaming" 或 "clip"
        motion_frame: 运动帧数
        sample_text_guide_scale: 文本引导尺度
        sample_audio_guide_scale: 音频引导尺度
        num_persistent_param_in_dit: 持久参数数量（低显存可设为 0)
        
-   Returns:
        生成的视频文件路径

        motion_frame: 指的是数字人除基础口型外的肢体 / 面部微运动的帧序列长度，或者是每一段语音 / 文本对应的 运动帧总数。
                它的核心作用是让数字人不只是 “动嘴”，还能搭配自然的头部转动、表情变化、轻微肢体动作，避免数字人呈现僵硬的 “木偶感”。

        sample_steps: 采样步数, 模型生成每帧口型 / 动作时的迭代采样步数（类似扩散模型的采样步数）。
                      步数越多，模型生成的口型 / 动作越精细、越贴合语音 / 文本，但生成速度越慢；步数越少，
                      生成越快，但效果可能粗糙、有 “毛刺”。
                      合理取值：
                        快速预览 / 流式生成(streaming):10~20 步（速度优先）；
                        高质量剪辑生成(clip):30~50 步（效果优先）；
                        低显存场景:≤15 步（减少计算量）。
        mode: "streaming" 或 "clip":  
                        "streaming": 实时流式生成，边输入语音 / 文本边输出帧序列，低延迟 | 实时对话、
                                     直播、语音通话等实时场景 |
                        "clip":先完整接收语音 / 文本，再一次性生成所有帧序列，效果更优 | 短视频、
                                     预录制内容、高质量口型对齐 |
                        关键注意: streaming 模式下需搭配较小的 sample_steps(≤20),避免延迟过高；
                             clip 模式可拉满 sample_steps 追求效果。  
        motion_frame(运动帧数):数字人除口型外的肢体 / 面部微运动帧数（比如头部转动、表情变化），或指 “每段语音对应的运动帧长度”。
                        控制数字人动作的丰富度，避免只有口型动、身体僵硬；若设为 0,数字人仅动嘴,无其他运动  

                        取值逻辑：通常和 “语音时长 * fps” 匹配，比如 5 秒语音(fps=30),motion_frame 
                        可设为 150(完整覆盖）；流式模式下可设为 “fps*1”(每秒更新一次运动帧）
        sample_text_guide_scale(文本引导尺度）: 控制文本内容对口型 / 动作生成的 “引导强度”（权重）
        sample_audio_guide_scale(音频引导尺度）: 控制音频（语音） 对口型对齐的 “引导强度”（权重）
        num_persistent_param_in_dit: 控制 DIT(Diffusion Transformer)模型中 “持久化保存的参数数量”，
                    是显存优化参数。
            作用：
                模型生成帧时，部分参数会被缓存（持久化）以加速后续计算，缓存的参数越多，生成越快，但显存占用越高；设为 0 则不缓存任何参数，显存占用最低，但生成速度变慢。
            取值建议：
                高显存(16G+）：设为 64/128(默认值，速度优先）；
                中显存(8~16G): 设为 32
                低显存(≤8G): 设为 0(避免 OOM 显存溢出）。   

## 生成唇形同步视频
        
        Args:
            image_path: 输入图像路径（I2V 模式）
            audio_path: 音频文件路径
            ref_video_path: 参考视频路径（V2V 模式，可选）
            output_stem: 输出文件名（不含扩展名）
            size: 分辨率 "infinitetalk-480" 或 "infinitetalk-720"
            sample_steps: 采样步数
            mode: "streaming" 或 "clip"
            motion_frame: 运动帧数
            fps: 视频帧率
            seed: 随机种子
            sample_text_guide_scale: 文本引导尺度
            sample_audio_guide_scale: 音频引导尺度
            num_persistent_param_in_dit: 持久参数数量（低显存可设为 0）
            quant: 量化选项
            quant_dir: 量化模型目录
            lora_dir: LoRA 目录
            lora_scale: LoRA 缩放比例
            
        Returns:
            生成的视频文件路径                