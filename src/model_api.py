import os
from uu import Error
import gradio as gr

from utils import generate_random_filename, save_uploaded_file
from constants import *  
from tools.video_tool import extract_audio_from_video
from tools.audio_tool_plus import transcribe
from tools.enhance_text_tool import qwen_generate
from tools.audio_text_tool import init_indexTTS2


# 视频提取文本
def process_video_to_text(video_file, progress=gr.Progress()):
    if not video_file:
        raise gr.Error("请上传视频文件")
    

    # 保存上传的音频
    progress(0.1, desc="视频文件上传...")
    video_path = save_uploaded_file(video_file, video_upload_path)
    if (video_path == None):
        raise gr.Error("视频文件上载错误")
    print(f"视频文件 保存在 {video_path}")
    progress(0.2, desc="视频文件上传完成")

    # to-do: 调用 ffmpeg 获取音频
    progress(0.3, desc="提取音频...")
    audio_path, isSuccess = extract_audio_from_video(video_path=video_path, output_audio_path=extract_audio_path)
    if isSuccess == False: 
        raise gr.Error("视频提取音频发生错误")
    print(f"音频文件保存在 {audio_path}")
    progress(0.4, desc="音频提取完成")
    
    # to-do: 调用openai/whisper-small模型提取音频中的文字
    # 下载地址： {
    #           "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    #           "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    #           "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    #           "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    #           "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    #           "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    #           "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    #           "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    #           "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    #           "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    #           "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    #           "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    #       }
    # 下载到本地./checkpoints/whipser-small/small.pt
    # 强制离线运行（禁止transformers联网）

    progress(0.5, desc="装载大模型...")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # model_dir 绝对路径
    model_dir = os.path.join(os.getcwd(), "checkpoints/whisper-small/small.pt")
    progress(0.7, desc="装载大模型完成")
    try:
        # 提取汉字
        progress(0.8, desc="大模型推理中...")
        audio_text = transcribe(
            model_dir=model_dir,
            audio_dir=audio_path,
        )

        # 输出结果
        print(f"\n✅ 提取的文本：{audio_text}")
        progress(1.0, desc="推理结束,提取文本完成")
        
        
    except FileNotFoundError as e:
        print(f"提取视频文字出现错误: {e}")
        raise gr.Error("提取视频文字出现错误")
    except Exception as e:
        print(f"提取视频文字出现错误: {e}")
        raise gr.Error("提取视频文字出现错误")   

    return audio_text, video_path
    
# 文本增强
def process_text_enhancement(original_text, description, progress=gr.Progress()):
    
    if not original_text:
        raise gr.Error("请先完成第一步音频转文本")
    if not description:
        raise gr.Error("请输入文案增强要求")

    # to-do: 调用Qwen/Qwen2.5-1.5B-Instruct大模型实现文本增强 mac 适配版（window 另有版本）
    # 下载地址： https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct
    try:
        enhanced_text = qwen_generate(original_text, description)
        return enhanced_text
    except RuntimeError as e:
        print(f"文字增强操作失败: {e}")
        raise gr.Error("增强文字出现错误")
    except Exception as e:
        print(f"文字增强操作失败: {e}")
        raise gr.Error("增强文字出现错误")
    

# 文本+引用音频实现语音克隆
def process_tts(text, ref_audio_file, progress=gr.Progress()):
    if not text:
        raise gr.Error("请先完成文本处理")
    if not ref_audio_file:
        raise gr.Error("请上传参考音频")
    
    # 保存参考音频
    progress(0.1, desc="保存音频文件...")
    ref_audio_path = save_uploaded_file(ref_audio_file, audio_upload_path)

    #  to-do:调用IndexTeam/IndexTTS-2模型实现语音克隆
    #  下载地址： https://hf-mirror.com/IndexTeam/IndexTTS-2
    #   调用源码下载：https://github.com/index-tts/index-tts
    #  Index-2用到的4个模型：
    #    facebook/w2v-bert-2.0：基于Conformer的W2v-BERT 2.0语音编码器
    #        下载地址： https://hf-mirror.com/facebook/w2v-bert-2.0
    #    amphion/MaskGCT：基于掩码生成编解码器转换器的零样本文本转语音 
    #        下载地址： https://hf-mirror.com/amphion/MaskGCT
    #    funasr/campplus：FunASR 是一个基础的语音识别工具包，具备多种功能，包括语音识别（ASR）、
    #                     语音活动检测（VAD）、标点恢复、语言模型、说话人验证、说话人分离以及多说话人语音识别。
    #                     FunASR 提供了便捷的脚本和教程，支持预训练模型的推理和微调
    #         下载地址： https://hf-mirror.com/funasr/campplus
    #    nvidia/bigvgan_v2_22khz_80band_256x：大规模训练的通用神经声码器
    #         下载地址： https://hf-mirror.com/nvidia/bigvgan_v2_22khz_80band_256x
    #  

    output_audio_dir = os.path.join(os.getcwd(), audio_download_path)
    if not os.path.exists(output_audio_dir):
        ensure_directories()

    output_audio_path = output_audio_dir + "/" + generate_random_filename("wav", "audio", True) 
    

    model_dir = os.path.join(os.getcwd(), "checkpoints/IndexTTS-2")
    try:
        progress(0.2, desc="装载模型...")
        tts = init_indexTTS2(model_dir)
        progress(0.4, desc="开始合成...")
        tts.infer(spk_audio_prompt=ref_audio_path, text=text, output_path=output_audio_path, verbose=True)
        progress(1.0, desc="合成完成")
    except FileNotFoundError as e:
        print(f"语音克隆出现错误: {e} ")
        raise gr.Error("语音克隆出现错误")
    except Exception as e:
        print(f"语音克隆出现错误: {e} ")
        raise gr.Error("语音克隆出现错误")    

    return output_audio_path, output_audio_path

# 根据图片和音频生成视频，对齐口型
def process_video_generation(image_file, tts_audio_path, progress=gr.Progress()):
    if not tts_audio_path:
        raise gr.Error("请先完成配音生成")
    if not image_file:
        raise gr.Error("请上传人物正脸图片")
    
    
    # 保存上传的图片
    image_path = save_uploaded_file(image_file, image_upload_path)
    
    # to-do:调用 MeiGen-AI/InfiniteTalk 模型生成视频
    # 下载地址： https://hf-mirror.com/MeiGen-AI/InfiniteTalk  single
    # 调用源码下载： https://github.com/MeiGen-AI/InfiniteTalk
    # 
    
    video_path = None
    return video_path, None  