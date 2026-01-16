import torch
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 加载本地Whisper模型（无librosa/resampy依赖）
def load_local_whisper_transformers(
    model_dir: str,
    language: str = "Chinese",
    task: str = "transcribe"
):
    """
        加载本地Whisper模型(无librosa/resampy依赖)
    """
    # 绝对路径校验，避免相对路径问题
    # model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"本地模型文件夹不存在：{model_dir}")
    
    # 核心文件校验（权重二选一，配置文件必需）
    weight_files = ["pytorch_model.bin", "model.safetensors"]
    config_files = ["config.json", "preprocessor_config.json", "tokenizer.json"]
    weight_exists = any(os.path.exists(os.path.join(model_dir, f)) for f in weight_files)
    missing_config = [f for f in config_files if not os.path.exists(os.path.join(model_dir, f))]
    
    if not weight_exists:
        raise FileNotFoundError(f"缺少权重文件（二选一）：{weight_files}")
    if missing_config:
        raise FileNotFoundError(f"缺少配置文件：{missing_config}")

    
    # 增强MPS检测：同时校验是否可用+是否编译支持（避免误判）
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16  # CUDA用float16，提速+降显存
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        torch_dtype = torch.float16  # MPS支持float16，性能更优（也可保留float32，精度更高）
    else:
        device = "cpu"
        torch_dtype = torch.float32  # CPU用float32，稳定性优先

    # 增强日志：输出设备详情（如GPU型号/MPS标识）
    device_detail = torch.cuda.get_device_name(0) if device == "cuda" else "Apple Silicon" if device == "mps" else "x86/ARM CPU"
    print(f"📌 运行设备：{device} ({device_detail}) | 数据类型：{torch_dtype}")

    # 加载本地处理器（强制离线）
    processor = WhisperProcessor.from_pretrained(
        model_dir,
        language=language,
        task=task,
        local_files_only=True  # 禁止联网，仅读本地文件
    )

    # 加载本地模型（强制离线）
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True  # 减少内存占用
    ).to(device)

    # 强制中文解码（关键：避免语言识别错误）
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task=task
    )

    return processor, model, forced_decoder_ids, device

def transcribe_audio_to_chinese(
    audio_path: str,
    processor,
    model,
    forced_decoder_ids,
    device: str
) -> str:
    """提取音频中的汉字(纯transformers实现,无额外依赖）"""
    # 1. 校验音频文件
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    # 2. 核心：用WhisperProcessor直接加载音频（自动处理采样率/声道）
    # 无需librosa/resampy，processor内置FFmpeg解码逻辑
    print(f"🔊 预处理音频：{audio_path}")
    input_features = processor(
        audio=os.path.abspath(audio_path),          # 直接传音频文件路径
        sampling_rate=16000,       # Whisper强制16kHz
        return_tensors="pt",       # 返回PyTorch张量
        padding=True               # 自动填充
    ).input_features.to(device)   # 移到指定设备（GPU/CPU）， 既支持 torch.device 对象，也支持 字符串（str）

    # 3. 模型推理（优化中文参数）
    print("⚙️ 模型推理中...")
    with torch.no_grad():  # 推理禁用梯度计算
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids, # 中文转写核心：强制模型输出中文（禁用自动检测）
            max_new_tokens=4096,       # 适配长音频 4096 覆盖绝大多数长语音转写需求
            num_beams=5,               # 束搜索提升准确率 5 = 平衡准确率和速度（中文推荐 3-5）
            temperature=0.0,           # 0=确定性输出，无随机性
            repetition_penalty=1.1,    # 抑制重复文本 轻微惩罚（中文避免过度截断语义）
            no_repeat_ngram_size=3     # 禁止3字以上重复
        )

    # 4. 解码为汉字（跳过特殊token）
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0].strip()

    return transcription

def transcribe(model_dir: str,
    audio_dir: str,
    language: str = "Chinese",
    task: str = "transcribe"
) -> str: 
    
    processor, model, forced_decoder_ids, device = load_local_whisper_transformers(
        model_dir=model_dir,
        language="Chinese"
    )

    transcription = transcribe_audio_to_chinese(
            audio_path=audio_dir,
            processor=processor,
            model=model,
            forced_decoder_ids=forced_decoder_ids,
            device=device
        )
    return transcription

