import os
import gradio as gr
from pathlib import Path
import uuid
import random
import string
import time
from typing import Optional

# 生成不重复的随机文件名
def generate_random_wav_filename(
    file_ext: str,
    prefix: Optional[str] = "audio",  # 文件名前缀
    use_timestamp: bool = True,       # 是否加入时间戳
    random_length: int = 8,           # 随机字符串长度
    
) -> str:
    """
    生成随机的 WAV 文件名（合规、唯一、易识别）
    
    Args:
        prefix: 文件名前缀（如 "recording" "audio"）
        use_timestamp: 是否加入时间戳（格式：YYYYMMDDHHMMSS）
        random_length: 随机字符长度（建议 6-10 位，保证唯一性）
    
    Returns:
        str: 随机 WAV 文件名（如 "audio_20250120153045_8792ab56.wav"）
    """
    # 1. 定义随机字符池（字母+数字，避免易混淆字符：0/O、1/l）
    safe_chars = string.ascii_lowercase + string.digits
    safe_chars = safe_chars.replace("0", "").replace("o", "").replace("1", "").replace("l", "")
    
    # 2. 生成随机字符串
    random_str = ''.join(random.choice(safe_chars) for _ in range(random_length))
    
    # 3. 生成时间戳（可选，格式：YYYYMMDDHHMMSS）
    timestamp = time.strftime("%Y%m%d%H%M%S") if use_timestamp else ""
    
    # 4. 拼接文件名（前缀 + 时间戳 + 随机字符串 + .wav）
    filename_parts = [prefix]
    if timestamp:
        filename_parts.append(timestamp)
    filename_parts.append(random_str)
    
    # 5. 拼接并添加后缀
    filename = "_".join(filename_parts) + ".wav"
    
    return filename
# 保存上载文件到指定文件夹
def save_uploaded_file(file_path, target_dir):
    """
    修复：适配Gradio的filepath类型（传入的是文件路径字符串，而非文件对象）
    复制上传的文件到指定目录并返回新路径
    """
    if not file_path or not isinstance(file_path, str):
        return None
        
    print(f"文件名 {file_path}")
    # 从路径中提取文件名和后缀
    original_filename = os.path.basename(file_path)
    file_ext = os.path.splitext(original_filename)[1]
    # 生成唯一文件名避免冲突
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    save_path = target_dir / unique_filename
    
    # 复制文件到目标目录
    try:
        with open(file_path, "rb") as src_file, open(save_path, "wb") as dst_file:
            dst_file.write(src_file.read())
        return str(save_path)
    except Exception as e:
        raise gr.Error(f"保存文件失败：{str(e)}")



