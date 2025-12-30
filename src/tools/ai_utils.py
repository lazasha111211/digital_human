from typing import Dict, Any, Optional
import torch
import importlib
import random
import string
import time
from transformers import BitsAndBytesConfig 

from torch.types import Device

# 是否启用量化（仅 GPU 生效）
USE_QUANTIZATION = True
# 量化位数（4bit/8bit，仅 GPU 生效）
QUANT_BIT = 4

# 自动适配 macOS MPS / Windows CUDA / CPU
def get_accelerator_device() -> Device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")  # macOS MPS
    elif torch.cuda.is_available():
        return torch.device("cuda:0")  # Windows 英伟达 CUDA
    else:
        return torch.device("cpu")
     


# ===================== 核心：参数支持性校验函数 =====================
def check_param_support() -> dict:
    """
    校验当前系统对各参数的支持性，返回校验结果字典
    返回结构：{
        "device": "cuda"/"mps"/"cpu",
        "use_fp16": {"supported": bool, "reason": str},
        "use_deepspeed": {"supported": bool, "reason": str},
        "use_cuda_kernel": {"supported": bool, "reason": str}
    }
    """
    result = {
        "device": None,
        "use_fp16": {"supported": False, "reason": ""},
        "use_deepspeed": {"supported": False, "reason": ""},
        "use_cuda_kernel": {"supported": False, "reason": ""}
    }

    # 1. 先检测基础设备
    if torch.cuda.is_available():
        result["device"] = "cuda"
        # 获取 GPU 信息（用于 FP16 校验）
        gpu_name = torch.cuda.get_device_name(0)
        gpu_capability = torch.cuda.get_device_capability(0)  # (算力主版本, 次版本)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        result["device"] = "mps"
    else:
        result["device"] = "cpu"

    # 2. 校验 use_fp16
    if result["device"] == "cuda":
        # CUDA：算力 ≥5.0 支持 FP16（如 GTX 10xx/RTX 20xx+）
        if gpu_capability >= (5, 0):
            result["use_fp16"]["supported"] = True
            result["use_fp16"]["reason"] = f"CUDA GPU ({gpu_name}) 算力 {gpu_capability[0]}.{gpu_capability[1]} 支持 FP16"
        else:
            result["use_fp16"]["reason"] = f"CUDA GPU ({gpu_name}) 算力 {gpu_capability[0]}.{gpu_capability[1]} < 5.0，不支持 FP16"
    elif result["device"] == "mps":
        # MPS：Apple Silicon 原生支持 FP16
        result["use_fp16"]["supported"] = True
        result["use_fp16"]["reason"] = "MPS (Apple Silicon) 支持 FP16 推理"
    else:
        # CPU：FP16 无收益且易出错
        result["use_fp16"]["reason"] = "CPU 环境下 FP16 无性能收益，且精度稳定性差"

    # 3. 校验 use_deepspeed
    if result["device"] == "cuda":
        # 检查 deepspeed 库是否安装
        if importlib.util.find_spec("deepspeed") is not None:
            result["use_deepspeed"]["supported"] = True
            result["use_deepspeed"]["reason"] = "CUDA 环境 + deepspeed 库已安装，支持 DeepSpeed"
        else:
            result["use_deepspeed"]["reason"] = "CUDA 环境但未安装 deepspeed 库（pip install deepspeed）"
    else:
        result["use_deepspeed"]["reason"] = f"{result['device'].upper()} 环境不支持 DeepSpeed（仅 CUDA 支持）"

    # 4. 校验 use_cuda_kernel
    if result["device"] == "cuda":
        # 检查 CUDA 编译环境（简易版：通过 torch 检测 CUDA 版本是否有效）
        try:
            # 尝试执行简单 CUDA 核操作（验证 CUDA 编译环境）
            torch.randn(1).cuda()
            result["use_cuda_kernel"]["supported"] = True
            result["use_cuda_kernel"]["reason"] = "CUDA 环境有效，支持 CUDA Kernel 加速"
        except Exception as e:
            result["use_cuda_kernel"]["reason"] = f"CUDA 环境异常，不支持 CUDA Kernel：{str(e)[:50]}..."
    else:
        result["use_cuda_kernel"]["reason"] = f"{result['device'].upper()} 环境不支持 CUDA Kernel（仅 CUDA 支持）"

    return result  

def get_model_kwargs(device: str) -> Dict[str, Any]:
    """
    根据设备动态生成模型加载参数（核心适配逻辑）
    :param device: 设备字符串（"cuda"/"mps"/"cpu"）
        当加载非 Hugging Face 官方内置的自定义模型（如 Qwen、Baichuan、LLaMA 等）时，这些模型的配置 / 架构代码并未
    内置在 transformers 库中，而是存储在模型仓库的 modeling_xxx.py/configuration_xxx.py 等文件里（即「远程代码」）。
    trust_remote_code=True 表示：👉 允许 transformers 从模型仓库下载并执行这些自定义代码，以正确加载模型架构和配置
    
    :return: 模型加载参数字典
    """
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,  # 所有设备都开启，降低内存占用
    }

    # -------------------- GPU（CUDA）适配 --------------------
    if device == "cuda":
        # 可以选择量化配置（4bit/8bit），这里通过全局变量设置使用4bit量化配置和，
        if USE_QUANTIZATION and QUANT_BIT in [4, 8]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=(QUANT_BIT == 4),
                load_in_8bit=(QUANT_BIT == 8),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            model_kwargs.update({
                "quantization_config": quantization_config,
                "load_in_4bit": (QUANT_BIT == 4),
                "load_in_8bit": (QUANT_BIT == 8),
            })
        # 也可以选择非量化配置
        else:
            model_kwargs.update({
                "quantization_config": None,
                "load_in_4bit": False,
                "load_in_8bit": False,
            })
        # GPU 最优 dtype：bfloat16（支持的话）/float16
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["device_map"] = device

    # -------------------- MPS（macOS）适配 --------------------
    elif device == "mps":
        model_kwargs.update({
            "quantization_config": None,  # MPS 暂不支持 BitsAndBytes 量化
            "load_in_4bit": False,
            "load_in_8bit": False,
            "torch_dtype": torch.float16,  # MPS 用 float16 提速降内存
            "device_map": device
        })

    # -------------------- CPU 适配 --------------------
    else:  # cpu
        model_kwargs.update({
            "quantization_config": None,  # CPU 量化无收益
            "load_in_4bit": False,
            "load_in_8bit": False,
            "torch_dtype": torch.float32,  # CPU 用 float32 更稳定
            "device_map": device
        })

    return model_kwargs          



# 生成不重复的随机文件名
def generate_random_filename(
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
    filename = "_".join(filename_parts) + "." + file_ext
    
    return filename    