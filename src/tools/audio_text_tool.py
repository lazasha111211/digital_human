import os
import torch
from indextts.infer_v2 import IndexTTS2
from ai_utils import *

# ===================== TTS 初始化适配函数 =====================
def init_indexTTS2(model_dir: str) -> IndexTTS2:
    """
    适配多设备的 IndexTTS2 初始化
    :param model_dir: 模型目录路径
    :return: 初始化后的 IndexTTS2 实例
    """

    # 1. 获取当前最优设备
    device = get_accelerator_device()

    # 1. 执行参数支持性校验
    support_check = check_param_support()
    
    print("=== 参数支持性校验结果 ===")
    for param, info in support_check.items():
        if param == "device":
            print(f"当前设备：{info}")
        else:
            print(f"{param}: {'支持' if info['supported'] else '不支持'} | 原因：{info['reason']}")

    # 2. 自动适配参数（优先用 cmd_args，不支持则强制覆盖为 False）
    # use_fp16：用户指定且系统支持则用，否则禁用
    use_fp16 = True if support_check["use_fp16"]["supported"] else False
    # use_deepspeed：用户指定且系统支持则用，否则禁用
    use_deepspeed = True if support_check["use_deepspeed"]["supported"] else False
    # use_cuda_kernel：用户指定且系统支持则用，否则禁用
    use_cuda_kernel = True if support_check["use_cuda_kernel"]["supported"] else False


    # 3. 拼接配置文件路径
    cfg_path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件不存在：{cfg_path}")

    # 4. 初始化 IndexTTS2
    tts = IndexTTS2(
        model_dir=model_dir,
        cfg_path=cfg_path,
        use_fp16=use_fp16,
        use_deepspeed=use_deepspeed,
        use_cuda_kernel=use_cuda_kernel,
    )

    # 可选：设置全局设备（若 TTS 内部需要）
    torch.set_default_device(device)

    

    return tts