import os
from indextts.infer_v2 import IndexTTS2
from tools.ai_utils import *  # noqa: F403

# ===================== TTS 初始化适配函数 =====================
def init_indexTTS2(model_dir: str) -> IndexTTS2:
    """
    适配多设备的 IndexTTS2 初始化
    :param model_dir: 模型目录路径
    :return: 初始化后的 IndexTTS2 实例
    """

   
    # 3. 拼接配置文件路径
    cfg_path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件不存在：{cfg_path}")

    # 4. 初始化 IndexTTS2
    tts = IndexTTS2(
        model_dir=model_dir,
        cfg_path=cfg_path,
        use_fp16=False,
        use_deepspeed=False,
        use_cuda_kernel=False,
    )



    return tts