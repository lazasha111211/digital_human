# InfiniteTalk API 封装类
import json
import subprocess
import tempfile
import shlex
import os
from pathlib import Path
'''
备用：利用 python 调用generate_infinitetalk.py执行图+音频生成视频，第一选择是直接从代码层面接口调用
'''
# infinitetalk github 源码所在目录
infinitetalk_repo_dir_ = os.path.join(os.getcwd(), "scr/infinitetalk") 
# Wan2.1 模型目录
ckpt_dir_ = os.path.join(os.getcwd(), "checkpoints/Wan2.1-I2V-14B-480P") 
# wav2vec2 模型目录
wav2vec_dir_ = os.path.join(os.getcwd(), "checkpoints/chinese-wav2vec2-base")
# InfiniteTalk 权重 single
infinitetalk_weight_ = os.path.join(os.getcwd(), "checkpoints/InfiniteTalk/single/infinitetalk.safetensors")
# 执行
python_exec_ = "python"

class InfiniteTalkAPI:
    """
    InfiniteTalk API 封装类
    通过调用官方 generate_infinitetalk.py 脚本实现唇形同步
    """
    def __init__(
        self,
        infinitetalk_repo_dir="InfiniteTalk",  # InfiniteTalk 仓库目录
        ckpt_dir="weights/Wan2.1-I2V-14B-480P",  # Wan2.1 模型目录
        wav2vec_dir="weights/chinese-wav2vec2-base",  # wav2vec2 模型目录
        infinitetalk_weight="weights/InfiniteTalk/single/infinitetalk.safetensors",  # InfiniteTalk 权重
        python_exec="python",
    ):
        self.infinitetalk_repo_dir = infinitetalk_repo_dir_
        self.ckpt_dir = ckpt_dir_
        self.wav2vec_dir = wav2vec_dir_
        self.infinitetalk_weight = infinitetalk_weight_
        self.python_exec = python_exec_
        
    def _ensure_paths(self, *paths):
        """检查路径是否存在"""
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"路径不存在: {p}")
    
    def _build_input_json(
        self,
        image_path=None,
        audio_path=None,
        ref_video_path=None,
        # Frames Per Second，每秒帧数
        # 口型对齐的本质是让数字人的口型帧与语音的音频帧一一匹配，fps决定了口型序列的时间粒度：
        # 比如fps=30，表示模型会为每一秒的语音生成 30 帧口型数据（每帧间隔≈33ms）；
        # fps=25则是每秒 25 帧（每帧间隔 40ms），是视频领域的常用帧率。
        # 常见取值：24/25/30（主流视频帧率），60（高流畅度场景），数值越高口型越细腻，但生成的数据量也越大。
        fps=25,
        # 随机种子 决定了模型生成口型序列时的 “随机行为”；infinitetalk 生成口型时，即使输入的语音 / 
        # 文本完全相同，若seed不同，模型输出的口型细节（比如嘴唇开合的幅度、细微的面部肌肉运动帧）会有微小差异；
        # 若seed固定，每次调用模型都会生成完全一致的口型序列。
        seed=42,   
    ):
        """
        构建 InfiniteTalk 的输入 JSON 配置文件
        根据官方 examples/single_example_image.json 格式
        """
        if image_path and ref_video_path:
            raise ValueError("请提供 image_path (I2V) 或 ref_video_path (V2V)，不能同时提供")
        if not audio_path:
            raise ValueError("audio_path 是必需的")
        
        payload = {
            "seed": seed,
            "fps": fps,
        }
        
        if image_path:
            payload["ref_image"] = str(Path(image_path).resolve())
            payload["ref_audio"] = str(Path(audio_path).resolve())
        else:
            payload["ref_video"] = str(Path(ref_video_path).resolve())
            payload["ref_audio"] = str(Path(audio_path).resolve())
        
        return payload
    
    def generate(
        self,
        image_path=None,
        audio_path=None,
        ref_video_path=None,
        output_stem="infinitetalk_output",
        size="infinitetalk-480",  # "infinitetalk-480" 或 "infinitetalk-720"
        sample_steps=40,
        mode="streaming",  # "streaming" 长视频 或 "clip" 短片
        motion_frame=9,
        fps=25,
        seed=42,
        sample_text_guide_scale=5,  # 未使用 LoRA 时推荐 5
        sample_audio_guide_scale=4,  # 未使用 LoRA 时推荐 4
        num_persistent_param_in_dit=None,  # 低显存可设为 0
        quant=None,  # 量化选项，如 "fp8"
        quant_dir=None,
        lora_dir=None,
        lora_scale=None,
    ):

        # 检查路径
        self._ensure_paths(self.infinitetalk_repo_dir, self.ckpt_dir, 
                          self.wav2vec_dir, self.infinitetalk_weight)
        
        if image_path:
            self._ensure_paths(image_path)
        if ref_video_path:
            self._ensure_paths(ref_video_path)
        self._ensure_paths(audio_path)
        
        # 检查 generate_infinitetalk.py 是否存在
        script_path = self.infinitetalk_repo_dir / "generate_infinitetalk.py"
        if not script_path.exists():
            raise FileNotFoundError(
                f"找不到 generate_infinitetalk.py，请确保 InfiniteTalk 仓库已克隆到: {self.infinitetalk_repo_dir}\n"
                f"克隆命令: git clone https://github.com/bmwas/InfiniteTalk.git {self.infinitetalk_repo_dir}"
            )
        
        # 创建临时 JSON 配置文件
        with tempfile.TemporaryDirectory() as td:
            input_json_path = Path(td) / "input.json"
            cfg = self._build_input_json(
                image_path=image_path,
                audio_path=audio_path,
                ref_video_path=ref_video_path,
                fps=fps,
                seed=seed,
            )
            input_json_path.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), 
                encoding="utf-8"
            )
            
            # 构建命令
            cmd = [
                self.python_exec, str(script_path),
                "--ckpt_dir", str(self.ckpt_dir),
                "--wav2vec_dir", str(self.wav2vec_dir),
                "--infinitetalk_dir", str(self.infinitetalk_weight),
                "--input_json", str(input_json_path),
                "--size", size,
                "--sample_steps", str(sample_steps),
                "--mode", mode,
                "--motion_frame", str(motion_frame),
                "--save_file", output_stem,
            ]
            
            # 添加可选参数
            if num_persistent_param_in_dit is not None:
                cmd += ["--num_persistent_param_in_dit", str(num_persistent_param_in_dit)]
            if sample_text_guide_scale is not None:
                cmd += ["--sample_text_guide_scale", str(sample_text_guide_scale)]
            if sample_audio_guide_scale is not None:
                cmd += ["--sample_audio_guide_scale", str(sample_audio_guide_scale)]
            if lora_dir:
                cmd += ["--lora_dir", str(lora_dir)]
            if lora_scale is not None:
                cmd += ["--lora_scale", str(lora_scale)]
            if quant:
                cmd += ["--quant", quant]
            if quant_dir:
                cmd += ["--quant_dir", str(quant_dir)]
            
            print("执行 InfiniteTalk 生成命令:")
            print(" ".join(shlex.quote(str(x)) for x in cmd))
            print("-" * 50)
            
            # 执行命令
            try:
                subprocess.run(cmd, cwd=str(self.infinitetalk_repo_dir), check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"InfiniteTalk 生成失败: {e}")
        
        # 查找输出文件
        # 输出文件通常在 repo 目录下，文件名格式为 {output_stem}.mp4
        output_mp4 = self.infinitetalk_repo_dir / f"{output_stem}.mp4"
        if not output_mp4.exists():
            # 也可能在当前目录
            output_mp4 = Path(f"{output_stem}.mp4")
            if not output_mp4.exists():
                raise FileNotFoundError(
                    f"找不到输出文件: {output_stem}.mp4\n"
                    f"请检查 InfiniteTalk 脚本是否成功执行"
                )
        
        print(f"✓ 视频生成成功: {output_mp4}")
        return str(output_mp4)


def lipsync_with_infinitetalk(image_path, audio_path, output_path="output_lipsync.mp4",size="infinitetalk-480"):
                            #   fps=25,
                            #   sample_steps=35,
                            #   mode="clip",
                            #   motion_frame=9,
                            #   sample_text_guide_scale=5,
                            #   sample_audio_guide_scale=4,
                            #   num_persistent_param_in_dit=None):
    """
    使用 InfiniteTalk 进行唇形同步（实际 API 调用）
                
    """
    print("=" * 50)
    print("开始 InfiniteTalk 唇形同步处理...")
    print("=" * 50)
    
    # 创建 API 实例
    api = InfiniteTalkAPI(
        infinitetalk_repo_dir=infinitetalk_repo_dir_,
        ckpt_dir=ckpt_dir_,
        wav2vec_dir=wav2vec_dir_,
        infinitetalk_weight=infinitetalk_weight_,
    )
    
    # 生成视频
    output_stem = Path(output_path).stem
    video_path = api.generate(
        image_path=image_path,
        audio_path=audio_path,
        output_stem=output_stem,
        size="infinitetalk-480",
        sample_steps=35,
        mode="clip",
        motion_frame=9,
        fps=25,
        sample_text_guide_scale=5,
        sample_audio_guide_scale=4,
        num_persistent_param_in_dit=32,
    )
    
    # 如果输出文件不在预期位置，移动到目标位置
    video_path_obj = Path(video_path)
    target_path = Path(output_path)
    if video_path_obj != target_path and video_path_obj.exists():
        if target_path.exists():
            target_path.unlink()
        video_path_obj.rename(target_path)
        print(f"✓ 视频已移动到: {target_path}")
        return str(target_path)
    
    return video_path