import ffmpeg
import os
import cv2
from pathlib import Path
from utils import generate_random_filename
from constants import ensure_directories



# 从视频文件中提取音频
def extract_audio_from_video(
    video_path: str,
    output_audio_path: str,
    target_sr: int = 16000,  # Whisper 要求 16kHz
    mono: bool = True,       # 转为单声道
    audio_format: str = "wav"  # 输出格式：wav/mp3/aac
):
    """
    从视频中提取音频，并转为 Whisper 适配格式
    :param video_path: 输入视频路径
    :param output_audio_path: 输出音频路径
    :param target_sr: 目标采样率
    :param mono: 是否转为单声道
    :param audio_format: 输出音频格式
    """
    # 校验视频文件是否存在，可以注释掉，因为上一层调用函数已经判断
    # if not os.path.exists(video_path):
    #     raise FileNotFoundError(f"视频文件不存在：{video_path}")
    
    # 构建 FFmpeg 命令
    stream = ffmpeg.input(video_path)
    # 只提取音频流
    stream = stream.audio     
    
    # 音频参数配置（适配 Whisper）
    if mono:
        stream = stream.filter("channelsplit", channel_layout="mono")
    
    # 重采样
    stream = stream.filter("aresample", target_sr)  
    
    output_audio_dir = os.path.join(os.getcwd(), output_audio_path)
    if not os.path.exists(output_audio_dir):
        ensure_directories()

    file_name = output_audio_dir + "/" + generate_random_filename("wav", "audio", True)    
    
    # 编码器配置
    if audio_format == "wav":
        stream = stream.output(file_name, acodec="pcm_s16le")
    elif audio_format == "mp3":
        stream = stream.output(file_name, acodec="libmp3lame", qscale_a=2)
    elif audio_format == "aac":
        stream = stream.output(file_name, acodec="copy")
    
    # 执行命令（静默模式，不输出日志）
    try: 
        ffmpeg.run(stream, overwrite_output=True, quiet=False)
        print(f"✅ 音频提取完成：{file_name}")
        return file_name, True
    except ffmpeg.Error as e:
        # 4. 执行到这里 = 进程已结束但执行失败
        print(f"FFmpeg执行完成(失败），错误码：{e.exit_code}")
        print(f"错误信息：{e.stderr.decode('utf-8')}")
        return None, False    

    

    """
    保存视频并合并音频
    Args:
        video_frames: 视频帧列表(numpy数组,形状为[H, W, 3])
        audio_path: 音频文件路径
        output_path: 输出视频路径
        fps: 视频帧率
    """
    print(f"保存视频到: {output_path}")
    
    if not video_frames:
        print("错误: 没有视频帧")
        return
    
    # 获取视频尺寸
    height, width = video_frames[0].shape[:2]
    print(f"视频尺寸: {width}x{height}, 帧数: {len(video_frames)}")
    
    # 使用 OpenCV 写入视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = output_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    for frame in video_frames:
        # 转换为 BGR 格式（OpenCV使用）
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"视频已保存到: {temp_video_path}")

    # 使用 ffmpeg 合并音频（需要安装 ffmpeg）
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path,
            '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
            '-pix_fmt', 'yuv420p', '-shortest', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"最终视频已保存到: {output_path}")
        # 删除临时文件
        Path(temp_video_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"音频合并失败: {e}")
        print(f"请手动使用 ffmpeg 合并音频，或使用原始视频: {temp_video_path}")
        # 如果合并失败，将临时文件重命名
        if Path(temp_video_path).exists():
            Path(temp_video_path).rename(output_path)       