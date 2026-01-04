import ffmpeg
import os
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
        print(f"FFmpeg执行完成（失败），错误码：{e.exit_code}")
        print(f"错误信息：{e.stderr.decode('utf-8')}")
        return None, False    

    