from pathlib import Path
# 常量定义
audio_download_path = "outputs/audio"
video_download_path = "outputs/video"
image_upload_path = "inputs/image"
audio_upload_path = "inputs/audio"
video_upload_path = "inputs/video"
extract_audio_path = "extract/audio"

# 确保输出目录存在
def ensure_directories():
    Path(audio_download_path).mkdir(parents=True, exist_ok=True)
    Path(video_download_path).mkdir(parents=True, exist_ok=True)
    Path(image_upload_path).mkdir(parents=True, exist_ok=True)
    Path(audio_upload_path).mkdir(parents=True, exist_ok=True)
    Path(video_upload_path).mkdir(parents=True, exist_ok=True)
    Path(extract_audio_path).mkdir(parents=True, exist_ok=True)