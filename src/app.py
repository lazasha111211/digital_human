# app.py（修改布局后）
import gradio as gr

from model_api import *
from custom_css import footer_hide, gr_divider
from constants import *
from utils import *


def create_interface():
    with gr.Blocks(title="文生视频工具") as app:
        gr.Markdown("# 🎬 文生视频一站式工具")
        
        gr.HTML(f"""
             <style>
                {footer_hide}
                {gr_divider}
                
        
            </style>
        """)
        gr.HTML("""
            <style>
                span.md.svelte-1hf8a14.prose h1 {
                    color: white; /* 可选：如果h1文字也需要白色 */
                    text-align: center; /* 核心：文字水平居中 */
                }

                
                span.md.svelte-1hf8a14.prose h2 {
                    color: white; /* 可选：如果h1文字也需要白色 */
                    
                }
                .svelte-16ln60g {
                    background-color: blue; /* 浏览器内置的浅蓝命名色 */
                    
                }
                .spaced-row {
                    margin: 0.1rem 0 !important; /* 上下间距1rem，左右0 */
                }

                div.svelte-16ln60g > div[data-testid="markdown"].prose.svelte-1xjkzpp > span.md.svelte-1hf8a14.prose > h1 font {
                    text-align: center !important;
                    color: white !important;
                }    

                div.svelte-16ln60g > div[data-testid="markdown"].prose.svelte-1xjkzpp > span.md.svelte-1hf8a14.prose > h2 font {
                    text-align: center !important;
                    color: white !important;
      
                }
               
            </style>
        """)
        
        # 状态变量存储中间结果
        original_video_path = gr.State(None)
        tts_audio_path = gr.State(None)
        
        # 第一排：前两个功能
        with gr.Row():
            # 步骤1: 音频转文本
            with gr.Group():
                gr.Markdown("## 🎤 1: 视频读取文案")
                with gr.Row():
                    video_input = gr.Video(label="上传视频（支持avi、mov、mp4）",
                                            sources=["upload"],
                                            width=600,  # 固定宽度（像素）
                                            height=395, # 固定高度（像素）
                                            format="mp4",
                                            include_audio = True)  
                with gr.Row():
                    video_to_text_btn = gr.Button("读取文案", variant="primary")
            
            # 步骤2: 文本增强
            with gr.Group():
                gr.Markdown("## ✏️ 2: 文案增强")
                with gr.Row(elem_classes="spaced-row"):
                    video_text_output = gr.Textbox(label="视频读取文案结果", lines=10)
                with gr.Row():
                    description_input = gr.Textbox(label="输入加工要求", placeholder="请输入对文案的加工要求...", lines=3)
                with gr.Row():
                    enhance_text_btn = gr.Button("增强文案", variant="primary")
        
        # 第二排：后两个功能
        with gr.Row():
            # 步骤3: 生成配音
            with gr.Group():
                gr.Markdown("## 🎧 3: 生成配音")
                with gr.Row():
                    ref_audio_input = gr.Audio(type="filepath", label="上传参考音频")
                with gr.Row():
                    tts_btn = gr.Button("生成配音", variant="primary")
                tts_audio_output = gr.Audio(label="生成的配音")
                
            # 步骤4: 生成视频
            with gr.Group():
                gr.Markdown("## 🎥 4: 生成视频")
                with gr.Row():
                    image_input = gr.Image(type="filepath", label="上传人物正脸图片")
                with gr.Row():
                    generate_video_btn = gr.Button("生成视频", variant="primary")
                video_output = gr.Video(label="生成的视频")

        # 设置事件处理（保持不变）
        video_to_text_btn.click(
            fn=process_video_to_text,
            inputs=[video_input],
            outputs=[video_text_output, original_video_path]
        )
        
        enhance_text_btn.click(
            fn=process_text_enhancement,
            inputs=[video_text_output, description_input],
            outputs=[video_text_output]
        )
        
        tts_btn.click(
            fn=process_tts,
            inputs=[video_text_output, ref_audio_input],
            outputs=[tts_audio_output, tts_audio_path]
        )
        
        generate_video_btn.click(
            fn=process_video_generation,
            inputs=[image_input, tts_audio_path],
            outputs=[video_output]
        )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch()