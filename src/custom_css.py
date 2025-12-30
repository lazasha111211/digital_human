# footer的显示

footer_hide = """
    footer.svelte-zxu34v {
        display: none !important;
        visibility: hidden !important;  /* 双重保障，避免残留占位 */
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
}
"""

# 浏览器title显示内容
browser_title = """
    function() {
        document.title = '智能工作台';
    }
"""

# ======================== 自定义样式（可选，增强视觉分隔） ========================
step_block = """
    .step-block {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        background-color: #fafafa;
    }
    
    """
gr_divider = """
    .gr-divider {
            margin: 10px 0 !important;
        }
    """    