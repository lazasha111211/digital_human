import whisper
import jieba
import pangu
import re
from zhon.hanzi import punctuation  # 中文标点库

def semantic_sentence_segmentation(text: str) -> str:
    """
    基于语义的中文断句（核心逻辑）
    :param text: Whisper识别的无间隔文字
    :return: 按语义断句后的文本
    """
    # 步骤1：Jieba 语义分词（获取词边界）
    words = jieba.lcut(text)  # 精确分词，保留语义完整
    
    # 步骤2：定义语义断句触发词（根据中文句法规则）
    # 核心：遇到「主语/连词/语气词」或「动作转折」时，判定为句子边界
    break_triggers = {
        # 主语类：我/你/他/我们/他们/这/那/此
        "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那", "此", "该",
        # 连词类：但/却/而/且/或/于是/因此/另外
        "但", "却", "而", "且", "或", "于是", "因此", "另外", "同时", "此外",
        # 语气/时态类：了/呢/吧/吗/也/就/才/又/还
        "了", "呢", "吧", "吗", "也", "就", "才", "又", "还", "刚", "已", "将",
        # 动作类：想/要/会/能/应/该/可
        "想", "要", "会", "能", "应", "该", "可", "愿意", "打算"
    }

    # 步骤3：遍历分词结果，按语义拼接+断句
    segmented = []
    current_sentence = []
    for idx, word in enumerate(words):
        current_sentence.append(word)
        
        # 触发条件1：当前词是断句触发词，且不是句子开头 → 断句
        if word in break_triggers and idx > 0 and len(current_sentence) > 3:
            # 拼接当前句 + 补全标点 + 换行
            sentence = "".join(current_sentence[:-1]).strip()
            if sentence:
                segmented.append(f"{sentence}，")  # 动作/连词后加逗号
            current_sentence = [word]  # 重置当前句，从触发词开始
        
        # 触发条件2：遇到句末标点（若有）→ 直接断句
        if re.search(r"[。！？；]", word):
            sentence = "".join(current_sentence).strip()
            if sentence:
                segmented.append(f"{sentence}\n")  # 句末加换行
            current_sentence = []

        # 处理最后一句
    if current_sentence:
        last_sentence = "".join(current_sentence).strip()
        if last_sentence:
            segmented.append(f"{last_sentence}。")
    
    # 步骤4：拼接结果 + 补全中文空格
    final_text = "".join(segmented)
    final_text = pangu.spacing_text(final_text)  # 标点与文字分离
    # 清理多余标点/换行
    final_text = re.sub(r"，+", "，", final_text)
    final_text = re.sub(r"\n+", "\n", final_text).strip()
    
    return final_text    

def transcribe(model_dir: str,
                audio_dir: str, 
                language: str="Chinese", 
                initial_prompt: str="请用简体中文转写，保持语义完整") -> str:

    # 加载模型（首次运行自动下载）
    model = whisper.load_model(model_dir)  # 可选：tiny/base/medium/large

    # 识别音频（自动处理采样率/声道）
    result = model.transcribe(audio_dir, 
                        language="Chinese", 
                        fp16=False, 
                        temperature=0.0,
                        initial_prompt="请用简体中文转写，保持语义完整")  # 提示模型输出标点)
    raw_text = result["text"].strip()
    segmented_text = semantic_sentence_segmentation(raw_text)
    return segmented_text