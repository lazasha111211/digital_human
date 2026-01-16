import torch
import os
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig 
from ai_utils import get_accelerator_device, get_model_kwargs # noqa: F403
# 彻底屏蔽无关警告（如参数提示、设备适配）
warnings.filterwarnings("ignore")
# ===================== 全局配置（可根据需求调整） =====================
# 模型名称/路径
MODEL_NAME = "./checkpoints/Qwen2.5-1.5B-Instruct"

# 最大生成长度
# MAX_NEW_TOKENS = 512
# 推理温度
# TEMPERATURE = 0.7

# 装载模型
def load_qwen25_15b(model_dir: str = "./checkpoints/Qwen2.5-1.5B-Instruct"):
    """
    加载 Qwen2.5-1.5B-Instruct:
    1. 加载本地模型
    2. 通配MPS(macOS)、CPU、GPU(英伟达 CUDA)
    3. 如果需要设置全局缓存变量，可以加快后面操作速度
    
    """
    local_model_dir = os.path.join(os.getcwd(), model_dir)
    print(f"model dir : {local_model_dir}")
  
    try:
        print("📥 加载 Qwen2.5-1.5B-Instruct ...")
            
        # 1. 加载分词器（Qwen 专属，需 trust_remote_code）
        TOKENIZER = AutoTokenizer.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            clean_up_tokenization_spaces=True
        )

        # 2. 核心配置：适配全环境
        device = get_accelerator_device()
        
        
        print(f"当前加速设备：{device.type}")
        model_kwargs = get_model_kwargs(device.type)  
        

        # 3. 加载模型（自加载本地）
        MODEL = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            **model_kwargs
        ).eval()  # 推理模式，禁用梯度
        
        MODEL = MODEL.to(device)
        
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{str(e)}")

    return TOKENIZER, MODEL

# 调用模型生成文字
def qwen_generate(
    raw_text: str,
    requirements: str,
    model_dir: str = "./checkpoints/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 512,
    temperature: float = 0.3
) -> str:
    """
    调用 Qwen2.5-1.5B-Instruct 生成文本：
    :param raw_text: 原始文字（不能为空）
    :param requirements: 对文字的修改/生成要求（不能为空）
    :param model_dir: 模型路径（本地/ Hugging Face 远程）,本地相对路径      
    :param max_new_tokens: 最大生成长度
    :param temperature: 生成随机性(0-1,越小越稳定）
    :return: 符合要求的最终文本
    """

    # 加载模型
    tokenizer, model = load_qwen25_15b(model_dir)

    # 构造 Qwen 专属 Prompt 格式（关键：适配模型指令理解逻辑）
    messages = [
        {"role": "system", "content": "你是一个专业的文本处理助手，严格按照用户要求处理文字，直接输出最终结果，不添加任何额外解释、标题或格式。"},
        {"role": "user", "content": f"原始文字：{raw_text}\n要求：{requirements}"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 配置生成参数（通过 GenerationConfig 确保参数生效）
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=False,  # 确定性生成，严格按要求输出
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # 编码输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    )

    device = get_accelerator_device()
   
    
    # 张量迁移到目标设备（兼容所有设备类型）：实现硬件加速、保证计算一致性
    # 模型在 GPU 上，但输入张量在 CPU 上，直接推理会抛出 RuntimeError
    # 提升鲁棒性（Robustness）： 指程序 / 系统在面对「异常输入、硬件 / 环境波动、边界场景」时，
    # 仍能保持稳定运行、不崩溃，且能合理处理错误的能力。简单来说：鲁棒的代码不怕 “意外”，能扛住各种 “非正常情况”。
    try:
        inputs = inputs.to(device)
    except RuntimeError as e:
        if "MPS" in str(e):
            warnings.warn(f"MPS 迁移失败,降级到CPU:{e}")
            device = torch.device("cpu")
            inputs = inputs.to(device)
        else:
            raise

    # 生成文本（无梯度，节省内存）
    # torch.no_grad() 是 PyTorch 的上下文管理器，进入该上下文后：
    #    禁用所有张量的梯度计算：模型前向传播时，不再记录梯度（requires_grad=False）；
    #    节省显存 / 内存：梯度信息会占用大量显存（尤其是大模型），禁用后可减少 30%+ 显存占用；
    #    提升推理速度：无需计算 / 存储梯度，前向计算效率更高。
    #    大模型生成（推理）是「只读」过程，不需要反向传播更新参数，梯度计算完全是冗余开销

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_config
        )

    # 解码并清理结果（仅保留生成的内容，剔除 Prompt）
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_result = output_text.replace(prompt, "").strip()

    
    # 清理空行和冗余内容
    #next_text = "\n".join([line.strip() for line in final_result.split("\n") if line.strip()])
    
    
    final_result = final_result.split("\nassistant\n")[1]
    
  
    
    return final_result