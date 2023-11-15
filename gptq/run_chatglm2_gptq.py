from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import argparse

parser = argparse.ArgumentParser(
        'LLM GPTQ INT4 quantization tool', add_help=True, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
        '-m', '--model_id', default="ZhipuAI/chatglm2-6b", help='model folder to original ChatGLM FP16 pytorch model', required=TabError)
parser.add_argument(
        '-o', '--output_dir', default="ChatGLM2-GPTQ-INT4", help='model folder to save quantized GPTQ INT4 pytorch model', required=TabError)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
quantize_config = GPTQConfig(
    bits=4,
    dataset=[
        "新風系統是透過系統設計送風和排風使室內空氣存在一定的壓差",
        "向室內提供足夠的新風並排出室內汙濁空氣 ",
        "無需開窗全天持續不斷有組織的向室內引入新風",
        "為室內人員呼吸代謝提供所需氧氣",
        "使用超过2.4万亿tokens的数据进行预训练, 包含高质量中、英、多语言、代码、数学等数据，涵盖通用及专业领域的训练语料。通过大量对比实验对预训练语料分布进行了优化"
        "相比目前以中英词表为主的开源模型, Qwen-7B使用了约15万大小的词表。该词表对多语言更加友好, 方便用户在不扩展词表的情况下对部分语种进行能力增强和扩展。",
    ],
    tokenizer=tokenizer,
    block_name_to_quantize="transformer.encoder.layers",
    cache_block_outputs=False
)

model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=quantize_config, device_map="auto", trust_remote_code=True
)
model.save_pretrained(args.output_dir)
#tokenizer.save_pretrained(args.output_dir)
