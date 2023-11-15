## System Requirment
This example has following requirements
- Latest chatglm2-6b model from [hugging-face](https://hf-mirror.com/THUDM/chatglm2-6b) or [model scope](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary)
- Request Nvidia GPU for GPTQ quantization, GPU memory >= 12GB
- Verified system: Ubuntu 18.04, Nvidia 1080TI，Driver Version: 520.61.05, CUDA Version:11.8, GPU memory 24 GB


## Setup Environment
```bash
conda deactivate
conda create -n chatglm2-gptq python=3.10
conda activate chatglm2-gptq
python -m pip install -r requirements.txt
```

## Download ChatGLM2-6B FP16 original model 
```python
python download_chatglm2_6b.py
```

## Run GPTQ INT4 Quantization on Nvidia GPU
```python
python run_chatglm2_gptq.py --model_id ZhipuAI/chatglm2-6b --output_dir ChatGLM2-GPTQ-INT4
```

## Copy tokenzier related files from chatglm2-6b to ChatGLM2-GPTQ-INT4
```bash
cp ZhipuAI/chatglm2-6b/tokenizer.model ChatGLM2-GPTQ-INT4/
cp ZhipuAI/chatglm2-6b/tokenizer_config.json ChatGLM2-GPTQ-INT4/
cp ZhipuAI/chatglm2-6b/tokenization_chatglm.py ChatGLM2-GPTQ-INT4/
```

Please note, ChatGLM2 tokenzier has issue with tranformer 4.35.0 to save tokenizer via `tokenizer.save_pretrained(args.output_dir)`: [Issue 152](https://github.com/THUDM/ChatGLM3/issues/152), [Issue 199](https://github.com/InternLM/xtuner/issues/199)

Because GPTQ method only apply to pytorch model, tokenzier itself remain unchanged. So we manually copy tokenzier related files from original chatglm2-6b to quantized ChatGLM2-GPTQ-INT4


