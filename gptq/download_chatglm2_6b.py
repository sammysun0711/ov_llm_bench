from modelscope import AutoTokenizer, AutoModelForCausalLM, snapshot_download
model_dir = snapshot_download("ZhipuAI/chatglm2-6b", revision = 'v1.0.12', cache_dir=".")
