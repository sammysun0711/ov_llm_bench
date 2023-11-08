import openvino.runtime as ov 
from openvino.tools.mo import convert_model
from openvino._offline_transformations import apply_moc_transformations, compress_model_transformation

fp32_model_path = "llm_models/Qwen-7B-Chat-GPTQ_INT4_FP16/openvino_model.xml"
fp16_model_path = "llm_models/Qwen-7B-Chat-GPTQ_INT4_FP16/openvino_model_fp16.xml"

core = ov.Core()
print("Read FP32 OV Model ...")
ov_model = core.read_model(fp32_model_path)

print("Convert FP32 OV Model to FP16 OV Model...")
apply_moc_transformations(ov_model, cf=False)
compress_model_transformation(ov_model)

print(f"Serialize Converted FP16 Model as {fp16_model_path}")
ov.serialize(ov_model, fp16_model_path)

print("Done.")
