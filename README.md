# ov_llm_bench
OpenVINO LLM Benchmark, including model conversion and benchmark script, required minimum openvino version>=2023.2


## Setup Environment
### Setup Python Environment
```bash
conda create -n ov_llm_bench python=3.10
conda activate ov_llm_bench
pip install -r requirements.txt
```
### Build OpenVINO 2023.2 github master on Linux:
```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino && git submodule update --init --recursive 
python -m pip install -U pip 
python -m pip install -r ./src/bindings/python/src/compatibility/openvino/requirements-dev.txt
python -m pip install -r ./src/bindings/python/wheel/requirements-dev.txt
python -m pip install -r ./src/bindings/python/requirements.txt
mkdir build && cd build
cmake -DENABLE_INTEL_GNA=OFF -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<ov install dir> ..
make --jobs=$(nproc --all)
make install
cd <ov install dir>/tools/
python -m pip install  openvino*.whl
```

### Build OpenVINO 2023.2 github master on Windows:
```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino && git submodule update --init --recursive
python -m pip install -U pip
python -m pip install -r ./src/bindings/python/src/compatibility/openvino/requirements-dev.txt
python -m pip install -r ./src/bindings/python/wheel/requirements-dev.txt
python -m pip install -r ./src/bindings/python/requirements.txt
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -DENABLE_INTEL_GNA=OFF -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<ov install dir> ..
cmake --build . --config Release --verbose -j8
cmake --install .
cd <ov install dir>/tools/
python -m pip install openvino*.whl
```

## Llama-2-7B-Chat-GPTQ
### Download Llama-2-7B-Chat-GPTQ Pytorch INT4-FP16 Model locally via HF mirror in PRC:
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download TheBloke/Llama-2-7B-Chat-GPTQ --local-dir Llama-2-7B-Chat-GPTQ
```

### Convert Llama-2-7B-Chat-GPTQ Pytorch Model to OpenVINO INT4-FP16 model
```python
python convert.py --model_id Llama-2-7B-Chat-GPTQ/ --output_dir Llama-2-7B-Chat-GPTQ-OV --precision FP16
```

### Run benchmark with OpenVINO INT4-FP16 model with prompt length 9/32/256/512/1024 on intel CPU/GPU
```python
python benchmark.py -m Llama-2-7B-Chat-GPTQ-OV/GPTQ_INT4-FP16 -d CPU -pl 9
```

## ChatGLM3-6B-GPTQ-INT4
### Download ChatGLM3-6B-GPTQ-INT4 Pytorch INT4-FP16 Model locally via HF mirror in PRC:
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download ranchlai/chatglm3-6B-gptq-4bit --local-dir ChatGLM3-6B-GPTQ-INT4
```

### Convert ChatGLM3-6B-GPTQ-INT4 Pytorch Model to OpenVINO INT4-FP16 model
```python
python convert.py --model_id ChatGLM3-6B-GPTQ-INT4 --output_dir ChatGLM3-6B-GPTQ-INT4-OV --precision FP16
```
### Run benchmark with OpenVINO INT4-FP16 model with prompt length 9/32/256/512/1024 on intel CPU/GPU
```python
python benchmark.py -m ChatGLM3-6B-GPTQ-INT4-OV/GPTQ_INT4-FP16 -d CPU -pl 9
```

## Qwen-7B-Chat-Int4
### Download Qwen-7B-Chat-Int4 Pytorch INT4-FP16 locally via HF mirror in PRC:
```
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Qwen/Qwen-7B-Chat-Int4 --local-dir Qwen-7B-Chat-Int4
```
### Modify Qwen-7B-Chat-Int4/modeling_qwen.py to enable export on CPU only mode
```python
#SUPPORT_CUDA = torch.cuda.is_available()
SUPPORT_CUDA = False
#SUPPORT_TORCH2 = hasattr(torch, '__version__') and int(torch.__version__.split(".")[0]) >= 2
SUPPORT_TORCH2=False
```
### Convert Qwen-7B-Chat-Int4 Pytorch Model to OpenVINO INT4-FP16 model
```python
python convert.py --model_id Qwen-7B-Chat-Int4 --output_dir Qwen-7B-Chat-Int4-OV --precision FP16 
```

### Run benchmark with OpenVINO INT4-FP16 model with prompt length 9/32/256/512/1024 on intel CPU/GPU
```python
python benchmark.py -m Qwen-7B-Chat-Int4-OV/GPTQ_INT4-FP16 -d CPU -pl 9
```
