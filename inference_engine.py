import time
from transformers import AutoTokenizer, TextIteratorStreamer, AutoConfig, GenerationConfig
import gc
from optimum.intel.openvino import OVModelForCausalLM
from threading import Thread, Event
from time import perf_counter
from typing import List
import numpy as np
from modeling import OVQwenModel, OVChatGLM2Model
from utils import print_perf_counters_sort

"""
from utils import MemConsumption
mem_consumption = MemConsumption()
max_rss_mem_consumption = ''
max_shared_mem_consumption = ''
"""


class InferenceEngine:
    def __init__(self, args=None, ov_config=None):
        self.args = args

        self.config = AutoConfig.from_pretrained(
                self.args.model_id, trust_remote_code=True)
        s = time.time()
        if self.config.model_type == "llama":
            print("Loading Llama2 model")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_id, trust_remote_code=True)

            self.ov_model = OVModelForCausalLM.from_pretrained(self.args.model_id,
                                                               compile=False,
                                                               device=self.args.device,
                                                               ov_config=ov_config,
                                                               trust_remote_code=True)

        elif self.config.model_type == "chatglm":
            print("Loading ChatGLM2 model")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_id, trust_remote_code=True)

            self.ov_model = OVChatGLM2Model.from_pretrained(self.args.model_id,
                                                            config=self.config,
                                                            compile=False,
                                                            device=self.args.device,
                                                            ov_config=ov_config,
                                                            trust_remote_code=True)

        elif self.config.model_type == "qwen":
            print("Loading Qwen model")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_id,
                pad_token='<|extra_0|>',
                eos_token='<|endoftext|>',
                padding_side='left',
                trust_remote_code=True)

            self.ov_model = OVQwenModel.from_pretrained(self.args.model_id,
                                                        config=self.config,
                                                        compile=False,
                                                        device=self.args.device,
                                                        ov_config=ov_config,
                                                        pad_token_id=self.tokenizer.pad_token_id,
                                                        trust_remote_code=True)
            self.ov_model.generation_config = GenerationConfig.from_pretrained(self.args.model_id, pad_token_id=self.tokenizer.pad_token_id)

        print("read model time: {:.3f} s".format(time.time() - s))

        s = time.time()
        self.ov_model.compile()
        print("compile model time: {:.3f} s".format(time.time() - s))

        print("intial model successed")
        gc.collect()

    def chat_stream(self, text):
        print("text: ", text)
        if self.args.use_prompt_template: text = self.build_inputs(text)
        prompt = text
        self.model_inputs = self.tokenizer(prompt, return_tensors="pt")
        self.model_inputs.pop("token_type_ids", None)
        streamer = TextIteratorStreamer(self.tokenizer,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        generate_kwargs = dict(self.model_inputs,
                               streamer=streamer,
                               max_new_tokens=self.args.max_new_tokens,
                               do_sample=self.args.do_sample,
                               top_p=self.args.top_p,
                               temperature=self.args.temperature,
                               top_k=self.args.top_k,
                               repetition_penalty=self.args.repetition_penalty,
                               eos_token_id=self.tokenizer.eos_token_id)
        stream_complete = Event()

        def generate_and_signal_complete():
            """
            genration function for single thread
            """
            self.ov_model.generate(**generate_kwargs)
            stream_complete.set()

        # t = Thread(target=self.ov_model.generate, kwargs=generate_kwargs)
        t = Thread(target=generate_and_signal_complete)
        t.start()

        # Pull the generated text from the streamer, and update the model output.
        model_output = ""
        perf_text = ""
        per_token_time = []
        tokens_per_second = []

        num_tokens = 0
        start = perf_counter()

        for new_text in streamer:
            if len(new_text) == 0:
                continue
            current_time = perf_counter() - start
            if num_tokens == 0:
                print(f"first token time: {current_time:.3f} s")
            perf_text, num_tokens, current_tokens_per_second = self.estimate_latency(current_time, perf_text,
                                                                                     new_text, per_token_time,
                                                                                     num_tokens)
            if current_tokens_per_second is not None: tokens_per_second.append(current_tokens_per_second)
            yield new_text
            start = perf_counter()

        self.generate_num_tokens = num_tokens
        print("Skip last average token per second to avoid outlier")
        self.average_tokens_per_second = np.mean(tokens_per_second[:-1])
        gc.collect()
        # torch.cuda.empty_cache()

    def generate(self, text):
        '''
        if self.args.enable_memory_profiling:
            mem_consumption.start_collect_memory_consumption()
        '''
        out = ""
        for output in self.chat_stream(text):
            out += output
        print(f"Input num tokens: {len(self.model_inputs.input_ids[0])}, generated num tokens: {self.generate_num_tokens}")
        print(f"Total average generaton speed: {self.average_tokens_per_second:.3f} tokens/s")      
        print(f"Generated response: {out}")
        # yield output
        '''
        if self.args.enable_memory_profiling:
            mem_consumption.end_collect_momory_consumption()
            max_rss_mem_consumption, max_shared_mem_consumption = mem_consumption.get_max_memory_consumption()
            mem_consumption.clear_max_memory_consumption()
            print("Max RSS memory consumption: ", max_rss_mem_consumption)
            print("Max Shared memory consumption: ", max_shared_mem_consumption)
        '''

    def build_inputs(self, query):
        prompt = None

        if self.config.model_type == "llama":
            system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{query} [/INST]"

        elif self.config.model_type == "chatglm":
            if "chatglm2" in self.args.model_id.lower():
                prompt = f"[Round 0]\n\n问：{query}\n\n答：\n\n"
            elif "chatglm3" in self.args.model_id.lower():
                system_message = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."
                prompt = f"<|system|>\n{system_message}\n<|user|>{query}<|assistant|>"

        elif self.config.model_type == "qwen":
            im_start = "<|im_start|>"
            im_end = "<|im_end|>"
            system_message = "You are a helpful assistant."
            query = query.lstrip("\n").rstrip()
            prompt = f"{im_start}system\n{system_message}{im_end}"
            prompt += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        return prompt
    
    def estimate_latency(self, current_time: float, current_perf_text: str,
                         new_gen_text: str, per_token_time: List[float],
                         num_tokens: int):
        """
        Helper function for performance estimation

        Parameters:
          current_time (float): This step time in seconds.
          current_perf_text (str): Current content of performance UI field.
          new_gen_text (str): New generated text.
          per_token_time (List[float]): history of performance from previous steps.
          num_tokens (int): Total number of generated tokens.

        Returns:
          update for performance text field
          update for a total number of tokens
        """
        current_tokens = self.tokenizer.encode(
            '$' + new_gen_text, add_special_tokens=False)
        special_token = self.tokenizer.encode('$', add_special_tokens=False)
        if current_tokens[0] == special_token[0]:
            current_tokens = current_tokens[1:]

        num_current_toks = len(current_tokens)

        num_tokens += num_current_toks
        per_token_time.append(num_current_toks / current_time)
        if len(per_token_time) > 10 and len(per_token_time) % 4 == 0:
            current_bucket = per_token_time[-10:]
            tokens_per_second = np.mean(current_bucket)
            current_perf_text = f"Average generaton speed: {tokens_per_second:.2f} tokens/s. Total generated tokens: {num_tokens}"
            print(current_perf_text)
            return current_perf_text, num_tokens, tokens_per_second
        return current_perf_text, num_tokens, None

    def get_profiling_data(self):
        perfs_count_list = self.ov_model.request.profiling_info
        total_sorted_list = print_perf_counters_sort(
            [perfs_count_list], sort_flag="simple_sort")

        return total_sorted_list
