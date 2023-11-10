# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import sys
import gc
import time
import logging as log
from argparse import ArgumentParser
from enum import Enum
from functools import wraps
from pathlib import Path
import types
from typing import Tuple, Dict, Optional
import torch
from nncf import compress_weights
from openvino import Type, PartialShape, save_model, convert_model
from openvino.runtime import Core
from optimum.exporters import TasksManager
from optimum.exporters.tasks import make_backend_config_constructor_for_task
from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import (
    NormalizedTextConfig, NormalizedConfigManager, DEFAULT_DUMMY_SHAPES,
    DummyPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyInputGenerator
)
from optimum.exporters.openvino import export_models
from optimum.intel.utils.modeling_utils import _prepare_decoder_attention_mask
from optimum.intel.openvino import (
    OVModelForCausalLM,
    OVQuantizer)

from optimum.exporters.onnx import __main__ as optimum_main
try:
    from optimum.exporters.openvino.__main__ import _get_submodels_and_export_configs
except ImportError:
    from optimum.exporters.onnx.__main__ import _get_submodels_and_onnx_configs as _get_submodels_and_export_configs

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel, PreTrainedModel

class BackendType(Enum):
    PYTORCH = 'pytorch'
    OPENVINO = 'openvino'

def save_tokenizer(tokenizer, out_dir):
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        log.error(f'tokenizer loading failed with {e}')

def compress_ov_model_weights_helper(ov_model, tok, config, out_path, fp16=False):
    compressed_ov_model = compress_weights(ov_model)
    save_ov_model_helper(compressed_ov_model, out_path, fp16=fp16, tok=tok, config=config)


def save_ov_model_helper(ov_model, out_path, model_name='openvino_model', fp16=False, tok=None, config=None):
    save_model(ov_model, Path(out_path) / f'{model_name}.xml', compress_to_fp16=fp16)
    if tok is not None:
        save_tokenizer(tok, out_path)
    if config is not None:
        config.save_pretrained(out_path)


def is_gptq(config):
    config_dict = config.to_dict()
    quantization_config = config_dict.get("quantization_config", None)
    return quantization_config and quantization_config["quant_method"] == "gptq"


def patch_gptq(config):
    do_gptq_patching = False
    config_dict = config.to_dict()
    quantization_config = config_dict.get("quantization_config", None)
    do_gptq_patching = quantization_config and quantization_config["quant_method"] == "gptq"
    orig_cuda_check = torch.cuda.is_available
    orig_post_init_model = None
    if do_gptq_patching:
        torch.set_default_dtype(torch.float32)
        torch.cuda.is_available = lambda: True

        from optimum.gptq import GPTQQuantizer

        orig_post_init_model = GPTQQuantizer.post_init_model

        def post_init_model(self, model):
            from auto_gptq import exllama_set_max_input_length

            class StoreAttr(object):
                pass

            model.quantize_config = StoreAttr()
            model.quantize_config.desc_act = self.desc_act
            if self.desc_act and not self.disable_exllama and self.max_input_length is not None:
                model = exllama_set_max_input_length(model, self.max_input_length)
            return model

        GPTQQuantizer.post_init_model = post_init_model
    return orig_cuda_check, orig_post_init_model


def unpatch_gptq(orig_cuda_check, orig_post_init_model):
    from optimum.gptq import GPTQQuantizer
    torch.cuda.is_available = orig_cuda_check
    GPTQQuantizer.post_init_model = orig_post_init_model


@torch.jit.script_if_tracing
def _chatglm2_get_context_layer(query_layer: torch.Tensor, key_layer: torch.Tensor, value_layer: torch.Tensor):
    mask = torch.zeros((query_layer.shape[-2], key_layer.shape[-2]), dtype=query_layer.dtype)
    if query_layer.shape[2] == key_layer.shape[2]:
        tmp_mask = torch.ones((query_layer.shape[-2], key_layer.shape[-2]), dtype=torch.bool).triu(diagonal=1)
        mask.masked_fill_(tmp_mask, float("-inf"))

    context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=mask)
    return context_layer


def _core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
    if attention_mask is None:
        context_layer = _chatglm2_get_context_layer(query_layer, key_layer, value_layer)
    else:
        attention_mask = ~attention_mask
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attention_mask
        )
    context_layer = context_layer.permute(2, 0, 1, 3)
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.reshape(*new_context_layer_shape)

    return context_layer


def _patch_chatglm_core_attention_forward(model: "PreTrainedModel"):
    for block in model.transformer.encoder.layers:
        block.self_attention.core_attention.forward = types.MethodType(
            _core_attention_forward, block.self_attention.core_attention
        )


def _update_qwen_rotary_embedding_cache(model):
    #model.transformer.rotary_emb(model.config.seq_length)
    model.transformer.rotary_emb(2048) # WA, patch Qwen sequence length from 8K to 2K to save system compute memory


def patch_model_for_optimum_export(model):
    if model.config.model_type == "chatglm":
        _patch_chatglm_core_attention_forward(model)
    elif model.config.model_type == "qwen":
        _update_qwen_rotary_embedding_cache(model)
    return model


def convert_optimum_causallm_base(model, args):
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = patch_model_for_optimum_export(model)
    model_config = model.config
    gptq_applied = is_gptq(model_config)
    precision = args.precision if not gptq_applied else f"GPTQ_INT4-{args.precision}"
    if gptq_applied and args.compress_weights:
        log.info("Weights compression will be skipped for GPTQ models")
    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    if args.save_orig:
        pt_out_dir = Path(args.output_dir) / 'pytorch'
        model.save_pretrained(pt_out_dir)
        save_tokenizer(tok, pt_out_dir)
    dummy_shapes = DEFAULT_DUMMY_SHAPES
    onnx_config, models_and_onnx_configs = _get_submodels_and_export_configs(
        model=model,
        task="text-generation-with-past",
        custom_onnx_configs={},
        custom_architecture=None,
        fn_get_submodels=None,
        preprocessors=None,
        _variant="default",
        monolith=False
    )
    if "decoder_with_past_model" in models_and_onnx_configs:
        models_and_onnx_configs = {"model": models_and_onnx_configs["decoder_with_past_model"]}
    ov_out_dir = Path(args.output_dir) / precision
    model.config.save_pretrained(ov_out_dir)
    files_subpaths = ["openvino_" + model_name + ".xml" for model_name in models_and_onnx_configs.keys()]
    export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        output_dir=ov_out_dir,
        output_names=files_subpaths,
        input_shapes=dummy_shapes,
        device="cpu",
        fp16=args.precision == "FP16",
        int8=False,
        model_kwargs={},
    )
    save_tokenizer(tok, ov_out_dir)
    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends and not gptq_applied:
        ov_int8_dir = Path(args.output_dir) / 'compressed_weights' / f'OV_{args.precision}-INT8'
        model.config.save_pretrained(ov_int8_dir)
        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            output_dir=ov_int8_dir,
            output_names=files_subpaths,
            input_shapes=dummy_shapes,
            device="cpu",
            fp16=args.precision == "FP16",
            int8=True,
            model_kwargs={},
        )
        save_tokenizer(tok, ov_int8_dir)
    if pt_compress_weights and not gptq_applied:
        compressed_model = compress_weights(model)
        onnx_config, models_and_onnx_configs = _get_submodels_and_export_configs(
            model=compressed_model,
            task="text-generation-with-past",
            custom_onnx_configs={},
            custom_architecture=None,
            fn_get_submodels=None,
            preprocessors=None,
            _variant="default",
            monolith=False
        )
        pt_out_dir = Path(args.output_dir) / 'compressed_weights' / f'PT_{args.precision}-INT8'
        model.config.save_pretrained(pt_out_dir)
        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            output_dir=pt_out_dir,
            output_names=files_subpaths,
            input_shapes=dummy_shapes,
            device="cpu",
            fp16=args.precision == "FP16",
            int8=False,
            model_kwargs={},
        )
        save_tokenizer(tok, pt_out_dir)
    return


def convert_causal_lm(args):
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    gptq_applied = is_gptq(model_config)
    precision = args.precision if not gptq_applied else f"GPTQ_INT4-{args.precision}"
    if gptq_applied and args.compress_weights:
        log.info("Weights compression will be skipped for GPTQ models")

    start = time.perf_counter()
    if args.save_orig or pt_compress_weights:
        pt_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
        )
        if args.save_orig:
            pt_out_dir = Path(args.output_dir) / 'pytorch'
            pt_model.save_pretrained(pt_out_dir)
            save_tokenizer(tok, pt_out_dir)
        if pt_compress_weights and not gptq_applied:
            feature = 'text-generation'
            quantizer = OVQuantizer.from_pretrained(pt_model, task=feature)
            pt_out_dir = Path(args.output_dir) / 'compressed_weights' / f'PT_{args.precision}-INT8'
            quantizer.quantize(save_directory=pt_out_dir, weights_only=True)
            save_tokenizer(tok, pt_out_dir)
        del pt_model
        gc.collect()

    model = OVModelForCausalLM.from_pretrained(
        args.model_id,
        export=True,
        compile=False,
        trust_remote_code=True,
        load_in_8bit=False,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    end = time.perf_counter()

    log.info(f'Conversion total time {end - start}s')
    if args.precision == 'FP16':
        model.half()
    ov_out_dir = Path(args.output_dir) / precision
    save_tokenizer(tok, ov_out_dir)

    start1 = time.perf_counter()
    model.save_pretrained(ov_out_dir)
    end1 = time.perf_counter()
    log.info(f'Serialization total time {end1 - start1}s')

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends and not gptq_applied:
        ov_int8_dir = Path(args.output_dir) / 'compressed_weights' / f'OV_{args.precision}-INT8'
        model.model = compress_weights(model.model)
        model.save_pretrained(ov_int8_dir)
        save_tokenizer(tok, ov_int8_dir)

    del model
    gc.collect()

def convert_chatglm2(args):
    class ChatGLM2NormalizedConfig(NormalizedTextConfig):
        NUM_LAYERS = "num_layers"
        VOCAB_SIZE = "padded_vocab_size"

    class ChatGLM2DummyTextInputGenerator(DummyTextInputGenerator):
        SUPPORTED_INPUT_NAMES = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
        }

        def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
            input = super().generate(input_name, framework, int_dtype, float_dtype)
            if input_name == "attention_mask":
                input = torch.ones((input.shape[0], input.shape[1] + 1), dtype=input.dtype)
            if input_name == "position_ids":
                input = torch.range(0, input.shape[1] + 1, dtype=input.dtype).repeat(1, 1)
            return input

    class ChatGLM2DummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
        def __init__(
            self,
            task: str,
            normalized_config: NormalizedTextConfig,
            batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
            sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
            random_batch_size_range: Optional[Tuple[int, int]] = None,
            random_sequence_length_range: Optional[Tuple[int, int]] = None,
            **kwargs,
        ):
            super().__init__(
                task=task,
                normalized_config=normalized_config,
                batch_size=batch_size,
                sequence_length=sequence_length,
                random_batch_size_range=random_batch_size_range,
                random_sequence_length_range=random_sequence_length_range,
            )
            self.multi_query_group_num = normalized_config.multi_query_group_num
            self.head_dim = self.hidden_size // self.num_attention_heads

        def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
            past_key_shape = (
                self.sequence_length,
                self.batch_size,
                self.multi_query_group_num,
                self.head_dim,
            )
            past_value_shape = (
                self.sequence_length,
                self.batch_size,
                self.multi_query_group_num,
                self.head_dim,
            )
            return [
                (
                    self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.num_layers)
            ]

    class ChatGLM2OpenVINOConfig(TextDecoderOnnxConfig):
        NORMALIZED_CONFIG_CLASS = ChatGLM2NormalizedConfig
        DUMMY_INPUT_GENERATOR_CLASSES = (ChatGLM2DummyTextInputGenerator, ChatGLM2DummyPastKeyValuesGenerator)
        DUMMY_PKV_GENERATOR_CLASS = ChatGLM2DummyPastKeyValuesGenerator
        no_position_ids = False

        @property
        def inputs(self) -> Dict[str, Dict[int, str]]:
            common_inputs = super().inputs
            if not self.no_position_ids and self.task == "text-generation":
                common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}

            return common_inputs

        def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
            """
            Fills `input_or_outputs` mapping with past_key_values dynamic axes considering the direction.

            Args:
                inputs_or_outputs (`Dict[str, Dict[int, str]]`): The mapping to fill.
                direction (`str`):
                    either "inputs" or "outputs", it specifies whether `input_or_outputs` is the input mapping or the
                    output mapping, this is important for axes naming.
            """
            if direction not in ["inputs", "outputs"]:
                raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

            if direction == "inputs":
                decoder_sequence_name = "past_sequence_length"
                name = "past_key_values"
            else:
                decoder_sequence_name = "past_sequence_length + 1"
                name = "present"

            for i in range(self._normalized_config.num_layers):
                inputs_or_outputs[f"{name}.{i}.key"] = {1: "batch_size", 0: decoder_sequence_name}
                inputs_or_outputs[f"{name}.{i}.value"] = {1: "batch_size", 0: decoder_sequence_name}

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    pt_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=config,
    )
    pt_model.to(torch.float32)

    NormalizedConfigManager._conf[pt_model.config.model_type] = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers", num_attention_heads="num_attention_heads"
    )
    export_config = ChatGLM2OpenVINOConfig
    TasksManager._SUPPORTED_MODEL_TYPE[pt_model.config.model_type] = {
        'onnx': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
        'openvino': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
    }
    convert_optimum_causallm_base(pt_model, args)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_chatglm(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        pt_model.config.torchscript = True
        last_token = torch.tensor([[130328]])
        past = torch.zeros(28, 2, 5, 1, 32, 128)
        position_ids = torch.tensor([[[2], [4]]])
        dummy_input = {
            'input_ids': last_token,
            'past_key_values': past,
            'position_ids': position_ids,
        }
        ov_model = convert_model(pt_model, example_input=dummy_input)
        ov_model.outputs[0].get_tensor().set_names({'logits'})
        for i in range(1, len(ov_model.outputs), 2):
            idx = (i - 1) // 2
            ov_model.outputs[i].get_tensor().set_names({f'present.{int(idx)}.key'})
            ov_model.outputs[i + 1].get_tensor().set_names({f'present.{int(idx)}.value'})
        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    pt_model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    pt_model.config.use_cache = True
    pt_model.to(torch.float32)
    pt_model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    compress_to_fp16 = args.precision == 'FP16'
    ov_out_path = Path(args.output_dir) / args.precision
    convert_to_ov(pt_model, tok, ov_out_path, compress_to_fp16=compress_to_fp16)

    if args.compress_weights:
        ov_model_path = ov_out_path / 'openvino_model.xml'
        ov_model = Core().read_model(ov_model_path)
        ov_compressed_path = Path(args.output_dir) / 'compressed_weights' / f'OV_{args.precision}-INT8'
        compress_ov_model_weights_helper(ov_model, tok, pt_model.config, ov_compressed_path, compress_to_fp16)

def convert_qwen(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    normalized_config = NormalizedTextConfig.with_args(num_layers='num_hidden_layers', num_attention_heads='num_attention_heads', hidden_size='hidden_size')
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
    model.to(torch.float32)
    class QwenDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
        def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
            shape = (
                self.batch_size,
                self.sequence_length,
                self.num_attention_heads,
                self.hidden_size // self.num_attention_heads,
            )
            return [
                (
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.num_layers)
            ]

    class QwenOpenVINOConfig(TextDecoderOnnxConfig):
        DEFAULT_ONNX_OPSET = 13
        NORMALIZED_CONFIG_CLASS = normalized_config
        DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, QwenDummyPastKeyValuesGenerator)
        DUMMY_PKV_GENERATOR_CLASS = QwenDummyPastKeyValuesGenerator

        def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
            dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

            dummy_inputs = {}
            input_names = [key for key in self.inputs.keys() if not key.startswith("past_key_values")]
            if self.use_past_in_inputs and self.use_cache_branch is not False:
                input_names.append("past_key_values")

            for input_name in input_names:
                input_was_inserted = False
                for dummy_input_gen in dummy_inputs_generators:
                    if dummy_input_gen.supports_input(input_name):
                        dummy_inputs[input_name] = self.overwrite_shape_and_generate_input(
                            dummy_input_gen,
                            input_name,
                            framework,
                            input_shapes=kwargs,
                        )
                        input_was_inserted = True
                        break
                if not input_was_inserted:
                    raise RuntimeError(
                        f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model ONNX config.'
                    )

            # refer to https://github.com/huggingface/optimum/pull/764
            if (
                self.use_past_in_inputs
                and self.PAD_ATTENTION_MASK_TO_PAST
                and self.use_cache_branch is not False
                and "attention_mask" in dummy_inputs
            ):
                # Obtain the past sequence length from the value instead of the key (Bloom).
                past_length = dummy_inputs["past_key_values"][0][1].shape[1]

                dummy_inputs["attention_mask"] = DummyInputGenerator.pad_input_on_dim(
                    dummy_inputs["attention_mask"],
                    desired_length=past_length + 1,
                    dim=1,
                    dtype=dummy_inputs["attention_mask"].dtype,
                )

            return dummy_inputs

        def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
            """
            Fills `input_or_outputs` mapping with past_key_values dynamic axes considering the direction.
            Args:
                inputs_or_outputs (`Dict[str, Dict[int, str]]`): The mapping to fill.
                direction (`str`):
                    either "inputs" or "outputs", it specifies whether `input_or_outputs` is the input mapping or the
                    output mapping, this is important for axes naming.
            """
            if direction not in ["inputs", "outputs"]:
                raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

            if direction == "inputs":
                decoder_sequence_name = "past_sequence_length"
                name = "past_key_values"
            else:
                decoder_sequence_name = "past_sequence_length + 1"
                name = "present"

            for i in range(self._normalized_config.num_layers):
                inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 1: decoder_sequence_name}
                inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 1: decoder_sequence_name}

    model_type = model.config.model_type.replace("-", "_")
    export_config = QwenOpenVINOConfig
    TasksManager._SUPPORTED_MODEL_TYPE[model_type] = {
        'onnx': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
        'openvino': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
    }
    NormalizedConfigManager._conf[model_type] = normalized_config
    convert_optimum_causallm_base(model, args)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)

converters = {
    'decoder': convert_causal_lm,
    'chatglm2': convert_chatglm2,
    'chatglm3': convert_chatglm2,
    'qwen': convert_qwen,
}


def get_convert_model_type(model_id):
    default = 'decoder'
    for key in converters:
        if key in model_id:
            return key

    return default

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    parser = ArgumentParser()
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--save_orig', action='store_true')
    parser.add_argument('--precision', choices=['FP32', 'FP16'], default='FP32')
    compression_group = parser.add_argument_group('Weights compression parameters')
    compression_group.add_argument('--compress_weights', action='store_true')
    compression_group.add_argument(
        '--compress_weights_backends',
        help='Backend names used to compress the input model weights separated by space.',
        choices=[BackendType.PYTORCH.value, BackendType.OPENVINO.value],
        default=BackendType.OPENVINO.value,
        type=str.lower,
        nargs='+',
    )

    args = parser.parse_args()
    model_type = get_convert_model_type(args.model_id.lower())
    converter = converters[model_type]
    converter(args)

if __name__ == "__main__":
    main()
