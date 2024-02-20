import os, torch, logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
import dataclasses
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import logging
from argparse import ArgumentParser
import time 
import torch.cuda.nvtx as nvtx
from torch import nn
from transformers import LlamaModel, LlamaPreTrainedModel
from modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

import torch
from our_trainer import OurSFTTrainer

# def backward_hook(module, grad_input, grad_output):
#     nvtx.range_push("backward")
#     torch.cuda.synchronize()  # Ensure backward range covers entire backward pass
#     nvtx.range_pop()


# class Cache:
#     """
#     Base, abstract class for all caches. The actual data structure is specific to each subclass.
#     """

#     def update(
#         self,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         layer_idx: int,
#         cache_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

#         Parameters:
#             key_states (`torch.Tensor`):
#                 The new key states to cache.
#             value_states (`torch.Tensor`):
#                 The new value states to cache.
#             layer_idx (`int`):
#                 The index of the layer to cache the states for.
#             cache_kwargs (`Dict[str, Any]`, `optional`):
#                 Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
#                 cache to be created.

#         Return:
#             A tuple containing the updated key and value states.
#         """
#         raise NotImplementedError("Make sure to implement `update` in a subclass.")

#     def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
#         """Returns the sequence length of the cached states. A layer index can be optionally passed."""
#         raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

#     def get_max_length(self) -> Optional[int]:
#         """Returns the maximum sequence length of the cached states, if there is any."""
#         raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

#     def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
#         """Given the sequence length of the new inputs, returns the usable length of the cache."""
#         # Cache without size limit -> all cache is usable
#         # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
#         #   length, we will need to evict part of the cache (and thus not all cache is usable)
#         max_length = self.get_max_length()
#         previous_seq_length = self.get_seq_length(layer_idx)
#         if max_length is not None and previous_seq_length + new_seq_length > max_length:
#             return max_length - new_seq_length
#         return previous_seq_length

# class DynamicCache(Cache):
#     """
#     A cache that grows dynamically as more tokens are generated. This is the default for generative models.

#     It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
#     `[batch_size, num_heads, seq_len, head_dim]`.
#     """

#     def __init__(self) -> None:
#         self.key_cache: List[torch.Tensor] = []
#         self.value_cache: List[torch.Tensor] = []
#         self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

#     def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
#         """
#         Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
#         sequence length.
#         """
#         if layer_idx < len(self):
#             return (self.key_cache[layer_idx], self.value_cache[layer_idx])
#         else:
#             raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

#     def __iter__(self):
#         """
#         Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
#         keys and values
#         """
#         for layer_idx in range(len(self)):
#             yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

#     def __len__(self):
#         """
#         Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
#         to the number of layers in the model.
#         """
#         return len(self.key_cache)

#     def update(
#         self,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         layer_idx: int,
#         cache_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

#         Parameters:
#             key_states (`torch.Tensor`):
#                 The new key states to cache.
#             value_states (`torch.Tensor`):
#                 The new value states to cache.
#             layer_idx (`int`):
#                 The index of the layer to cache the states for.
#             cache_kwargs (`Dict[str, Any]`, `optional`):
#                 Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

#         Return:
#             A tuple containing the updated key and value states.
#         """
#         # Update the number of seen tokens
#         if layer_idx == 0:
#             self.seen_tokens += key_states.shape[-2]

#         # Update the cache
#         if len(self.key_cache) <= layer_idx:
#             self.key_cache.append(key_states)
#             self.value_cache.append(value_states)
#         else:
#             self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
#             self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

#         return self.key_cache[layer_idx], self.value_cache[layer_idx]

#     def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
#         """Returns the sequence length of the cached states. A layer index can be optionally passed."""
#         if len(self.key_cache) <= layer_idx:
#             return 0
#         return self.key_cache[layer_idx].shape[-2]

#     def get_max_length(self) -> Optional[int]:
#         """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
#         return None

#     def reorder_cache(self, beam_idx: torch.LongTensor):
#         """Reorders the cache for beam search, given the selected beam indices."""
#         for layer_idx in range(len(self.key_cache)):
#             device = self.key_cache[layer_idx].device
#             self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
#             device = self.value_cache[layer_idx].device
#             self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

#     def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
#         """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
#         legacy_cache = ()
#         for layer_idx in range(len(self)):
#             legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
#         return legacy_cache

#     @classmethod
#     def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
#         """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
#         cache = cls()
#         if past_key_values is not None:
#             for layer_idx in range(len(past_key_values)):
#                 key_states, value_states = past_key_values[layer_idx]
#                 cache.update(key_states, value_states, layer_idx)
#         return cache

# class SinkCache(Cache):
#     """
#     A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
#     generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
#     tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

#     It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
#     `[batch_size, num_heads, seq_len, head_dim]`.

#     Parameters:
#         window_length (`int`):
#             The length of the context window.
#         num_sink_tokens (`int`):
#             The number of sink tokens. See the original paper for more information.
#     """

#     def __init__(self, window_length: int, num_sink_tokens: int) -> None:
#         self.key_cache: List[torch.Tensor] = []
#         self.value_cache: List[torch.Tensor] = []
#         self.window_length = window_length
#         self.num_sink_tokens = num_sink_tokens
#         self.cos_sin_cache = {}
#         self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

#     @staticmethod
#     def _rotate_half(x):
#         x1 = x[..., : x.shape[-1] // 2]
#         x2 = x[..., x.shape[-1] // 2 :]
#         return torch.cat((-x2, x1), dim=-1)

#     def _apply_key_rotary_pos_emb(
#         self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
#     ) -> torch.Tensor:
#         rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
#         return rotated_key_states

#     def _get_rerotation_cos_sin(
#         self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if key_states.shape[-2] not in self.cos_sin_cache:
#             # Upcast to float32 temporarily for better accuracy
#             cos = cos.to(torch.float32)
#             sin = sin.to(torch.float32)

#             # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
#             original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
#             shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
#             original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
#             shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
#             rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
#             rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

#             self.cos_sin_cache[key_states.shape[-2]] = (
#                 rerotation_cos.to(key_states.dtype).unsqueeze(0),
#                 rerotation_sin.to(key_states.dtype).unsqueeze(0),
#             )
#         return self.cos_sin_cache[key_states.shape[-2]]

#     def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
#         """Returns the sequence length of the cached states. A layer index can be optionally passed."""
#         # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
#         if len(self.key_cache) <= layer_idx:
#             return 0
#         return self.key_cache[layer_idx].shape[-2]

#     def get_max_length(self) -> Optional[int]:
#         """Returns the maximum sequence length of the cached states."""
#         return self.window_length

#     def update(
#         self,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         layer_idx: int,
#         cache_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

#         Parameters:
#             key_states (`torch.Tensor`):
#                 The new key states to cache.
#             value_states (`torch.Tensor`):
#                 The new value states to cache.
#             layer_idx (`int`):
#                 The index of the layer to cache the states for.
#             cache_kwargs (`Dict[str, Any]`, `optional`):
#                 Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
#                 `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
#                 rotation as the tokens are shifted.

#         Return:
#             A tuple containing the updated key and value states.
#         """
#         # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
#         # with partially rotated position embeddings, like Phi or Persimmon.
#         sin = cache_kwargs.get("sin")
#         cos = cache_kwargs.get("cos")
#         partial_rotation_size = cache_kwargs.get("partial_rotation_size")
#         using_rope = cos is not None and sin is not None

#         # Update the number of seen tokens
#         if layer_idx == 0:
#             self.seen_tokens += key_states.shape[-2]

#         # [bsz, num_heads, seq_len, head_dim]
#         if len(self.key_cache) <= layer_idx:
#             # Empty cache
#             self.key_cache.append(key_states)
#             self.value_cache.append(value_states)

#         elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
#             # Growing cache
#             self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
#             self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

#         else:
#             # Shifting cache
#             keys_to_keep = self.key_cache[layer_idx][
#                 :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
#             ]

#             # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
#             if using_rope:
#                 rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
#                     key_states, cos[: self.window_length], sin[: self.window_length]
#                 )
#                 if partial_rotation_size is not None:
#                     keys_to_keep, keys_pass = (
#                         keys_to_keep[..., :partial_rotation_size],
#                         keys_to_keep[..., partial_rotation_size:],
#                     )
#                 keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)
#                 if partial_rotation_size is not None:
#                     keys_to_keep = torch.cat((keys_to_keep, keys_pass), dim=-1)

#             # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
#             sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]
#             self.key_cache[layer_idx] = torch.cat([sink_keys, keys_to_keep, key_states], dim=-2)

#             sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]
#             values_to_keep = self.value_cache[layer_idx][
#                 :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
#             ]
#             self.value_cache[layer_idx] = torch.cat([sink_values, values_to_keep, value_states], dim=-2)

#         return self.key_cache[layer_idx], self.value_cache[layer_idx]

#     def reorder_cache(self, beam_idx: torch.LongTensor):
#         """Reorders the cache for beam search, given the selected beam indices."""
#         for layer_idx in range(len(self.key_cache)):
#             device = self.key_cache[layer_idx].device
#             self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
#             device = self.value_cache[layer_idx].device
#             self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

# class LlamaForCausalLM(LlamaPreTrainedModel):
#     _tied_weights_keys = ["lm_head.weight"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = LlamaModel(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def set_decoder(self, decoder):
#         self.model = decoder

#     def get_decoder(self):
#         return self.model

#     # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         r"""
#         Args:
#             labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#                 config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#                 (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Returns:

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, LlamaForCausalLM

#         >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
#         >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

#         >>> prompt = "Hey, are you conscious? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
#         ```"""
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         torch.cuda.nvtx.range_push("forward")
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         torch.cuda.nvtx.range_pop()

#         hidden_states = outputs[0]
#         if self.config.pretraining_tp > 1:
#             lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
#             logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
#             logits = torch.cat(logits, dim=-1)
#         else:
#             logits = self.lm_head(hidden_states)
#         logits = logits.float()

#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def prepare_inputs_for_generation(
#         self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
#     ):
#         if past_key_values is not None:
#             if isinstance(past_key_values, Cache):
#                 cache_length = past_key_values.get_seq_length()
#                 past_length = past_key_values.seen_tokens
#                 max_cache_length = past_key_values.get_max_length()
#             else:
#                 cache_length = past_length = past_key_values[0][0].shape[2]
#                 max_cache_length = None

#             # Keep only the unprocessed tokens:
#             # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
#             # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
#             # input)
#             if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
#                 input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
#             # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
#             # input_ids based on the past_length.
#             elif past_length < input_ids.shape[1]:
#                 input_ids = input_ids[:, past_length:]
#             # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

#             # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
#             if (
#                 max_cache_length is not None
#                 and attention_mask is not None
#                 and cache_length + input_ids.shape[1] > max_cache_length
#             ):
#                 attention_mask = attention_mask[:, -max_cache_length:]

#         position_ids = kwargs.get("position_ids", None)
#         if attention_mask is not None and position_ids is None:
#             # create position_ids on the fly for batch generation
#             position_ids = attention_mask.long().cumsum(-1) - 1
#             position_ids.masked_fill_(attention_mask == 0, 1)
#             if past_key_values:
#                 position_ids = position_ids[:, -input_ids.shape[1] :]

#         # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
#         if inputs_embeds is not None and past_key_values is None:
#             model_inputs = {"inputs_embeds": inputs_embeds}
#         else:
#             model_inputs = {"input_ids": input_ids}

#         model_inputs.update(
#             {
#                 "position_ids": position_ids,
#                 "past_key_values": past_key_values,
#                 "use_cache": kwargs.get("use_cache"),
#                 "attention_mask": attention_mask,
#             }
#         )
#         return model_inputs

#     @staticmethod
#     def _reorder_cache(past_key_values, beam_idx):
#         reordered_past = ()
#         for layer_past in past_key_values:
#             reordered_past += (
#                 tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
#             )
#         return reordered_past
     

def main(args):
    training_data = load_dataset(
        "mlabonne/guanaco-llama2-1k",
        split="train").select(range(args.num))
    tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", use_auth_token="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")
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
    else: 
        bnb_config=None 
    
    if args.flash:
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                quantization_config=bnb_config,
                use_auth_token="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")
    else: 
        model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                trust_remote_code=True,
                quantization_config=bnb_config,
                use_auth_token="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")
    # model.register_full_backward_hook(backward_hook)
    model.config.use_cache=False
    model.config.pretraining_tp=1
    if args.lora: 
        peft_parameters = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
         peft_parameters = None
    # Training Params
    train_params = TrainingArguments(
        output_dir="./",
        num_train_epochs=1,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        save_strategy="no",
        evaluation_strategy="no",
        logging_strategy="no",
        report_to="none",
    )
    # Trainer
    fine_tuning = OurSFTTrainer(
        model=model,
        train_dataset=training_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=train_params,
        max_seq_length=args.max_length,
    )
    start = time.time()
    fine_tuning.train()
    print(f"llama finetune time: {time.time() - start}")



if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('--model',type=str,required=True)
    # parser.add_argument('--cpu',action="store_true",default=False)
    parser.add_argument('--num',type=int,default=500,help="number of wikitext instances to run inference on.")
    parser.add_argument('--use_4bit',default=False,action="store_true")
    parser.add_argument('--flash',default=False, action='store_true')
    parser.add_argument('--lora',default=False,action="store_true")
    parser.add_argument('--bsz',type=int,default=4)
    parser.add_argument('--max_length',type=int,default=256)
    # parser.add_argument('--max_length',type=int,default=256)
    args = parser.parse_args()
    main(args)
