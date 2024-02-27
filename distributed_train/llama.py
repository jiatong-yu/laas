import deepspeed
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
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

import torch


TRAINING_ARGS = {
    "output_dir": "output",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-6,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "linear",
    "warmup_steps": 500,
    "logging_dir": "logs",
    "logging_steps": 10,
}

DS_CONFIG = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 2
    },
    "pipeline_parallelism": {
        "pipeline_parallel_degree": 2, # Adjust based on your GPU setup
        "pipeline_parallel_partitions": 2,
        "pipeline_parallel_schedule": "1f1b",
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": TRAINING_ARGS["learning_rate"],
            "betas": (0.9, 0.999),
            "eps": TRAINING_ARGS["adam_epsilon"],
            "weight_decay": TRAINING_ARGS["weight_decay"]
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": TRAINING_ARGS["learning_rate"],
            "warmup_num_steps": TRAINING_ARGS["warmup_steps"]
        }
    }
}

def main(args):
    dataset = load_dataset(
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
    # model.config.use_cache=False
    # model.config.pretraining_tp=1
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
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    dataloader = DataLoader(tokenized_dataset,batch_size=args.bsz,shuffle=True)

    model_engine, optimizer, training_dataloader, _ = deepspeed.initialize(
            args=TRAINING_ARGS,
            model=model,
            model_parameters=model.parameters(),
            training_data=dataset, # Replace with your dataset
            config_params=DS_CONFIG,
        )
    for epoch in range(TRAINING_ARGS["num_train_epochs"]):
        for batch in training_dataloader:
            model_engine.train()
            optimizer.zero_grad()
            inputs = {k: v.to(model_engine.local_rank) for k, v in batch.items()}
            outputs = model_engine(**inputs)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()




if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('--model',type=str,required=True)
    # parser.add_argument('--cpu',action="store_true",default=False)
    parser.add_argument('--num',type=int,default=500,help="number of wikitext instances to run inference on.")
    parser.add_argument('--use_4bit',default=False,action="store_true")
    parser.add_argument('--flash',default=False, action='store_true')
    parser.add_argument('--lora',default=False,action="store_true")
    parser.add_argument('--bsz',type=int,default=4)
    parser.add_argument('--max_length',type=int,default=512)
    # parser.add_argument('--max_length',type=int,default=256)
    args = parser.parse_args()
    main(args)
