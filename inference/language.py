import torch
import numpy as np
import time 
from datasets import load_dataset
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BertTokenizer, BertModel,
    BitsAndBytesConfig,
)

import logging
logging.getLogger().setLevel(logging.ERROR)


def main(args):
    model_name = args.model 
    data = load_dataset("wikitext","wikitext-103-raw-v1", split="train")
    def length_mask(inst):
        return (len(inst['text']) > 0 and len(inst['text']) < args.max_length)
    data = data.filter(length_mask).select(range(args.num))['text']
    if args.cpu:
        raise NotImplementedError()
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype="float16",bnb_4bit_use_double_quant=True,
    )
    
    #======================== BERT ========================#
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
        encoded_inputs = tokenizer(data,return_tensors="pt",padding="max_length", truncation=True, max_length=args.max_length)
        print(encoded_inputs['input_ids'].shape)
        # move to CUDA
        start = time.time()
        model.to("cuda")
        input_ids = encoded_inputs["input_ids"].to("cuda")
        attn_mask = encoded_inputs["attention_mask"].to("cuda")
        print(f"time to load {model_name}: {time.time() - start}")
        # inference time 
        inf_start = time.time()
        with torch.no_grad():
            output = model(input_ids,attention_mask = attn_mask)
        print(f"time to inference {model_name}: {time.time() - inf_start}")
        print(f"total time: {time.time() - start}")

    #======================== LLAMA ========================#
    elif model_name == "llama_7b":
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", use_auth_token="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")
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            trust_remote_code=True,
            quantization_config=bnb_config,
            use_auth_token="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")
        pipe_start = time.time()
        encoded_inputs = tokenizer(data, return_tensors="pt", padding=True)
        # move to cuda 
        start = time.time()
        encoded_inputs = encoded_inputs.to("cuda")
        if not args.use_4bit:
            model.to("cuda")
        print(f"time to load {model_name}: {time.time() - start}")
        # inference 
        start = time.time()
        with torch.no_grad():
            output = model.generate(**encoded_inputs, max_new_tokens=20)
        print(f"time to inference {model_name}: {time.time() - start}")
        print(f"total time: {time.time() - pipe_start}")

    #======================== MISTRAL ========================#
    elif model_name == "mistral_7b":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            quantization_config=bnb_config)
        start = time.time()
        model.to("cuda")
        loaded = time.time()
        print(f"time to load {model_name}: {loaded - start}")
        messages = []
        for msg in data:
            messages.append({
                "role":"user",
                "content":msg
            })

        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model',type=str,required=True)
    parser.add_argument('--cpu',action="store_true",default=False)
    parser.add_argument('--use_4bit',action="store_true",default=False)
    parser.add_argument('--num',type=int,default=100,help="number of wikitext instances to run inference on.")
    parser.add_argument('--max_length',type=int,default=256)
    args = parser.parse_args()
    main(args)
