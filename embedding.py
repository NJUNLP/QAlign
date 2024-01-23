from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Optional, List
from datasets import load_dataset
import torch
import json
import transformers
from transformers import GenerationConfig
import os
import re
import copy

from train import smart_tokenizer_and_embedding_resize, \
	DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, \
	PROMPT_DICT, \
    DataArguments

import train


@dataclass
class ModelArguments(train.ModelArguments):
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the model in 8-bit mode."},
    )
    torch_dtype: torch.dtype = field(
        default=torch.bfloat16,
        metadata={"help": "The dtype to use for inference."},
    )


@dataclass
class GeneratingArguments:
    batch_size: int = field(default=8)
    output_file: str = field(default=None, metadata={"help": "Path to the output."})
    template: str = field(default="alpaca")


def inference():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GeneratingArguments))
    model_args, data_args, generating_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=model_args.load_in_8bit,
        torch_dtype=model_args.torch_dtype,
        device_map="auto",
    )
    model.eval()

    if torch.cuda.device_count() > 1:
        from accelerate import load_checkpoint_and_dispatch
        load_checkpoint_and_dispatch(
            model,
            model_args.model_name_or_path,
            device_map="auto",
            offload_state_dict=True,
            no_split_module_classes=["LlamaDecoderLayer"],
        )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.padding_side = "left"

    data_path_base, data_path_name = data_args.data_path.rsplit(os.path.sep, maxsplit=1)
    dataset_name, dataset_config = data_path_name.split("_", maxsplit=1)
    test_dataset = load_dataset(os.path.join(data_path_base, dataset_name), config=dataset_config, split="test")

    def generate_prompt(instruction, input=None, template="alpaca"):
        if template == "alpaca":
            if input:
                return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
            else:
                return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
        elif template == "raw":
            if input:
                return f"{instruction}\n\n{input}"
            else:
                return f"{instruction}"
        elif template == "instruction":
            return instruction
        elif template == "input":
            if input:
                return f"{input}"
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
    def evaluate_by_perplexity(
        dataset,
        template,
    ):
        prompt = [generate_prompt(ins, inp, template) for ins, inp in zip(dataset["instruction"], dataset["input"])]
        inputs = tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            )
        mean_hidden_states = [
            torch.mul(
                inputs["attention_mask"].unsqueeze(-1).expand(hidden_state.shape), 
                hidden_state
            ).mean(dim=1).cpu()
            for hidden_state in output.hidden_states
        ]
        return mean_hidden_states
    
    # with open(generating_args.output_file, "w") as output_file:
    hidden_state_by_batch = []
    for i in tqdm(range(0, len(test_dataset), generating_args.batch_size)):
        d = test_dataset[i:i + generating_args.batch_size]
        hidden_state_by_batch.append(evaluate_by_perplexity(d, template=generating_args.template))
    hidden_state_by_layer = [
        torch.concat(hidden, dim=0) 
        for hidden in zip(*hidden_state_by_batch)]
    
    torch.save(hidden_state_by_layer, generating_args.output_file)
    
if __name__ == "__main__":
    inference()
    