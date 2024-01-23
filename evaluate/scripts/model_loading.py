from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel, LlamaForCausalLM, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Literal, Optional, Type, TypeVar, Union
import math
import torch

class ContrastiveDecodeModel(PreTrainedModel):
    def __init__(self, expert: LlamaForCausalLM, amateur: LlamaForCausalLM, alpha: float, beta: float) -> None:
        super().__init__(expert.config)
        self.expert = expert
        self.amateur = amateur
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> CausalLMOutputWithPast:
        assert return_dict
        assert not output_attentions
        assert not output_hidden_states

        if past_key_values is not None:
            expert_past_key_values = past_key_values[:-1]
            amateur_past_key_values = past_key_values[-1]
        else:
            expert_past_key_values = None
            amateur_past_key_values = None

        output_expert: CausalLMOutputWithPast = self.expert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=expert_past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        output_amateur: CausalLMOutputWithPast = self.amateur(
            input_ids=input_ids.to(self.amateur.device),
            attention_mask=attention_mask.to(self.amateur.device) if attention_mask is not None else None,
            position_ids=position_ids.to(self.amateur.device) if position_ids is not None else None,
            past_key_values=amateur_past_key_values,
            inputs_embeds=inputs_embeds.to(self.amateur.device) if inputs_embeds is not None else None,
            labels=labels.to(self.amateur.device) if labels is not None else None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        expert_logits = output_expert.logits
        amateur_logits = output_amateur.logits.to(expert_logits.device)

        cutoff = math.log(self.alpha) + expert_logits.max(dim=-1, keepdim=True).values
        diffs = (1 + self.beta) * expert_logits - self.beta * amateur_logits
        cd_logits = diffs.masked_fill(expert_logits < cutoff, -float('inf'))

        if use_cache:
            past_key_values = (output_expert.past_key_values + (output_amateur.past_key_values,))
        else:
            past_key_values = None

        outputs = CausalLMOutputWithPast(
            logits=cd_logits,
            past_key_values=past_key_values,
        )
        outputs['expert_logits'] = expert_logits
        return outputs
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.expert.prepare_inputs_for_generation(*args, **kwargs)
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return (
            LlamaForCausalLM._reorder_cache(past_key_values[:-1], beam_idx),
            LlamaForCausalLM._reorder_cache(past_key_values[-1], beam_idx),
        )
    

class ContrastiveDecodeWithDifferentPromptModel(PreTrainedModel):
    def __init__(self, expert: LlamaForCausalLM, amateur: LlamaForCausalLM, alpha: float, beta: float) -> None:
        super().__init__(expert.config)
        self.expert = expert
        self.amateur = amateur
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> CausalLMOutputWithPast:
        assert input_ids.size(0) == 2
        expert_input_ids = input_ids[:1]
        amateur_input_ids = input_ids[1:]
        
        if attention_mask is not None:
            assert attention_mask.size(0) == 2
            expert_attention_mask = attention_mask[:1]
            amateur_attention_mask = attention_mask[1:]
        else:
            expert_attention_mask = None
            amateur_attention_mask = None

        if position_ids is not None:
            assert position_ids.size(0) == 2
            expert_position_ids = position_ids[:1]
            amateur_position_ids = position_ids[1:]
        else:
            expert_position_ids = None
            amateur_position_ids = None

        if past_key_values is not None:
            expert_past_key_values = past_key_values[:-1]
            amateur_past_key_values = past_key_values[-1]
        else:
            expert_past_key_values = None
            amateur_past_key_values = None

        assert inputs_embeds is None
        assert labels is None

        assert not output_attentions
        assert not output_hidden_states
        assert return_dict

        output_expert: CausalLMOutputWithPast = self.expert(
            input_ids=expert_input_ids,
            attention_mask=expert_attention_mask,
            position_ids=expert_position_ids,
            past_key_values=expert_past_key_values,
            inputs_embeds=None,
            labels=None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        output_amateur: CausalLMOutputWithPast = self.amateur(
            input_ids=amateur_input_ids.to(self.amateur.device),
            attention_mask=amateur_attention_mask.to(self.amateur.device) if attention_mask is not None else None,
            position_ids=amateur_position_ids.to(self.amateur.device) if position_ids is not None else None,
            past_key_values=amateur_past_key_values,
            inputs_embeds=None,
            labels=None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        expert_logits = output_expert.logits
        amateur_logits = output_amateur.logits.to(expert_logits.device)

        cutoff = math.log(self.alpha) + expert_logits.max(dim=-1, keepdim=True).values
        diffs = (1 + self.beta) * expert_logits - self.beta * amateur_logits
        cd_logits = diffs.masked_fill(expert_logits < cutoff, -float('inf'))

        if use_cache:
            past_key_values = (output_expert.past_key_values + (output_amateur.past_key_values,))
        else:
            past_key_values = None

        outputs = CausalLMOutputWithPast(
            logits=cd_logits.expand(2, -1, -1),
            past_key_values=past_key_values,
        )
        outputs['expert_logits'] = expert_logits
        return outputs
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.expert.prepare_inputs_for_generation(*args, **kwargs)
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return (
            LlamaForCausalLM._reorder_cache(past_key_values[:-1], beam_idx),
            LlamaForCausalLM._reorder_cache(past_key_values[-1], beam_idx),
        )


def load_contrastive_decoding_model(
    expert_model_dir: str,
    amateur_model_dir: str,
    alpha: float,
    beta: float,
) -> ContrastiveDecodeModel:
    expert_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        expert_model_dir,
        torch_dtype=torch.float16,
        device_map='auto',
        # device_map='balanced_low_0',
        max_memory={0: '24GiB', 1: '24GiB'},
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
    )
    print('expert_model loaded')

    amateur_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        amateur_model_dir,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map='balanced',
        # max_memory={0: '24GiB', 1: '24GiB'},
        max_memory={2: '24GiB'},
        # max_memory={6: '24GiB'},
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
    )
    print('amateur_model loaded')

    model = ContrastiveDecodeModel(expert_model, amateur_model, alpha=alpha, beta=beta)
    model.eval()
    return model


T = TypeVar('T')
def load_hf_model(
    path: str, 
    bit: Literal[4, 8, None] = 4, 
    autoload_cls: Optional[Type[T]] = None,
) -> Union[T, PreTrainedModel]:
    assert bit in (4, 8, None)
    kwargs = {}
    if bit == 4:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=False,
        )
    elif bit == 8:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    if autoload_cls is None:
        autoload_cls = AutoModelForCausalLM
    
    print(f'Loading model {path}')

    model = autoload_cls.from_pretrained(
        path,
        # device_map='balanced_low_0',
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attention_2=torch.cuda.get_device_capability()[0] >= 8,
        **kwargs,
    )
    return model

def load_autogptq_model(path):
    from auto_gptq import AutoGPTQForCausalLM
    model = AutoGPTQForCausalLM.from_quantized(
        path, 
        device='cuda:0', 
        model_basename="gptq_model-4bit",
        use_triton=False, 
        low_cpu_mem_usage=True,
        inject_fused_attention=False,
        trainable=True,
    )
    return model