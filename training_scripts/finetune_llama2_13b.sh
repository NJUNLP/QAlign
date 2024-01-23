# stage 1: question alignment
# finetuning LLaMA2-13B on question translation data
bash finetune_dp.sh llama2-13b-hf gsmtrans_gsm8kinstruct_question_all-en

# stage 2: response alignment
# finetuning stage 1 model with MetaMathQA dataset
bash finetune_dp.sh llama2-13b-hf.gsmtrans_gsm8kinstruct_question_all-en.finetune metamath_all

# you may also want to finetune llama2-13b with GSM8KInstruct dataset, here is the command
# bash finetune_dp.sh llama2-13b-hf gsm8kinstruct_all

# you may also want to finetune llama2-13b with GSM8K dataset, here is the command
# bash finetune_dp.sh llama2-13b-hf gsm8kinstruct_en