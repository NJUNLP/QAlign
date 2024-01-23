#!/bin/bash
BASE_MODEL=$1
DATASET=$2
METHOD=${3:-"finetune"}

PORT=$(( $RANDOM % 1000 + 32768 ))
CPFS_PATH=/cpfs01/user/zhuwenhao
PROJECT_PATH=$CPFS_PATH/project/QAlign
OUTPUT_NAME=$BASE_MODEL.$DATASET.$METHOD

MODEL_ARGS=()
case $BASE_MODEL in  
    "llama2-7b-hf")
		MODEL_ARGS+=("--num_train_epochs 3")
		MODEL_ARGS+=("--learning_rate 2e-5")
        FSDP="full_shard offload auto_wrap"
		;;
	"llama2-13b-hf")
		MODEL_ARGS+=("--num_train_epochs 3")
		MODEL_ARGS+=("--learning_rate 2e-5")
        FSDP="full_shard offload auto_wrap"
		;;
	*)  
		MODEL_ARGS+=("--num_train_epochs 3")
		MODEL_ARGS+=("--learning_rate 2e-5")
        FSDP="full_shard auto_wrap"
		;;
esac

METHOD_ARGS=()
case $METHOD in  
	"finetune")
		;;  
	*)  
		;;  
esac

source $CPFS_PATH/miniconda3/bin/activate $PROJECT_PATH/.env

torchrun --nproc_per_node=8 --master_port=$PORT \
    $PROJECT_PATH/finetune.py \
	${METHOD_ARGS[@]} \
	${MODEL_ARGS[@]} \
    --data_path "$PROJECT_PATH/data/$DATASET" \
    --model_name_or_path "$PROJECT_PATH/model/$BASE_MODEL" \
    --output_dir "$PROJECT_PATH/model/$OUTPUT_NAME" \
	--deepspeed "$PROJECT_PATH/config/ds.json" \
    --bf16 True \
    --tf32 True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --logging_steps 1 \
    --report_to wandb tensorboard \
    --seed 1234 \
    --logging_dir "$CPFS_PATH/log/tensorboard/$OUTPUT_NAME"