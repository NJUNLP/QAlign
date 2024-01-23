PROJECT_PATH=/path/to/this/project
MODEL_PATH=/path/to/your/model

# For 13B model, you may need to set batch_size smaller, like 16, to avoid OOM issue.
python $PROJECT_PATH/scripts/mgsm_test.py \
    --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 32 \
    --lang_only Bengali Thai Swahili Japanese Chinese German French Russian Spanish English