
PATH_TO_DEEPSPEED_CONFIG=playground/deepspeed_config_s2.json

deepspeed fastchat/train/train_lora.py \
    --model_name_or_path /scratch/ml/mfx/huggingface/Llama-2-13b-chat-hf \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path /scratch/ml/xiaojuan/datasets/FOLIO/data/train_ft.json \
    --output_dir /scratch/ml/xiaojuan/ft_models/llama2-13-chat-folio \
    --num_train_epochs 1 \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --q_lora False \
    --deepspeed $PATH_TO_DEEPSPEED_CONFIG \
    --gradient_checkpointing True \
    --flash_attn True
