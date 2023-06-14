export MODEL_NAME="/home/ubuntu/roentgen"
export TRAIN_DIR="/home/ubuntu/CS231n-Final-Project/generated_images"
export OUTPUT_DIR="/home/ubuntu/CS231n-Final-Project/new_models"

CUDA_LAUNCH_BLOCKING=1 accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --max_train_steps=100 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub
