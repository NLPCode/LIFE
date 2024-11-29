DATA_NAME=fact_verification_data
DATA_DIR=../$DATA_NAME

# 1. Train
CUDA_VISIBLE_DEVICES=0,1
python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=9536  \
    ../models/fact_verifier.py  \
    --data_name $DATA_NAME \
    --train_file $DATA_DIR/train.jsonl \
    --validation_file $DATA_DIR/dev.jsonl \
    --model_name_or_path roberta-base \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr 2e-05 \
    --logging_steps 200 \
    --save_steps 200 --max_steps 4000 \
    --use_evidence true --use_gold_evidence true --num_evidence 2 \
    --fp16 --do_train 

# 2. Eval
    
python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=9536  \
    ../models/fact_verifier.py \
    --validation_file $DATA_DIR/dev.jsonl \
    --model_name_or_path ../checkpoints_verifier/roberta-base/${DATA_NAME}_seed42_lr2e-05_2-gold-evidence \
    --use_evidence true --use_gold_evidence true --num_evidence 2 \
    --resume --fp16 --do_eval




