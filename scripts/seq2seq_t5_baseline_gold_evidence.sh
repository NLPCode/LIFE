echo "Start to train the seq2se corrector for factual error correction."


# t5 models do not support fp16 training.
CUDA_VISIBLE_DEVICES=0,1

# 1. Train
python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=9536  \
    ../models/seq2seq_baseline.py  \
    --train_file $DATA_DIR/train.jsonl \
    --validation_file $DATA_DIR/dev.jsonl \
    --initialization t5-base \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr 4e-5 \
    --logging_steps 100 \
    --save_steps 200 --max_steps 4000 \
    --do_train --use_evidence --num_evidence 2 --use_gold_evidence

# 2. Eval
python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=9536  \
    ../models/seq2seq_baseline.py  \
    --validation_file $DATA_DIR/dev.jsonl \
    --initialization t5-base \
    --use_evidence --num_evidence 2 --use_gold_evidence \
    --model_path ../checkpoints_corrector/t5-base/seed42_lr4e-05_2-gold-evidence  --resume \
    --do_eval

# 3. Predict
# do not use fp16 when generate, otherwise the results will depend on batch size.
CUDA_VISIBLE_DEVICES=0
for num_beams in 5
do
    python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=9536  \
        ../models/seq2seq_baseline.py   \
        --test_file $DATA_DIR/test.jsonl \
        --initialization t5-base \
        --per_device_eval_batch_size 64 \
        --use_evidence --num_evidence 2 --use_gold_evidence \
        --model_path ../checkpoints_corrector/t5-base/seed42_lr4e-05_2-gold-evidence  --resume \
        --num_beams $num_beams \
        --do_predict
done


