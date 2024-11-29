
echo "Start to train the corruptor for factual error correction."

# t5 models do not support fp16 training.
CUDA_VISIBLE_DEVICES=0,1

train_file=$DATA_DIR/train.jsonl
dev_file=$DATA_DIR/dev.jsonl
test_file=$DATA_DIR/test.jsonl
# 1. Train

mask_strategy=heuristic
for mask_granularity in word
do
    for merge_mask in False
    do
        # retrieved evidence
        python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=9536 ../models/corruptor.py  \
            --train_file $train_file \
            --validation_file $dev_file \
            --initialization t5-base \
            --per_device_train_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --lr 4e-5 \
            --logging_steps 100 --save_steps 200 --max_steps 4000 \
            --do_train --use_evidence --num_evidence 2 \
            --mask_strategy $mask_strategy --mask_granularity $mask_granularity --merge_mask $merge_mask

    done
done


# 2. Predict
# do not use fp16 when generate, otherwise the results will depend on batch size.
CUDA_VISIBLE_DEVICES=0

mask_strategy=random
for mask_ratio in 0.3
do
    for mask_granularity in word
    do
        for test_file in $dev_file $train_file
        do
            # retrieved evidence
            python ../models/corruptor.py  \
                --test_file $test_file \
                --initialization t5-base \
                --per_device_eval_batch_size 64 \
                --do_predict --use_evidence --num_evidence 2 \
                --model_path ../checkpoints_corruptor/t5-base/seed42_lr4e-05_2-retrieved-evidence_heuristic_word --resume \
                --num_beams 1 \
                --mask_ratio $mask_ratio --mask_strategy $mask_strategy --mask_granularity $mask_granularity --merge_mask False
        done
    done
done


