

echo "Start to train the seq2se corrector on the synthetic data for factual error correction."

# t5 models do not support fp16 training.
NUM_DATA_INSTANCE=4000
input_file_prefix=../checkpoints_corruptor/t5-base/seed42_lr4e-05_2-gold-evidence_
model_path_prefix=../checkpoints_corrector/t5-base/seed42_lr4e-05_2-gold-evidence_

mask1=heuristic_word 
mask2=random_word_0.3 
leven_threshold=0.3
cls_threshold=0.2
min_prob=0.8
train_file=${input_file_prefix}${mask1}/leven${leven_threshold}_cls${cls_threshold}_${min_prob}_train_${mask2}_beam_1.txt
dev_file=${input_file_prefix}${mask1}/leven${leven_threshold}_cls${cls_threshold}_${min_prob}_dev_${mask2}_beam_1.txt
model_path=${model_path_prefix}${mask1}_leven${leven_threshold}_cls${cls_threshold}_${min_prob}_${mask2}_beam_1/${NUM_DATA_INSTANCE}data_seed42_lr4e-05_2-gold-evidence

wc -l $train_file
wc -l $dev_file
# 1. Train
CUDA_VISIBLE_DEVICES=0,1
python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=9536  \
    ../models/seq2seq_baseline.py  \
    --train_file $train_file \
    --validation_file $dev_file \
    --initialization t5-base \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr 4e-5 \
    --logging_steps 50 \
    --save_steps 50 --max_steps 1000 \
    --use_evidence --use_gold_evidence --num_evidence 2 \
    --num_data_instance $NUM_DATA_INSTANCE \
    --output_dir $model_path \
    --do_train 

DATA_DIR=../../PivotFEC/seq2seq_data
# 2. Predict
# do not use fp16 when generate, otherwise the results will depend on batch size.
CUDA_VISIBLE_DEVICES=0
for num_beams in 5
do
    python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=9536  \
        ../models/seq2seq_baseline.py  \
        --test_file $DATA_DIR/test.jsonl \
        --initialization t5-base \
        --per_device_eval_batch_size 64 \
        --use_evidence --use_gold_evidence --num_evidence 2 \
        --model_path $model_path --resume \
        --num_beams $num_beams \
        --do_predict
done

