
# This script aims to filter the generated data.
CUDA_VISIBLE_DEVICES=0

input_file_prefix=../checkpoints_corruptor/t5-base/seed42_lr4e-05_2-retrieved-evidence_
mask1=heuristic_word
mask2=random_word_0.3
for mode in train dev
do  
    input_file=${input_file_prefix}$mask1/${mode}_${mask2}_beam_1.txt 
    python ../models/filter_synthetic_data.py \
            --use_leven --leven_threshold 0.3 \
            --use_cls --cls_threshold 0.2 --min_prob 0.8 \
            --model_name_or_path ../checkpoints_verifier/roberta-base/fact_verification_data_seed42_lr2e-05_2-retrieved-evidence/ \
            --input_file $input_file
done


input_file_prefix=../checkpoints_corruptor/t5-base/seed42_lr4e-05_2-gold-evidence_
mask1=heuristic_word
mask2=random_word_0.3
for mode in train dev
do  
    input_file=${input_file_prefix}$mask1/${mode}_${mask2}_beam_1.txt 
    python ../models/filter_synthetic_data.py \
            --use_leven --leven_threshold 0.3 \
            --use_cls --cls_threshold 0.2 --min_prob 0.8 \
            --model_name_or_path ../checkpoints_verifier/roberta-base/fact_verification_data_seed42_lr2e-05_2-gold-evidence/ \
            --input_file $input_file
done
