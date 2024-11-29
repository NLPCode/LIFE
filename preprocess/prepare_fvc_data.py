"""
This script aims to prepare data for fact verification classifiers.

The data are created based on the the data in $DARA_DIR.
The mutated data in $DARA_DIR are either supported or refuted by the evidence. 
The following method is used to create NOTENOUGHINFO data:
    1. Replace the evidence in the mutated tuple (mutated, refuted, evidence) with another randomly sampled evidence.
"""
import os
import json
import random
import argparse
parser = argparse.ArgumentParser(description="Prepare data used to train the filter.")
parser.add_argument('--input_dir', type=str, help='input data directory.')
parser.add_argument('--output_dir', type=str, default="../fact_verification_data", help='Output data directory.')
args = parser.parse_args()

random.seed(42)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
# {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}

for mode in ['train.jsonl', 'dev.jsonl']:
    data_list = []
    retrieved_evidence_list = []
    gold_evidence_list = []
    with open(os.path.join(args.input_dir, mode), 'r') as fr:
        for line in fr:
            instance = json.loads(line)
            retrieved_evidence_list.append(instance['retrieved_evidence'])
            gold_evidence_list.append(instance['gold_evidence'])
            
    idx_list = range(len(gold_evidence_list))
    with open(os.path.join(args.input_dir, mode), 'r') as fr, open(os.path.join(args.output_dir, mode), 'w') as fw:
        for idx, line in enumerate(fr):
            instance = json.loads(line)
            fw.write(line)
            instance['verdict'] = "NOT ENOUGH INFO"
            sampled_idx = idx
            while sampled_idx==idx:
                sampled_idx = random.sample(idx_list, k=1)[0]
            instance['retrieved_evidence'] = retrieved_evidence_list[sampled_idx]
            instance['gold_evidence'] = gold_evidence_list[sampled_idx]
            fw.write(json.dumps(instance)+'\n')
            
        
          
    
        
        
        

