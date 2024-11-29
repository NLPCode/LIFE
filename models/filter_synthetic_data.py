"""
This script aims to filter the generated data, based on the following rules:
1. the generated text is the same with the src text.
2. the levenshtein edit distance between the src text and the generated text exceeds the theshold. 
"""
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW, Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from transformers import  AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import argparse
import json
import os
import sys
import numpy as np
import logging
from tqdm import tqdm
import Levenshtein
sys.path.append('../')
from models.fact_verifier import FVDataset
from utils.functions import set_seed
from utils.text_process import maybe_format
logger = logging.getLogger("__main__")


class FVDataset(Dataset):
    """
    this class is used for loading training/validation/testing set for fact verification models.
    """
    def __init__(self, data_instance_list, tokenizer, max_src_len=None, max_tgt_len=None, 
                 use_gold_evidence = False, num_evidence=None):
        """
        Args:
            data_instance_list (list): the list of data instances
            tokenizer (_type_): tokenizer
            max_src_len (int, optional): the maximum length of the source. Defaults to None.
            max_tgt_len (int, optional): the maximum length of the target. Defaults to None.
            use_gold_evidence (bool, optional): whether use gold or retrieved evidences.
            num_evidence (int, optional): the number of evidence used to revise the original claim. Defaults to None.
        """
        self.data_instance_list = data_instance_list
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.use_gold_evidence = use_gold_evidence
        self.num_evidence = num_evidence
        self.data_list = []
        self.label_dict = {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}
        for line in self.data_instance_list:
            instance = json.loads(line)
            if self.use_gold_evidence:
                collected_evidence = instance['gold_evidence'][:self.num_evidence]
            else:
                collected_evidence = instance['retrieved_evidence'][:self.num_evidence]
            collected_evidence = [maybe_format(title, content) for title, content in collected_evidence]
            evidence = " ### ".join(collected_evidence)

            data_instance = {
                "claim": instance["mutated"], # the claim mutated with the destructor based on the orginal claim.
                "evidence": evidence
            }

            self.data_list.append(data_instance)
        
        print(f"There are {len(self.data_list)} data instances.")
        self.len = len(self.data_list)
        
    
    def __getitem__(self, idx):
        data_instance = self.data_list[idx]
        inputs = self.tokenizer(data_instance['claim'], data_instance['evidence'], 
                                max_length=self.max_src_len, truncation=True, 
                                padding=False, add_special_tokens=True, return_tensors='pt')
        

        return {
                'input_ids': inputs['input_ids'][0],
                'attention_mask': inputs['attention_mask'][0],
                'token_type_ids': inputs['token_type_ids'][0] if 'token_type_ids' in inputs else None,
                'idx': idx}

    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        input_ids = [s['input_ids'] for s in samples]
        attention_mask = [s['attention_mask'] for s in samples]
        if samples[0]['token_type_ids'] is None:
            token_type_ids = None
        else:
            token_type_ids = [s['token_type_ids'] for s in samples]
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        # Mask to avoid performing attention on padding token indices in encoder_inputs.
        _mask = pad_sequence(input_ids, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)
        
        idx_list = [s['idx'] for s in samples]
        if token_type_ids is not None:
            return {"input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask, 
                    "idx_list": idx_list}
        else:
            return {"input_ids": input_ids,
                    "attention_mask": attention_mask, 
                    "idx_list": idx_list}    

def classifier_filter(args, model, dataloader):
    """
    Filter out the generated_text if the it cannot supported or refuted by the evidence. 
    """
    # get the logits for all data in the dataloader
    datasize = len(dataloader.dataset)
    model.eval()
    logit_list = []
    prob_list = []
    idx_list = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluate", disable=args.local_rank not in [-1, 0]):
            for k, v in data.items():
                if k!='idx_list':
                    data[k] = v.to(args.device)
            idx_list += data['idx_list']
            del data["idx_list"]
            output = model(**data)
            logits = output.logits
            probs = torch.softmax(logits, dim=-1)
            prob_list += probs.tolist()
            logit_list += logits.tolist()

    assert len(np.unique(idx_list)) == len(idx_list) == len(logit_list)
    filtered_idx_list = []    
    for idx, prob in zip(idx_list, prob_list):
        # {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}
        if prob[2]<args.cls_threshold and max(prob[0], prob[1])>args.min_prob:
            filtered_idx_list.append(idx)
        else:
            pass
             
    return filtered_idx_list

def levenshtein_filter(src_text, generated_text, threshold):
    """
    Filter out the generated_text if the Levenshtein distance between it and the source text over the threshold. 
    """
    if src_text.strip() == generated_text.strip():
        return True
    if Levenshtein.distance(src_text, generated_text) / len(src_text) >threshold:
        return True 
    return False

def set_env(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = torch.distributed.get_world_size()

    args.device = device
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                  args.local_rank, device, args.n_gpu, bool(args.local_rank != -1)
    )

    if args.local_rank != -1:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will create the output dir.


def load_model(args):
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    tokenizer = AutoTokenizer.from_pretrained(f'{args.model_name_or_path}')
    model = AutoModelForSequenceClassification.from_pretrained(f'{args.model_name_or_path}', num_labels=3) # three classification, support, refute, not enough information
    logger.info(f'Initialize {model.__class__.__name__} with parameters from {args.model_name_or_path}.')

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    return tokenizer, model

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factual Error Correction.")
    parser.add_argument('--input_file', type=str, default='',
                        help='The input file which contains the generated data.')
    parser.add_argument('--seed', type=int, default=42)
    # parameters for levenshtein_filter
    parser.add_argument('--use_leven', action='store_true', 
                       help='whether use the levenstein filter.')
    parser.add_argument('--leven_threshold', type=float, default=1,
                        help='Remove the data instance, if the levenshtein edit distance between the src text and the generated text exceeds the theshold.')
    # parameters for classifier_filter
    parser.add_argument('--use_cls', action='store_true', 
                        help='whether use the claissifier filter.')
    parser.add_argument('--cls_threshold', type=float, default=1/3,
                        help='Remove the data instance, if the probilitiy for the "NOT ENOUGH INFO" class exceeds the theshold.')
    parser.add_argument('--min_prob', type=float, default=1/3,
                        help='Remove the data instance, if the max probilitiy for the "REFUTES" and "SUPPORTS" class less than the min_prob.')

    # parameters for gpu
    parser.add_argument('--preprocessing_num_workers', type=int, default=5,
                    help='The number of processes to use for the preprocessing.')
    # torch2.0 use 'local-rank', other versions use 'local_rank'
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128)
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base', 
                    help='Path to pretrained model [roberta-base, roberta-large, bert-base-cased, bert-large-cased] or model identifier from huggingface.co/models')
        
          
    # parameters for models
    parser.add_argument('--max_src_len', type=int, default=512, help='the max length of the source text.')
    parser.add_argument('--max_tgt_len', type=int, default=256, help='the max length of the tgt text.')


    parser.add_argument('--use_gold_evidence', type=str2bool, default=False, 
                    help='whether use gold or retrieved evidences.')
    parser.add_argument('--num_evidence', type=int, default=1,
                        help='the number of evidences used to revise the original claim.')
    
    parser.add_argument('--use_mutation_type', type=str2bool, default=False, 
                        help='whether use the mutation type as input.')
    
    args = parser.parse_args()
    
    # extract parameters from the input path
    if 'gold-evidence' in args.input_file:
        args.use_gold_evidence = True
        args.num_evidence = int(args.input_file.split('-gold-evidence')[0][-1])
        assert args.num_evidence > 0
    else:
        args.use_gold_evidence = False
        args.num_evidence = int(args.input_file.split('-retrieved-evidence')[0][-1])
    
    assert args.use_leven + args.use_cls >0
    if args.use_leven and not args.use_cls:
        args.output_file = os.path.dirname(args.input_file) + f"/leven{args.leven_threshold}_" + os.path.basename(args.input_file)
    elif not args.use_leven and args.use_cls:
        args.output_file = os.path.dirname(args.input_file) + f"/cls{args.cls_threshold}_{args.min_prob}_" + os.path.basename(args.input_file)
    else:
        args.output_file = os.path.dirname(args.input_file) + f"/leven{args.leven_threshold}_cls{args.cls_threshold}_{args.min_prob}_" + os.path.basename(args.input_file)
    print(args)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    data_instance_list = []
    with open(args.input_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            data_instance_list.append(line)
    num_total = 1.0*len(data_instance_list)
    
    if args.use_leven:
        filtered_data_instance_list = []
        num_filter = 0
        for line in data_instance_list:
            data_instance = json.loads(line)
            src_text = data_instance['original']
            generated_text = data_instance['mutated']
            if not levenshtein_filter(src_text, generated_text, args.leven_threshold):
                filtered_data_instance_list.append(line)
            else:
                num_filter += 1
        data_instance_list = filtered_data_instance_list
        logger.info(f"{num_filter} data instances are filtered and {len(data_instance_list)} are left by levenshtein_filter.")
        
        
    if args.use_cls:
        set_env(args)
        print("args:\n%s", '\n'.join([f'    {arg}={getattr(args, arg)}'  for arg in vars(args)]))
        tokenizer, model = load_model(args)
        dataset = FVDataset(data_instance_list, tokenizer, 
                                max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                                use_gold_evidence=args.use_gold_evidence, num_evidence=args.num_evidence
                                )
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(dataset, sampler=sampler,
                                    collate_fn=dataset.create_mini_batch,
                                    batch_size=args.per_device_eval_batch_size, 
                                    num_workers=args.preprocessing_num_workers)
        
        filtered_idx_list  = classifier_filter(args, model, dataloader)

        filtered_data_instance_list = []
        num_filter = len(data_instance_list) - len(filtered_idx_list)
        for idx in filtered_idx_list:
            line = data_instance_list[idx]
            filtered_data_instance_list.append(line)

        data_instance_list = filtered_data_instance_list

        logger.info(f"{num_filter} data instances are filtered and {len(data_instance_list)} are left by classifier_filter.")
        
    with open(args.output_file, 'w', encoding='utf-8') as fw:   
        for line in data_instance_list:
            fw.write(line)
    logger.info(f"{len(data_instance_list)}, {len(data_instance_list)/num_total*100:.3}% generated data instances are left.")