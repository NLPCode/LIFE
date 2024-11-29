# -*- coding: utf-8 -*-
# @Author  : He Xingwei
# @Time : 2023/7/10

"""
This script aims to train a fact verification model on the fever data. 
The input format is as follows: claim <sep> evidence

"""
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW, Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from transformers import  AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import os
import sys
import time
import argparse
import random
import json
import warnings
import logging
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm
import math
import nltk
from tensorboardX import SummaryWriter
    
sys.path.append('../')
from utils.functions import set_seed, get_optimizer
from utils.text_process import maybe_format
# logger = logging.getLogger(__name__)
logger = logging.getLogger("__main__")

class FVDataset(Dataset):
    """
    this class is used for loading the training/validation set for fact verification models.
    """
    def __init__(self, filename, tokenizer, max_src_len=None, max_tgt_len=None, 
                 use_gold_evidence = False, num_evidence=None, 
                 dataset_percent = 1.0, num_data_instance = -1
                 ):
        """
        Args:
            filename (str): the name of the input file
            tokenizer (_type_): tokenizer
            max_src_len (int, optional): the maximum length of the source. Defaults to None.
            max_tgt_len (int, optional): the maximum length of the target. Defaults to None.
            use_gold_evidence (bool, optional): whether use gold or retrieved evidences.
            num_evidence (int, optional): the number of evidence used to revise the original claim. Defaults to None.
            dataset_percent (float): The percentage of data used to train the model.
            num_data_instance (int): The number of data instances used to train the model. -1 denotes using all data.
        """
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.num_evidence = num_evidence
        self.dataset_percent = dataset_percent
        self.use_gold_evidence = use_gold_evidence
        self.num_data_instance = num_data_instance
        print(f'Load source data from {self.filename}.')
        self.data_list = []
        self.label_dict = {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}
        with open(filename, 'r') as fr:
            for line in fr:
                instance = json.loads(line)
                if 'pipeline_text' in instance: #fever_gold_data
                    collected_evidence = instance['pipeline_text'][:self.num_evidence]
                else:
                    if self.use_gold_evidence:
                        collected_evidence = instance['gold_evidence'][:self.num_evidence]
                    else:
                        collected_evidence = instance['retrieved_evidence'][:self.num_evidence]
                
                collected_evidence = [maybe_format(title, content) for title, content in collected_evidence]
                evidence = " ### ".join(collected_evidence)
                
                data_instance = {
                    "claim": instance["claim"] if "claim" in instance else instance["mutated"],
                    "evidence": evidence,
                    "label": self.label_dict[instance['label']] if 'label' in instance else self.label_dict[instance['verdict']],
                }

                self.data_list.append(data_instance)
        if self.dataset_percent<1:
            self.num_data_instance = int(self.dataset_percent*len(self.data_list))
            
        if self.num_data_instance!=-1:
            self.data_list = self.data_list[:self.num_data_instance]
            print(f"Use {self.num_data_instance} to train the model.")
        
        print(f"{filename} has {len(self.data_list)} data instances.")
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
                'label': data_instance['label'],
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
        
        labels = [s['label'] for s in samples]
        labels = torch.tensor(labels, dtype=torch.long)
        if token_type_ids is not None:
            return {"input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask, 
                    "labels": labels}
        else:
            return {"input_ids": input_ids,
                    "attention_mask": attention_mask, 
                    "labels": labels}             
    
def evaluate_dev(args, model, dataloader, num_labels=3):
    """
    compute the average loss over the test or validation set.
    :param args:
    :param model:
    :param dataloader:
    :param num_labels:
    :return:
    """
    datasize = len(dataloader.dataset)
    model.eval()
    total_loss = 0
    step = 0
    correct = 0
    corrects = [0.0] *num_labels
    recalls = [0.0] *num_labels
    precisions = [0.0] *num_labels
    f1s = [0.0] *num_labels

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluate", disable=args.local_rank not in [-1, 0]):
            for k, v in data.items():
                data[k] = v.to(args.device)
            output = model(**data)
            loss, logits = output.loss, output.logits
            values, predict_label = torch.max(logits, dim=1)
            correct += (predict_label == data['labels']).sum()
            bts = data['input_ids'].shape[0]
            total_loss += bts*loss
            step += bts
            for i in range(num_labels):
                corrects[i] += ((predict_label == i) & (data['labels'] == i)).sum()
                recalls[i] += (data['labels'] == i).sum()
                precisions[i] += (predict_label == i).sum()
        if args.local_rank != -1:
            torch.distributed.all_reduce_multigpu([total_loss])
            torch.distributed.all_reduce_multigpu([correct])

        # merge results
        for i in range(num_labels):
            if args.local_rank != -1:
                torch.distributed.all_reduce_multigpu([corrects[i]])
                torch.distributed.all_reduce_multigpu([recalls[i]])
                torch.distributed.all_reduce_multigpu([precisions[i]])
            corrects[i] = corrects[i].item()
            recalls[i] = recalls[i].item()
            precisions[i] = precisions[i].item()

        for i in range(num_labels):
            if recalls[i]!=0:
                recalls[i] = corrects[i]/recalls[i]
            else:
                recalls[i] = 0

            if precisions[i]!=0:
                precisions[i] = corrects[i]/precisions[i]
            else:
                precisions[i] = 0

            if precisions[i]!=0:
                f1s[i] = 2*recalls[i]*precisions[i]/(recalls[i]+precisions[i])
            else:
                f1s[i] = 0

        total_loss = total_loss.item()
        average_loss = total_loss/datasize
        accuracy = correct.item()*1.0/datasize
        if args.local_rank in [-1, 0]:
            print()
    model.train()
    results = { "average_loss": round(average_loss, 3), 
                "accuracy": round(accuracy, 3), 
                "recalls": [round(e, 3) for e in recalls], 
                "precisions": [round(e, 3) for e in precisions], 
                "f1s": [round(e, 3) for e in f1s]}  
    logger.info(json.dumps(results))
    return results

    
def evaluate(model, tokenizer, args):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], 
                                                          output_device=args.local_rank, find_unused_parameters=False)

    dev_dataset = FVDataset(args.validation_file, tokenizer, 
                            max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                            use_gold_evidence=args.use_gold_evidence, num_evidence=args.num_evidence
                            )
    dev_sampler = SequentialSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler,
                                collate_fn=dev_dataset.create_mini_batch,
                                batch_size=args.per_device_eval_batch_size, 
                                num_workers=args.preprocessing_num_workers)

    results = evaluate_dev(args, model, dev_dataloader, num_labels=3)
    return results

def train(model, tokenizer, args):
    """ Train the model """
    if args.warmup_ratio == 0 and args.warmup_steps == 0:
        logger.warning('You are training a model without using warmup.')
    elif args.warmup_ratio>0 and args.warmup_steps>0:
        raise ValueError("You can only specify either warmup_ratio or warmup_steps.")
    elif args.warmup_ratio>0:
        args.warmup_steps = int(args.warmup_ratio*args.max_steps)
        logger.info(f'warmup_steps is {args.warmup_steps}.')
    else:
        logger.info(f'warmup_steps is {args.warmup_steps}.')
    
    tb_writer = SummaryWriter(log_dir=args.tensorboard_dir) if args.local_rank in [-1, 0] else None

    optimizer = get_optimizer(args.optimizer, model, weight_decay=args.weight_decay, lr=args.lr, adam_epsilon=args.adam_epsilon)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        # from apex.parallel import DistributedDataParallel as DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
    logger.info("  Max steps = %d", args.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.per_device_train_batch_size * args.gradient_accumulation_steps 
        *(torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    total_loss = 0.0
    model.zero_grad()
    model.train()

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )
    # optimizer = AdamW(model.parameters(), lr=args.lr)  # the learning rate is linearly scales with the #gpu
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=0, verbose=True, min_lr=1e-6)
    
    global_step = 0
    iter_count = 0

    train_dataset = FVDataset(args.train_file, tokenizer,
                              max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                              use_gold_evidence=args.use_gold_evidence, num_evidence=args.num_evidence, 
                              dataset_percent = args.dataset_percent, num_data_instance = args.num_data_instance
                            )
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                collate_fn=train_dataset.create_mini_batch,
                                batch_size=args.per_device_train_batch_size, 
                                num_workers=args.preprocessing_num_workers)
    if args.validation_file is not None:
        dev_dataset = FVDataset(args.validation_file, tokenizer, 
                                max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                                use_gold_evidence=args.use_gold_evidence, num_evidence=args.num_evidence
                                )
        dev_sampler = SequentialSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset, shuffle=False)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler,
                                    collate_fn=dev_dataset.create_mini_batch,
                                    batch_size=args.per_device_eval_batch_size, 
                                    num_workers=args.preprocessing_num_workers)

        dev_results = evaluate_dev(args, model, dev_dataloader, num_labels=3)
        best_dev_loss = dev_results['average_loss']
    trigger_times = 0
    while global_step < args.max_steps:
        iter_count += 1
        if args.num_train_epochs >0 and iter_count > args.num_train_epochs:
            break
        if trigger_times >= args.patience:
            logger.info('Early stopping!')
            break
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, data in enumerate(epoch_iterator):
            for k, v in data.items():
                data[k] = v.to(args.device)

            output = model(**data)
            loss, logits = output.loss, output.logits
            # loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

            loss = loss / args.gradient_accumulation_steps
            total_loss += loss.item()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    loss_scalar = total_loss / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs = {}
                    logs["train_learning_rate"] = learning_rate_scalar
                    logs["train_nll_loss"] = loss_scalar
                    total_loss = 0
                    if args.local_rank in [-1, 0]:
                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        logger.info(json.dumps({**logs, **{"step": global_step}}))

                if args.save_steps > 0 and global_step % args.save_steps == 0 and args.validation_file is not None:
                    dev_results = evaluate_dev(args, model, dev_dataloader, num_labels=3)
                    dev_loss = dev_results['average_loss']
                    model.train()
                    if args.local_rank in [-1, 0]:
                        # _save_checkpoint(args, model, optimizer, scheduler, global_step)
                        tb_writer.add_scalar("dev_nll_loss", dev_loss, global_step)
                        if dev_loss<best_dev_loss:
                            logger.info('Save the model at {}.'.format(args.output_dir))
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(args.output_dir)
                            tokenizer.save_pretrained(args.output_dir)
                    
                    if dev_loss<best_dev_loss:
                        trigger_times = 0
                        best_dev_loss = dev_loss
                    else:
                        trigger_times += 1
                        logger.info(f'Trigger times: {trigger_times}.')

            if global_step >= args.max_steps:
                break
            if trigger_times >= args.patience:
                logger.info('Early stopping!')
                break
    if args.validation_file is None:
        logger.info('Save the model at {}.'.format(args.output_dir))
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    return global_step

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parameter():
    parser = argparse.ArgumentParser(description="Factual Error Correction.")
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run eval on the dev/test set.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='fever_gold_data',
                        help='the path of fact verification data.')
    parser.add_argument('--preprocessing_num_workers', type=int, default=5,
                        help='The number of processes to use for the preprocessing.')
    # torch2.0 use 'local-rank', other versions use 'local_rank'
    parser.add_argument('--local-rank', type=int, default=-1)
    
    parser.add_argument('--data_name', type=str, default='fact_verification_data', help='The name of dataset.')
    parser.add_argument('--train_file', type=str, default=None,
                        help='The input training data file (a jsonlines).')
    parser.add_argument('--validation_file', type=str, default=None,
                        help='An optional input evaluation data file to evaluate the metrics (accuracy) on a jsonlines file.')  
    parser.add_argument('--test_file', type=str, default=None,
                    help='An optional input test data file to evaluate the metrics (accuracy) on a jsonlines file.')  
    
    parser.add_argument('--dataset_percent', type=float, default=1,
                        help='The percentage of data used to train the model.')
    parser.add_argument('--num_data_instance', type=int, default=-1,
                        help='The number of data instances used to train the model. -1 denotes using all data.')

    parser.add_argument('--model_name_or_path', type=str, default='roberta-base', 
                        help='Path to pretrained model [roberta-base, roberta-large, bert-base-cased, bert-large-cased] or model identifier from huggingface.co/models')
          

    # hyper-paramters for training
    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help= "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--num_train_epochs', type=int, default=-1)
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='If > 0: set total number of training steps to perform. Override num_train_epochs.')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Linear warmup over warmup_steps.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Linear warmup over warmup_ratio fraction of total steps.')
    
    parser.add_argument('--optimizer', type=str, default='adamW', 
                        help='The optimizer to use.')
    parser.add_argument('--lr', type=float, default=4e-5, help='The initial learning rate for training.')
    # adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    # adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='Weight decay for AdamW if we apply some.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, 
                    help='Epsilon for AdamW optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                help='Max gradient norm.')
    
    parser.add_argument('--patience', type=int, default=2,
                        help='If the performance of model on the validation does not improve for n times, '
                             'we will stop training.')

   
    # parameters for models
    parser.add_argument('--max_src_len', type=int, default=512, help='the max length of the source text.')
    parser.add_argument('--max_tgt_len', type=int, default=256, help='the max length of the tgt text.')

    parser.add_argument('--use_evidence', type=str2bool, default=False, 
                        help='whether use evidences to revise the original claim.')
    parser.add_argument('--use_gold_evidence', type=str2bool, default=False, 
                    help='whether use gold or retrieved evidences.')
    parser.add_argument('--num_evidence', type=int, default=1,
                        help='the number of evidences used to revise the original claim.')
    
    parser.add_argument('--use_mutation_type', type=str2bool, default=False, 
                        help='whether use the mutation type as input.')
        
    # parameters for fp 16 
    parser.add_argument('--fp16', action='store_true',
                        help='whether to use fp16 (mixed) precision instead of 32-bit.')
    parser.add_argument('--fp16_opt_level', type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                         "See details at https://nvidia.github.io/apex/amp.html")
    # paramters for log
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=100,
                    help='Save checkpoint every X updates steps.')

    parser.add_argument('--tensorboard_dir', type=str, default="../tensorboard_log",
                        help="Tensorboard log dir.")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="dir for model checkpoints, logs and generated text.",
    )

    parser.add_argument('--resume', action='store_true', help='whether load the best checkpoint or not.')
    args = parser.parse_args()
    assert args.do_train + args.do_eval == 1, print('Specify do_train or do_eval.')
    if args.use_evidence:
        assert args.num_evidence > 0

    if args.resume:
        print(args.model_name_or_path)
        assert os.path.exists(args.model_name_or_path), print('Please provide the checkpoint.')
        args.output_dir = args.model_name_or_path
        args.tensorboard_dir = args.output_dir.replace("checkpoints_verifier", "tensorboard_log_verifier")
    else:
        dir_prefix = f"{args.model_name_or_path}/{args.data_name}_seed{args.seed}_lr{args.lr}"

        if args.use_evidence:
            if args.use_gold_evidence:
                dir_prefix += f'_{args.num_evidence}-gold-evidence'
            else:
                dir_prefix += f'_{args.num_evidence}-retrieved-evidence'
        
        if args.dataset_percent<1:
            dir_prefix += f'_{args.dataset_percent}-data-percent'
        if args.num_data_instance>0:
            dir_prefix += f'_{args.num_data_instance}-data-instance'
        assert args.dataset_percent==1 or args.num_data_instance==-1, print("Do not set both dataset_percent and num_data_instance.")
        
        if args.output_dir is None:
            args.output_dir = f'../checkpoints_verifier/{dir_prefix}'
        args.tensorboard_dir = f'../tensorboard_log_verifier/{dir_prefix}'
    args.log_file = f'{args.output_dir}/log.txt'
    return args

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
    if args.n_gpu==0:
        args.fp16=False
    args.device = device
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                  args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16
    )

    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Create output file
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
        
    if not os.path.exists(args.tensorboard_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.tensorboard_dir)

    if args.local_rank != -1:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will create the output dir.

    basic_format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    formatter = logging.Formatter(basic_format)
    
    # sh = logging.StreamHandler()
    handler = logging.FileHandler(args.log_file, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(sh)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print(logger)

def load_model(args):
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    tokenizer = AutoTokenizer.from_pretrained(f'{args.model_name_or_path}')
    model = AutoModelForSequenceClassification.from_pretrained(f'{args.model_name_or_path}', num_labels=3) # three classification, support, refute, not enough information
    print(f'Initialize {model.__class__.__name__} with parameters from {args.model_name_or_path}.')

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    return tokenizer, model

def main():
    args = get_parameter()
    set_env(args)
    tokenizer, model = load_model(args)
    
    if args.do_train:
        logger.info("*** Train ***")
        logger.info("args:\n%s", '\n'.join([f'    {arg}={getattr(args, arg)}'  for arg in vars(args)]))
        global_step = train(model, tokenizer, args)
        logger.info(" global_step = %s", global_step)

    if args.do_eval:
        logger.info("*** Evaluate ***") 
        print("args:\n%s", '\n'.join([f'    {arg}={getattr(args, arg)}'  for arg in vars(args)]))
        evaluate(model, tokenizer, args)

if __name__ == "__main__":
    main()