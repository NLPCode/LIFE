# -*- coding: utf-8 -*-
# @Author  : He Xingwei
# @Time : 2023/5/8

"""
this scrit implements the seq2seq baselines, where the generator supports BART and T5. 
Encoder Input: origianl claim, evidence
Decoder output: revised the text
"""

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW, Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from transformers import  AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
# from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig, T5ForConditionalGeneration, T5BartTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
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
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
from tensorboardX import SummaryWriter
    
sys.path.append('../')
from utils.functions import set_seed, get_optimizer
from utils.text_process import maybe_format
# logger = logging.getLogger(__name__)
logger = logging.getLogger("__main__")

class Seq2SeqDataset(Dataset):
    """
    this class is used for loading training/validation/testing set for seq2seq models.
    """

    def __init__(self, filename, tokenizer, max_src_len=None, max_tgt_len=None, 
                 use_evidence=False, use_gold_evidence = False, num_evidence=None, 
                 use_mutation_type = False, source_prefix = None, 
                 dataset_percent = 1.0, num_data_instance = -1,
                 inference=False, start_idx = None, end_idx = None):
        """
        Args:
            filename (str): the name of the input file
            tokenizer (_type_): tokenizer
            max_src_len (int, optional): the maximum length of the source. Defaults to None.
            max_tgt_len (int, optional): the maximum length of the target. Defaults to None.
            use_evidence (bool, optional): whether use evidences to revise the original claim. Defaults to False.
            use_gold_evidence (bool, optional): whether use gold or retrieved evidences.
            num_evidence (int, optional): the number of evidence used to revise the original claim. Defaults to None.
            use_mutation_type (bool, optional): whether use the mutation type as input.
            source_prefix (str, optional): A prefix to add before every source text (useful for T5 models).
            dataset_percent (float): The percentage of data used to train the model.
            num_data_instance (int): The number of data instances used to train the model. -1 denotes using all data.
            inference (bool, optional): True means training mode, False means inference mode. Defaults to False.
            start_idx (int, optional): the start line of the input file, only effective when inference is True.
            end_idx (int, optional): the end line of the input file, only effective when inference is True.
        """
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.use_evidence = use_evidence 
        self.use_gold_evidence = use_gold_evidence
        self.num_evidence = num_evidence
        self.use_mutation_type = use_mutation_type
        self.source_prefix = source_prefix if source_prefix is not None else ""
        self.dataset_percent = dataset_percent
        self.num_data_instance = num_data_instance
        self.inference = inference
        self.start_idx = start_idx
        self.end_idx = end_idx

        print(f'Load source data from {self.filename}.')
        self.data_list = []
        with open(filename, 'r') as fr:
            for line in fr:
                instance = json.loads(line)
                if self.use_evidence:
                    if self.use_gold_evidence:
                        collected_evidence = instance['gold_evidence'][:self.num_evidence]
                    else:
                        collected_evidence = instance['retrieved_evidence'][:self.num_evidence]
                    collected_evidence = [maybe_format(title, content) for title, content in collected_evidence]
                    evidence = " ### ".join(collected_evidence)
                else:
                    evidence = None
                # print(evidence)
                # exit()
                data_instance = {
                    "src": instance["mutated"],
                    "tgt": instance["original"],
                    "evidence": evidence,
                    "mutation_type": instance["mutation"],
                    "veracity": instance["verdict"],
                }
                self.data_list.append(data_instance)
                
        if self.dataset_percent<1:
            self.num_data_instance = int(self.dataset_percent*len(self.data_list))
        
        if self.num_data_instance!=-1:
            self.data_list = self.data_list[:self.num_data_instance]
            print(f"Use {self.num_data_instance} to train the model.")
        
        if self.inference:
            self.data_list = self.data_list[self.start_idx:self.end_idx]
            
        self.len = len(self.data_list)
        
    def prepare_src(self, instance):
        if not self.use_evidence and not self.use_mutation_type:
            return instance["src"]

        src = "claim: " + instance["src"] 
        if instance["mutation_type"] is not None and self.use_mutation_type:
            src += " " + "mutation: " + instance["mutation_type"]
            
        if instance["evidence"] is not None and self.use_evidence:
            src += " " + "evidence: " + instance["evidence"]
        return src

    # def prepare_tgt(self, target, instance):
    #     if instance["mutation_type"] is not None and self.mutation_tgt:
    #         return (
    #             "mutation: " + instance["mutation_type"] + " " + "correction: " + target
    #         )

    #     return "correction: " + target
    
    def __getitem__(self, idx):
        data_instance = self.data_list[idx]
        src = self.source_prefix + self.prepare_src(data_instance)
        tgt = data_instance['tgt']
        
        src_tokenization = torch.tensor(self.tokenizer.encode(src, max_length=self.max_src_len, truncation=True, padding=False, add_special_tokens=True), 
                                        dtype=torch.long)
        if not self.inference:
            if 'T5Tokenizer' in self.tokenizer.__class__.__name__:
            # t5 has no bos_token_id, so we use pad_token_id (i.e., 0) as bos_token_id
            # please also refer to _shift_right in https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_t5.html#T5ForConditionalGeneration
                tgt_tokenization = torch.tensor([self.tokenizer.pad_token_id] + self.tokenizer.encode(tgt, max_length=self.max_tgt_len, truncation=True, padding=False, add_special_tokens=True), 
                                                dtype=torch.long)
            else:
                tgt_tokenization = torch.tensor(self.tokenizer.encode(tgt, max_length=self.max_tgt_len, truncation=True, padding=False, add_special_tokens=True), 
                                                dtype=torch.long)
        else:
            tgt_tokenization = None

        
        return {
                'src_tokenization': src_tokenization,
                'tgt_tokenization': tgt_tokenization,
                'idx': idx}

    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        encoder_input_list = [s['src_tokenization'] for s in samples]
        # Mask to avoid performing attention on padding token indices in encoder_inputs.
        _mask = pad_sequence(encoder_input_list, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)
        encoder_inputs = pad_sequence(encoder_input_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if self.inference:
            idx_list = [s['idx'] for s in samples]
            data = [self.data_list[idx] for idx in idx_list]
            return {"input_ids": encoder_inputs, 
                    "attention_mask": attention_mask, 
                    "data": data}

        decoder_input_list = [s['tgt_tokenization'][:-1] for s in samples]
        decoder_label_list = [s['tgt_tokenization'][1:] for s in samples]
        decoder_inputs = pad_sequence(decoder_input_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_labels = pad_sequence(decoder_label_list, batch_first=True, padding_value=-100)

        return {"input_ids": encoder_inputs,
                "attention_mask": attention_mask, 
                "decoder_input_ids": decoder_inputs, 
                "labels": decoder_labels}


def generate(model, tokenizer, test_dataloader, args):
    """
    Generate queries with the well-trained model.
    """
    if args.do_sample:
        if args.top_k>0:
            prefix = f'top_k_{args.top_k}'
        else:
            assert args.top_p<1
            prefix = f'top_p_{args.top_p}'
    else:
        prefix = f'beam_{args.num_beams}'
    
    if args.num_return_sequences>1: 
        prefix += f'_rs_{args.num_return_sequences}'
    if args.repetition_penalty>1:
        prefix += f'_rp_{args.repetition_penalty}'
    if args.temperature!=1: 
        prefix += f'_tp_{args.temperature}'
    model.eval()

    _id = 0
    if args.local_rank==-1:
        output_filename = os.path.join(args.output_dir, prefix + ".txt")
    else:
        output_filename = os.path.join(args.output_dir, prefix + f"_gpu_{args.local_rank}.txt")
    
    fw = open(output_filename, 'w', encoding='utf-8')

    with torch.no_grad():
        for batch_generator in tqdm(test_dataloader, desc="Generate", disable=args.local_rank not in [-1, 0]):
            
            f = model.module if hasattr(model, "module") else model
            outputs = f.generate(input_ids = batch_generator['input_ids'].to(args.device), 
                                 attention_mask = batch_generator['attention_mask'].to(args.device),
                                 max_length = args.max_tgt_len,
                                 num_beams = args.num_beams,
                                 do_sample  = args.do_sample,
                                 top_k = args.top_k,
                                 top_p = args.top_p,
                                 num_beam_groups = args.num_beam_groups,
                                 num_return_sequences = args.num_return_sequences,
                                 repetition_penalty = args.repetition_penalty,
                                 temperature = args.temperature)
            generated_example = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            assert len(generated_example)%args.num_return_sequences==0
            tmp_list = []
            example_list = []
            for i, example in enumerate(generated_example):
                tmp_list.append(example)
                if len(tmp_list) == args.num_return_sequences:
                    # tmp_list = sorted(tmp_list)
                    example_list.append(tmp_list)
                    tmp_list = []
            
            for data_instance, example in zip(batch_generator['data'], example_list):
                data_instance['generated_text'] = example
                fw.write(json.dumps(data_instance)+'\n')
            fw.flush()
    fw.close()
    if args.local_rank!=-1: 
        # merge the data from all gpus
        torch.distributed.barrier()
        if args.local_rank==0:
            with open(os.path.join(args.output_dir, prefix + ".txt"), 'w') as fw:
                for i in range(args.world_size):
                    output_filename = os.path.join(args.output_dir, prefix + f"_gpu_{i}.txt")
                    print(f'Load data from {output_filename}')
                    with open(output_filename, 'r', encoding='utf-8') as fr:
                        for line in fr:
                            fw.write(line)
                    os.remove(output_filename)
    logger.info(f"The generated text is saved at {os.path.join(args.output_dir, prefix + '.txt')}")
            

def predict(model, tokenizer, args):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)
    model.eval()
    with open(args.test_file, 'r') as fr:
        num_data_instances = len(fr.readlines())
    if args.local_rank!=-1:
        shard_size = num_data_instances // args.world_size
        start_idx = args.local_rank * shard_size
        end_idx = start_idx + shard_size
        if args.local_rank == args.world_size - 1:
            end_idx = num_data_instances
    else:
        start_idx = 0
        end_idx = num_data_instances
    
    test_dataset = Seq2SeqDataset(args.test_file, tokenizer,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                                    use_evidence=args.use_evidence, use_gold_evidence = args.use_gold_evidence, num_evidence=args.num_evidence, 
                                    use_mutation_type = args.use_mutation_type, source_prefix = args.source_prefix, 
                                    inference=True, start_idx=start_idx, end_idx=end_idx
                                )
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                collate_fn=test_dataset.create_mini_batch,
                                batch_size=args.per_device_eval_batch_size, 
                                num_workers=args.preprocessing_num_workers)
    generate(model, tokenizer, test_dataloader, args)


def evaluate_dev(args, model, dataloader):
    """
    compute the average loss over the test or validation set.
    :param args:
    :param model:
    :param dataloader:
    :return:
    """
    datasize = len(dataloader.dataset)
    model.eval()
    total_lm_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_generator in tqdm(dataloader, desc="Evaluate", disable=args.local_rank not in [-1, 0]):
            for k, v in batch_generator.items():
                batch_generator[k] = v.to(args.device)
            outputs = model(**batch_generator)
            decoder_labels = batch_generator['labels']
            
            lm_loss = outputs.loss
            bts = decoder_labels.shape[0]
            num_tokens = torch.sum(decoder_labels != -100)
            total_lm_loss += lm_loss * num_tokens
            total_tokens += num_tokens

        if args.local_rank in [-1, 0]:
            print()

        if args.local_rank != -1:
            torch.distributed.all_reduce(total_lm_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_tokens, op=torch.distributed.ReduceOp.SUM)

        average_lm_loss = total_lm_loss / total_tokens
        ave_loss = average_lm_loss.item()
    model.train()
    logger.info('Validation loss = %.3f.', ave_loss)
    return ave_loss
    
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
    model.eval()
    dev_dataset = Seq2SeqDataset(   args.validation_file, tokenizer,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                                    use_evidence=args.use_evidence, use_gold_evidence = args.use_gold_evidence, num_evidence=args.num_evidence, 
                                    use_mutation_type = args.use_mutation_type, source_prefix = args.source_prefix, 
                                    inference=False
                                )
    dev_sampler = SequentialSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler,
                                collate_fn=dev_dataset.create_mini_batch,
                                batch_size=args.per_device_eval_batch_size, 
                                num_workers=args.preprocessing_num_workers)

    dev_loss = evaluate_dev(args, model, dev_dataloader)
    return dev_loss


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
        pass
    
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

    train_dataset = Seq2SeqDataset( args.train_file, tokenizer,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                                    use_evidence=args.use_evidence, use_gold_evidence = args.use_gold_evidence, num_evidence=args.num_evidence, 
                                    use_mutation_type = args.use_mutation_type, source_prefix = args.source_prefix, 
                                    dataset_percent = args.dataset_percent, num_data_instance = args.num_data_instance,
                                    inference=False
                                )
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                collate_fn=train_dataset.create_mini_batch,
                                batch_size=args.per_device_train_batch_size, 
                                num_workers=args.preprocessing_num_workers)
    if args.validation_file is not None:
        dev_dataset = Seq2SeqDataset(   args.validation_file, tokenizer,
                                        max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                                        use_evidence=args.use_evidence, use_gold_evidence = args.use_gold_evidence, num_evidence=args.num_evidence, 
                                        use_mutation_type = args.use_mutation_type, source_prefix = args.source_prefix, 
                                        inference=False
                                    )
        dev_sampler = SequentialSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset, shuffle=False)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler,
                                    collate_fn=dev_dataset.create_mini_batch,
                                    batch_size=args.per_device_eval_batch_size, 
                                    num_workers=args.preprocessing_num_workers)

        dev_loss = evaluate_dev(args, model, dev_dataloader)
        best_dev_loss = dev_loss
    trigger_times = 0
    while global_step < args.max_steps:
        iter_count += 1
        if args.num_train_epochs >0 and iter_count > args.num_train_epochs:
            break
        if trigger_times >= args.patience:
            logger.info('Early stopping!')
            break
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch_generator in enumerate(epoch_iterator):
            for k, v in batch_generator.items():
                batch_generator[k] = v.to(args.device)

            outputs = model(**batch_generator)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

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
                    dev_loss = evaluate_dev(args, model, dev_dataloader)
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


def get_parameter():
    parser = argparse.ArgumentParser(description="Factual Error Correction.")
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run eval on the dev set.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predictions on the test set.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='seq2seq_data',
                        help='the path of the src and tgt data.')
    parser.add_argument('--train_file', type=str, default='../seq2seq_data/train.jsonl',
                        help='The input training data file (a jsonlines).')
    parser.add_argument('--validation_file', type=str, default=None,
                        help='An optional input evaluation data file to evaluate the metrics (nll loss) on a jsonlines file.')  
    parser.add_argument('--test_file', type=str, default='../seq2seq_data/test.jsonl',
                    help='An optional input test data file to evaluate the metrics (sari) on a jsonlines file.')  
    
    parser.add_argument('--dataset_percent', type=float, default=1,
                        help='The percentage of data used to train the model.')
    parser.add_argument('--num_data_instance', type=int, default=-1,
                        help='The number of data instances used to train the model. -1 denotes using all data.')

    parser.add_argument('--model_path', type=str, default='',
                    help='Path to pretrained model or model identifier from huggingface.co/models')
          
    parser.add_argument('--initialization', type=str, default='bart-base',
                        choices=["bart-random-base", 
                                 "facebook/bart-base", 
                                 "facebook/bart-large",         
                                 "t5-small",
                                 "t5-base",
                                 "t5-large",
                                 "t5-3b",
                                 "t5-11b"],
                        help='initialize the model with random values, bart or t5.')
    # hyper-paramters for training
    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help= "Number of updates steps to accumulate before performing a backward/update pass."
                        )
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

    parser.add_argument('--resume', action='store_true', help='whether load the best checkpoint or not.')
    # parameters for models
    parser.add_argument("--source_prefix", type=str, default=None, help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument('--max_src_len', type=int, default=512, help='the max length of the source text.')
    parser.add_argument('--max_tgt_len', type=int, default=256, help='the max length of the tgt text.')

    parser.add_argument('--use_evidence', action='store_true',
                        help='whether use evidences to revise the original claim.')
    parser.add_argument('--use_gold_evidence', action='store_true',
                    help='whether use gold or retrieved evidences.')
    parser.add_argument('--num_evidence', type=int, default=1,
                        help='the number of evidences used to revise the original claim.')
    
    parser.add_argument('--use_mutation_type', action='store_true',
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
    
    # parameters for decoding
    # Arguments will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.
    parser.add_argument('--num_beams', type=int, default=1,
                    help='Number of beams for beam search. 1 means no beam search.')
    parser.add_argument('--do_sample', action='store_true',
                    help='Whether or not to use sampling; use greedy/beam search decoding otherwise.')
    parser.add_argument('--top_k', type=int, default=0,
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
    parser.add_argument('--top_p', type=float, default=1.0,
                    help="If set to float < 1, only the most probable tokens with probabilities "
                          "that add up to `top_p` or higher are kept for generation.")

    parser.add_argument('--num_beam_groups', type=int, default=1,
                    help="Number of groups to divide num_beams into in order to ensure diversity among different groups of beams."
                         "This paper (https://arxiv.org/pdf/1610.02424.pdf) for more details.")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                    help="The number of independently computed returned sequences for each element in the batch.")
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                    help="The parameter for repetition penalty. 1.0 means no penalty. "
                        "See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.")
    parser.add_argument('--temperature', type=float, default=1.0,
                    help="The value used to module the next token probabilities.")
    
    parser.add_argument('--preprocessing_num_workers', type=int, default=5,
                help='The number of processes to use for the preprocessing.')
    # torch2.0 use 'local-rank', other versions use 'local_rank'
    parser.add_argument('--local-rank', type=int, default=-1)
    
    args = parser.parse_args()
    assert args.do_train + args.do_eval + args.do_predict == 1, print('Specify do_train, do_eval or do_predict.')
    
    
    if args.use_evidence:
        assert args.num_evidence > 0

    dir_prefix = f"{args.initialization.replace('/','-')}/seed{args.seed}_lr{args.lr}"

    if args.use_evidence:
        if args.use_gold_evidence:
            dir_prefix += f'_{args.num_evidence}-gold-evidence'
        else:
            dir_prefix += f'_{args.num_evidence}-retrieved-evidence'
    if args.use_mutation_type:
        dir_prefix += f'_use-mutation-type'

    if args.dataset_percent<1:
        dir_prefix += f'_{args.dataset_percent}-data-percent'
    if args.num_data_instance>0:
        dir_prefix += f'_{args.num_data_instance}-data-instance'
    assert args.dataset_percent==1 or args.num_data_instance==-1, print("Do not set both dataset_percent and num_data_instance.")
    
    if args.output_dir is None:
        if args.do_train:
            args.output_dir = f'../checkpoints_corrector/{dir_prefix}'
        else:
            args.output_dir = args.model_path
    args.tensorboard_dir = f'../tensorboard_log_corrector/{dir_prefix}'
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
    
    try:
        if args.resume:
            # load the pre-trained model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
            print(f'Initialize {model.__class__.__name__} from the checkpoint {args.model_path}.')
        else:
            raise ValueError('')
    except:
        args.resume = False
        if "bart-random-base" == args.initialization:
            tokenizer = AutoTokenizer.from_pretrained(f'facebook/bart-base')
            #  load pre-trained config
            config = AutoConfig.from_pretrained(f'facebook/bart-base')
            # pass the config to model constructor instead of from_pretrained
            # this creates the model as per the params in config
            # but with weights randomly initialized
            model = AutoModelForSeq2SeqLM.from_config(config)
            print(f'Randomly initialize {model.__class__.__name__}.')
        else:
            tokenizer = AutoTokenizer.from_pretrained(f'{args.initialization}')
            model = AutoModelForSeq2SeqLM.from_pretrained(f'{args.initialization}')
            print(f'Initialize {model.__class__.__name__} with default parameters {args.initialization}.')

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    return tokenizer, model

def main():
    args = get_parameter()
    set_env(args)
    tokenizer, model = load_model(args)
    
    if args.source_prefix is None and args.initialization in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    
    if args.do_train:
        logger.info("*** Train ***")
        logger.info("args:\n%s", '\n'.join([f'    {arg}={getattr(args, arg)}'  for arg in vars(args)]))
        global_step = train(model, tokenizer, args)
        logger.info(" global_step = %s", global_step)
        # if args.do_eval or args.do_predict:
        #     args.resume = True
        #     args.model_path = args.output_dir
        #     tokenizer, model = load_model(args)

    if args.do_eval:
        logger.info("*** Evaluate ***") 
        evaluate(model, tokenizer, args)

    if args.do_predict:
        logger.info("*** Predict ***")
        predict(model, tokenizer, args)
    

if __name__ == "__main__":
    main()

