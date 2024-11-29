# README
This repository contains the implementation of the AAAI 2024 paper: 
"[**Improving Factual Error Correction by Learning to Inject Factual Errors**](https://ojs.aaai.org/index.php/AAAI/article/view/29778)".
****
##  Abstract
Factual error correction (FEC) aims to revise factual errors in false claims with minimal editing, making them faithful to the provided evidence. This task is crucial for alleviating the hallucination problem encountered by large language models. Given the lack of paired data (i.e., false claims and their corresponding correct claims), existing methods typically adopt the 'mask-then-correct' paradigm. This paradigm relies solely on unpaired false claims and correct claims, thus being referred to as distantly supervised methods. These methods require a masker to explicitly identify factual errors within false claims before revising with a corrector. However, the absence of paired data to train the masker makes accurately pinpointing factual errors within claims challenging. To mitigate this, we propose to improve FEC by Learning to Inject Factual Errors (LIFE), a three-step distantly supervised method: 'mask-corrupt-correct'. Specifically, we first train a corruptor using the 'mask-then-corrupt' procedure, allowing it to deliberately introduce factual errors into correct text. The corruptor is then applied to correct claims, generating a substantial amount of paired data. After that, we filter out low-quality data, and use the remaining data to train a corrector. Notably, our corrector does not require a masker, thus circumventing the bottleneck associated with explicit factual error identification. Our experiments on a public dataset verify the effectiveness of LIFE in two key aspects: Firstly, it outperforms the previous best-performing distantly supervised method by a notable margin of 10.59 points in SARI Final (19.3% improvement). Secondly, even compared to ChatGPT prompted with in-context examples, LIFE achieves a superiority of 7.16 points in SARI Final.

****
## Requirements
python 3.10   
pip install torch==2.0.0+cu118  
pip install transformers==4.24.0  
pip install evaluate==0.4.0  
pip install tensorboardX==2.6  
pip install levenshtein==0.21.1  
pip install nltk==3.7  
pip install tqdm  
pip install tabulate  


git clone https://github.com/NVIDIA/apex  
cd apex  
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
****

## Dataset
The the evidence-based FEC dataset (FECDATA) used in our paper can be found at https://github.com/j6mes/acl2021-factual-error-correction. For convinence, we extract the gold evidence and retrieved evidence from FECDATA, and merge them into one file. Our processed data are available at https://drive.google.com/drive/folders/16H7WA8rZ-qlGNNIgylrPQox6qAdQq_qM?usp=sharing.

Before running our code, you should download the data, put them into the root directory of this project.

```bash
# Provide the data directory containing the downloaded data.
DATA_DIR=
```

****

## Train the supervised model for factual error correction
Fully supervised baselines estimate the ceiling performance of factual error correction models, under the assumption that a substantial amount of data is accessible. For this purpose, we fine-tune T5-base on FECDATA, where the encoder takes the false claim and corresponding evidence as inputs, while the decoder generates the revised claim.

```bash
cd scripts  
# revise factual errors with retrieved evidence
sh seq2seq_t5_baseline_retrieved_evidence.sh 
# revise factual errors with gold evidence
sh seq2seq_t5_baseline_gold_evidence.sh 
```

## Train our proposed LIFE for factual error correction

* Step 1: Create synthetic data for factual error correction.
  - 1: Corruptor training and inference
  ```bash
  cd scripts
  # train the corruptor with retrieved evidence 
  sh corruptor_retrieved_evidence.sh

  # train the corruptor with gold evidence 
  sh corruptor_gold_evidence.sh
  ```
  - 2: Synthetic Data Filtering
  ```bash
  # prepare data used to train the fact verification classifier-based filter
  cd preprocess
  python prepare_fvc_data.py --input_dir $DATA_DIR

  cd ..
  cd scripts 
  # train the fact verification filter with retrieved evidence
  sh fact_verifier_retrieved_evidence.sh

  # train the fact verification filter with gold evidence
  sh fact_verifier_gold_evidence.sh

  # synthetic data filtering
  sh filter.sh
  ```


* Step 2: Train the corrector on the synthetic data for factual error correction.
```bash
cd scripts 
# revise factual errors with retrieved evidence 
sh seq2seq_t5_syn_retrieved_evidence.sh

# revise factual errors with gold evidence
sh seq2seq_t5_syn_gold_evidence.sh
```

## Evaluation
Evaluate the output of the corrector with the following command:
```bash
python ../evaluation/evaluation.py \
  --input_file #file_name  \
  --output_file #file_name
```
'input_file' is the file containing the output of the corrector, 'output_file' is the file used to save the evaluation result.

## Try our model with the well-trained checkpoints 
| Model           |  Download link
|----------------------|--------|
| The corruptor of LIFE with retrieved evidence | [\[link\]](https://huggingface.co/He-Xingwei/LIFE-Corruptor-RE)  | 
| The corruptor of LIFE with gold evidence| [\[link\]](https://huggingface.co/He-Xingwei/LIFE-Corruptor-GE)  | 
| The corrector of LIFE with retrieved evidence | [\[link\]](https://huggingface.co/He-Xingwei/LIFE-Corrector-RE)  | 
| The corrector of LIFE with gold evidence| [\[link\]](https://huggingface.co/He-Xingwei/LIFE-Corrector-GE)  | 

Download the checkpoints, put them into the root directory.



## Citation
If you want to use this code in your research, please cite our [paper](https://ojs.aaai.org/index.php/AAAI/article/view/29778/):
```bash

@inproceedings{he2024improving,
  title={Improving Factual Error Correction by Learning to Inject Factual Errors},
  author={He, Xingwei and Zhang, Qianru and Jin, A-Long and Ma, Jun and Yuan, Yuan and Yiu, Siu Ming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={18197--18205},
  year={2024}
}

```
