from evaluate import load
import argparse
import json
import os
def load_data(filename):
    sources = []
    predictions = []
    references = []
    with open(filename, 'r') as fr:
        for line in fr:
            data_instance = json.loads(line)
            sources.append(data_instance['src'])
            references.append([data_instance['tgt']])
            if isinstance(data_instance['generated_text'], list) :
                predictions.append(data_instance['generated_text'][0])
            else:
                predictions.append(data_instance['generated_text'])
    return sources, predictions, references
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factual Error Correction.")
    parser.add_argument('--input_file', type=str, default='',
                        help='The input file for evaluation (a jsonlines).')
    parser.add_argument('--output_file', type=str,
                    help='The output file used to save the evaluation results.')
    
    args = parser.parse_args()

    sari = load("He-Xingwei/sari_metric")
    rouge = load('rouge')

    print(f'Evaluate {args.input_file}.')
    sources, predictions, references = load_data(args.input_file)
    assert len(sources) == 3882
    if not os.path.exists(args.output_file):
        with open(args.output_file, 'a') as fw:
            fw.write("Keep, Delete, Add, SARI, Rouge2, Filename\n")
    
    with open(args.output_file, 'a') as fw:
        results = sari.compute(sources=sources, predictions=predictions, references=references)
        results2 = rouge.compute(predictions=predictions, references=references)
        output = f"{results['keep']:.2f}, {results['del']:.2f}, {results['add']:.2f}, {results['sari']:.2f}, {100*results2['rouge2']:.2f}, {args.input_file}\n"
        fw.write(output)