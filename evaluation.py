import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = '/home/LAB/niezj/unsupervised/SentEval'
PATH_TO_DATA = '/home/LAB/niezj/unsupervised/SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def Construct_Eval_Distribution_Dataset(datapath, tokenizer):
    datas = []
    with open(datapath) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            datas.append((line[5], line[6]))
        text_a = [ele[0] for ele in datas]
        text_b = [ele[1] for ele in datas]
        text = text_a + text_b
        text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        tensors = list(text.values())
        return TensorDataset(*tensors)

def Construct_Eval_Test_Dataset(datapath, tokenizer):
    datas = []
    with open(datapath) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            datas.append((line[5], line[6], float(line[4])))
    text_a = [ele[0] for ele in datas]
    text_b = [ele[1] for ele in datas]
    labels = [ele[2] for ele in datas]
    texta = tokenizer(text_a, padding=True, truncation=True, return_tensors='pt')
    textb = tokenizer(text_b, padding=True, truncation=True, return_tensors='pt')
    labels = torch.LongTensor(labels)
    tensors = list(texta.values()) + list(textb.values()) + [labels]
    return TensorDataset(*tensors)


def pooling(outputs, attention_mask, args=None, to_cpu=False):
    last_hidden = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    hidden_states = outputs.hidden_states

    if args is None:
        return_tensor = last_hidden[:, 0]
        if to_cpu:
            return return_tensor.cpu()
        else:
            return return_tensor

    # Apply different poolers
    if args.pooler == 'cls':
        # There is a linear+activation layer after CLS representation
        return_tensor = pooler_output
    elif args.pooler == 'cls_before_pooler':
        return_tensor = last_hidden[:, 0]
    elif args.pooler == "avg":
        return_tensor = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
    elif args.pooler == "avg_first_last":
        first_hidden = hidden_states[0]
        last_hidden = hidden_states[-1]
        pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            -1).unsqueeze(-1)
        return_tensor = pooled_result
    elif args.pooler == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
            1) / attention_mask.sum(-1).unsqueeze(-1)
        return_tensor = pooled_result
    else:
        raise NotImplementedError
    if to_cpu:
        return return_tensor.cpu()
    else:
        return return_tensor


def compute_kernel_bias(vecs, n_components=256):
    # 计算kernel和bias 最后的变换：y = (x + bias).dot(kernel)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s ** 0.5))
    W = np.linalg.inv(W.T)
    W = W[:, :n_components]
    return W, -mu


def cal_angle(model, dataloader, args=None):
    results = None
    labels = []
    for batch in dataloader:
        with torch.no_grad():
            batch = [ele.to(model.device) for ele in batch]
            if len(batch) == 5:
                outputs = model(input_ids=batch[0], attention_mask=batch[1], output_hidden_states=True,
                                return_dict=True)
                emba = pooling(outputs, batch[1], args, to_cpu=True)
                outputs = model(input_ids=batch[2], attention_mask=batch[3], output_hidden_states=True,
                                return_dict=True)
                embb = pooling(outputs, batch[3], args, to_cpu=True)
                label = batch[4].detach().cpu().numpy().tolist()
            else:
                outputs = model(input_ids=batch[0], token_type_ids=batch[1], attention_mask=batch[2],
                                output_hidden_states=True, return_dict=True)
                emba = pooling(outputs, batch[2], args, to_cpu=True)
                outputs = model(input_ids=batch[3], token_type_ids=batch[4], attention_mask=batch[5],
                                output_hidden_states=True, return_dict=True)
                embb = pooling(outputs, batch[5], args, to_cpu=True)
                label = batch[6].detach().cpu().numpy().tolist()
            emba = F.normalize(emba, dim=-1)
            embb = F.normalize(embb, dim=-1)
            cos_sim = torch.mm(emba, embb.permute(1, 0))
        if results is None:
            results = cos_sim
            labels += label
        elif cos_sim.shape[1] != 64:
            continue
        else:
            results = torch.cat((results, cos_sim), dim=0)
            labels += label

    p_a_list = []
    n_a_list = []
    for i, label in enumerate(labels):
        if label > 3.5:
            positive_angle = torch.arccos(results[i][i % 64] - 1e-5)
            negative_sum_angle = torch.arccos(results[i][:i % 64] - 1e-5).sum() + torch.arccos(
                results[i][i % 64 + 1:] - 1e-5).sum()
            print(results[i][:i % 64].shape[0] + results[i][i % 64 + 1:].shape[0])
            negative_avg_angle = negative_sum_angle / (results.shape[1] - 1)
            p_a_list.append(positive_angle.item())
            n_a_list.append(negative_avg_angle.item())
    # print(p_a_list)
    metrics = {'positive angle': (sum(p_a_list) / len(p_a_list)) / np.pi * 180,
               'negative angle': (sum(n_a_list) / len(n_a_list)) / np.pi * 180}
    return metrics


def cal_align_uniform(model, dataloader, args=None):
    results = []
    labels = []
    embs = []
    for batch in dataloader:
        with torch.no_grad():
            batch = [ele.to(model.device) for ele in batch]

            # print(batch[0].size(),batch[1].size(),batch[2].size())
            # print(batch[3].size(),batch[4].size(),batch[5].size())
            if len(batch) == 5:
                outputs = model(input_ids=batch[0], attention_mask=batch[1], output_hidden_states=True,
                                return_dict=True)
                emba = pooling(outputs, batch[1], args, to_cpu=True)
                # emba = outputs.pooler_output
                outputs = model(input_ids=batch[2], attention_mask=batch[3], output_hidden_states=True,
                                return_dict=True)
                embb = pooling(outputs, batch[3], args, to_cpu=True)
                label = batch[4].detach().cpu().numpy().tolist()
            else:
                outputs = model(input_ids=batch[0], token_type_ids=batch[1], attention_mask=batch[2],
                                output_hidden_states=True, return_dict=True)
                emba = pooling(outputs, batch[2], args, to_cpu=True)
                outputs = model(input_ids=batch[3], token_type_ids=batch[4], attention_mask=batch[5],
                                output_hidden_states=True, return_dict=True)
                embb = pooling(outputs, batch[5], args, to_cpu=True)
                label = batch[6].detach().cpu().numpy().tolist()
            emba = F.normalize(emba, dim=-1)
            embb = F.normalize(embb, dim=-1)
            scores = torch.linalg.norm(emba - embb, dim=-1).pow(2)
            embs.append(emba)
            embs.append(embb)
            labels += label
        scores = scores.detach().cpu().numpy().tolist()
        results += scores
    align = []
    embs = torch.cat(embs, dim=0)
    uniform = F.pdist(embs).pow(2).neg().exp().mean().log().item()
    for score, label in zip(results, labels):
        if label > 4.0:
            align.append(score)
    align = sum(align) / len(align)
    old_metrics = {'align': align, 'uniform': uniform}
    new_metrics = {'new_align': align, 'uniform': F.pdist(embs).pow(2).mul(-2).exp().mean().log().item()}
    return old_metrics, new_metrics

def record_distribution(model, dataloader, args=None):
    sim_list = []
    for index, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = [ele.to(model.device) for ele in batch]
            outputs = model(input_ids=batch[0], token_type_ids=batch[1], attention_mask=batch[2],
                            output_hidden_states=True, return_dict=True)
            embs = pooling(outputs, batch[2], args)
            embs = F.normalize(embs, dim=-1)
            if index == 0:
                anchor = embs[0].unsqueeze(0)
                sim = torch.mm(embs[1:], anchor.permute(1, 0))
                sim = sim.detach().cpu().numpy().tolist()
                print(sim[0])
                break



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str,
                        choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'],
                        default='cls',
                        help="Which pooler to use")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+',
                        default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                                 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                                 'SICKRelatedness', 'STSBenchmark'],
                        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    parser.add_argument("--eval_path", default="data/sts-train.tsv", help="eval path")
    parser.add_argument("--angle_cal", type=bool, default=False)
    parser.add_argument("--use_paper", type=bool, default=False)
    args = parser.parse_args()

    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    eval_dataset = Construct_Eval_Test_Dataset(args.eval_path, tokenizer)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=64)
    old_scores, new_scores = cal_align_uniform(model, eval_dataloader, args)
    print("old score:", old_scores)
    print("new score:", new_scores)

    if args.angle_cal:
        scores = cal_angle(model, eval_dataloader, args)
        print(scores)

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    else:
        return True

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)

        # Get raw embeddings
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states

        # Apply different poolers
        if args.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            return pooler_output.cpu()
        elif args.pooler == 'cls_before_pooler':
            return last_hidden[:, 0].cpu()
        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(
                -1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch[
                'attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / \
                            batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            raise NotImplementedError

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


if __name__ == "__main__":
    main()
