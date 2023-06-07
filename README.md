# ICLR2023 Gradient Dissipation
**The Offical Code for Our Paper：[On The Inadequacy of Optimizing Alignment and Uniformity in Contrastive Learning of Sentence Representations](https://openreview.net/forum?id=BAyYNpcYSga), ICLR2023**.

**Most of our experiments were done in the changed version of  [SimCSE](https://github.com/princeton-nlp/SimCSE)'s codebase, thanks for their great work!**

### Environment Configuration

- Create virtual environment and necessary dependence

	```bash
	conda create -n temp python=3.7
	pip install -r requirement.txt
	```

- Install torch 1.12.0 for suitable CUDA version (higher versions should also be compatible)

	```bash
	# CUDA 11.6
	pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
	# CUDA 11.3
	pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
	# CUDA 10.2
	pip install torch==1.12.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
	```

### Data Preparation

- Obtain 1m corpus for fine-tuning from the [link](https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt) provided by [SimCSE](https://github.com/princeton-nlp/SimCSE)
- Obtain the datasets of all downstream tasks in SentEval for evaluating from the official repository of [SentEval](https://github.com/facebookresearch/SentEval)
- Obtain pretrained [BERT-base](https://huggingface.co/bert-base-uncased), [BERT-large](https://huggingface.co/bert-large-uncased), [RoBERTa-base](https://huggingface.co/roberta-base) and [RoBERTa-large](https://huggingface.co/roberta-large) checkpoints provided by [Hugging Face](https://github.com/huggingface/transformers)

### Path Modification

- Change `TRAIN_FILE_PATH` in `run_unsup_example.sh` to the path of the pre-trained corpus
- Change `EVAL_FILE_PATH` in `run_unsup_example.sh` to the path of the eval_dataset
- Change `MODEL_PATH` in `run_unsup_example.sh` to the path of the pretrained model
- Change `PATH_TO_SENTEVAL` in `evaluation.py #17` and `trainer.py #85` to the path  of SentEval
- Change `PATH_TO_DATA` in `evaluation.py #18` and `trainer.py #86` to the path of the datasets in SentEval

### Training and Evaluation

- replicate the results of the met loss function on BERT-base

```bash
bash run_unsup_example.sh
```

- Edit the hyper-parameter in the shell to replicate the results of other experiments

### **Todo List**

- [x] The implementation of all loss functions
- [x] The codes based on SimCSE
- [ ] The checkpoints in the paper

### Citetion

```latex
@inproceedings{nie2023inadequacy,
  title={On The Inadequacy of Optimizing Alignment and Uniformity in Contrastive Learning of Sentence Representations},
  author={Nie, Zhijie and Zhang, Richong and Mao, Yongyi},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```

