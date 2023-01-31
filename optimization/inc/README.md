# Test

This document is used to illustrate how to run the distillation for quantization examples.
These examples will take a NLP model fine tuned on the down stream task, use its copy as a teacher model, and do distillation during the process of quantization aware training.
For more informations of this algorithm, please refer to the paper [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/abs/2206.01861)

```bash
# install intel/neural-compressor from github
pip install git+https://github.com/intel/neural-compressor.git@0ca5db0bab21934d50ec9d75fea48255e0a267d1
```

```bash
python scripts/run_glue_no_trainer.py --task_name sst2 \
 --model_name_or_path yoshitomo-matsubara/bert-base-uncased-sst2  \
 --teacher_model_name_or_path yoshitomo-matsubara/bert-base-uncased-sst2  \
 --batch_size 64 \
 --do_eval \
 --do_quantization \
 --do_distillation \
 --pad_to_max_length \
 --num_train_epochs 9 \
 --output_dir test_zero
```

```bash
python scripts/run_glue_no_trainer.py --task_name sst2 \
 --model_name_or_path nreimers/MiniLMv2-L12-H384-distilled-from-RoBERTa-Large \
 --teacher_model_name_or_path howey/roberta-large-sst2 \
 --batch_size 32 \
 --do_eval \
 --do_quantization \
 --do_distillation \
 --pad_to_max_length \
 --num_train_epochs 9 \
 --output_dir test_zero
```


## TODO

install pypi version and fix improts