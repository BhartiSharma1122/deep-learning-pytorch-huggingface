# Test

This document is used to list steps of introducing [Prune Once For All](https://arxiv.org/abs/2111.05754) examples.

the pattern lock pruning, distillation and quantization aware training are performed simultaneously on the fine tuned model from stage 1 to obtain the quantized model with the same sparsity pattern as the pre-trained sparse language model.

The following example fine-tunes DistilBERT model of 90% sparsity on the sst-2 task through applying quantization aware-training, pattern lock pruning and distillation simultaneously.

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