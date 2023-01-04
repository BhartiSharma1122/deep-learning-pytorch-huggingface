# Test

This document is used to illustrate how to run the distillation for quantization examples.
These examples will take a NLP model fine tuned on the down stream task, use its copy as a teacher model, and do distillation during the process of quantization aware training.
For more informations of this algorithm, please refer to the paper [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/abs/2206.01861)

```bash
python scripts/run_glue.py \
    --model_name_or_path yoshitomo-matsubara/bert-base-uncased-sst2  \
    --task_name sst2 \
    --apply_distillation \
    --teacher_model_name_or_path yoshitomo-matsubara/bert-base-uncased-sst2  \
    --apply_quantization \
    --quantization_approach aware_training \
    --num_train_epochs 9 \
    --do_train \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/sst2_output
```