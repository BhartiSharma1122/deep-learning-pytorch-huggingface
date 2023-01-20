import gc
import os
import sys
import threading
import argparse
import transformers
import datasets
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from datasets import load_from_disk

from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from tqdm import tqdm
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)

# Metric
metric = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to the already processed dataset.")
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    
    
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    return args
 
 
def training_function(args):
    # set seed 
    set_seed(args.seed)
    # Initialize accelerator with config from configs/accelerate_ds_z3.yaml
    accelerator = Accelerator()
    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()    
        
    # define LorA fine-tuning config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    # Create PEFT model with LoraConfig
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()    
    
    # load dataset from disk and tokenizer
    train_dataset = load_from_disk(os.path.join(args.dataset_path, "train"))
    eval_dataset = load_from_disk(os.path.join(args.dataset_path, "eval"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.epochs),
    )

    # Prepare model, optimizer and dataloaders
    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )

    # Add a progress bar to keep track of training.
    progress_bar = tqdm(range(args.epochs * len(train_dataloader)), disable=not accelerator.is_main_process)
    # Now we train the model
    for epoch in range(args.epochs):
        model.train()
        for _, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)


        for _, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                
                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        # print results
        accelerator.print(f"epoch {epoch}:", result)
        
        
    accelerator.wait_for_everyone()
    checkpoint_name = (
        f"{args.model_id}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace("/", "_")
    )
    accelerator.save(get_peft_model_state_dict(model, state_dict=accelerator.get_state_dict(model)), checkpoint_name)
    accelerator.wait_for_everyone()

def main():
    args = parse_arge()
    training_function(args)  

if __name__ == "__main__":
    main()