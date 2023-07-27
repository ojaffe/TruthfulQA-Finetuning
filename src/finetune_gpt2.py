import os
import argparse
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from torch.optim import AdamW
import torch.nn as nn
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, TaskType
import wandb

from preprocess_data import download_truthfulqa, generate_labeled_qa_pairs
from dataset import create_qa_dataloaders
from utils import set_seed, print_trainable_parameters


def train(gpt2_model, 
          batch_size, 
          lr, 
          epochs,
          seed,
          int8_training, 
          lora_training):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    # Data
    data_raw_path = "data/truthfulqa_raw.csv"
    data_processed_path = "data/truthfulqa_processed.csv"
    if not os.path.exists(data_processed_path):
        download_truthfulqa(data_raw_path)
        generate_labeled_qa_pairs(data_raw_path, data_processed_path)

    train_prop = 0.8
    shuffle = True

    # Model
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
    model = GPT2ForSequenceClassification.from_pretrained(gpt2_model, num_labels=2, load_in_8bit=int8_training)

    if not int8_training:
        model = model.to(device)

    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    if int8_training:
        model = prepare_model_for_int8_training(model)
        scaler = torch.cuda.amp.GradScaler()
    if lora_training:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)

    optimizer = AdamW(model.parameters(), lr=lr)
    train_loader, test_loader = create_qa_dataloaders(data_processed_path, tokenizer, train_prop, batch_size, shuffle)

    # Logging
    wandb_config = {
        "gpt2_model": gpt2_model,
        "batch size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "int8_training": int8_training,
        "lora_training": lora_training,
    }
    wandb.init(
        project="Finetuning-TruthfulQA-MemoryOptim",
        name=None,
        config=wandb_config
    )
    
    acc_every_batch = 50  # GLobal step window to calculate accuracy over
    eval_every_batch = 50  # How many global steps inbetween evaluating model
    save_every_epoch = 5  # No. epochs between model saving

    # Train model
    global_step = 0
    train_acc = []
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    for e in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            if int8_training:
                with torch.cuda.amp.autocast():
                    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                loss = output.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                loss = output.loss
                loss.backward()
                optimizer.step()
            
            # Metrics
            metrics = {"train/loss": loss}
            wandb.log(metrics)

            probs = torch.softmax(output.logits, dim=-1)
            top_tokens = torch.argmax(probs, dim=-1)
            batch_acc = sum(top_tokens == labels) / len(labels)
            train_acc.append(batch_acc)

            if global_step % acc_every_batch == 0 and global_step != 0:
                avg_acc = sum(train_acc) / len(train_acc)
                wandb.log({"train/acc": avg_acc})
                train_acc = []

            global_step += 1

            # Test loop
            if global_step % eval_every_batch == 0:
                model.eval()
                total_test_loss = 0
                test_acc = []
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids, attention_mask, labels = batch
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        labels = labels.to(device)

                        if int8_training:
                            with torch.cuda.amp.autocast():
                                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        else:
                            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                        loss = output.loss
                        total_test_loss += loss.item()

                        probs = torch.softmax(output.logits, dim=-1)
                        top_tokens = torch.argmax(probs, dim=-1)
                        batch_acc = sum(top_tokens == labels) / len(labels)
                        test_acc.append(batch_acc)

                avg_loss = total_test_loss / len(test_loader)
                avg_acc = sum(test_acc) / len(test_acc)
                metrics = {"test/loss": avg_loss, "test/acc": avg_acc}
                wandb.log(metrics)

                model.train()

        if e % save_every_epoch == 0:
            model_save_path = "models/{:}-model-finetuned-epoch{:}.pt".format(gpt2_model, e)
            torch.save(model.state_dict(), model_save_path)
            wandb.save(model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2_model", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--int8_training', action='store_true', default=False)
    parser.add_argument('--lora_training', action='store_true', default=False)
    args = parser.parse_args()
    train(gpt2_model=args.gpt2_model,
          batch_size=args.batch_size,
          lr=args.lr,
          epochs=args.epochs,
          seed=args.seed,
          int8_training=args.int8_training,
          lora_training=args.lora_training)
