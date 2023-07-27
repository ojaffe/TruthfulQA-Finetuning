Efficient finetuning of huggingface GPT-2 models on TruthfulQA with a single GPU, using [LoRA](https://arxiv.org/abs/2106.09685) and eight bit quantization. We finetune a judge to classify whether a question is answered truthfully or not. Uses [wandb](https://wandb.ai/) to log results.

Results on the AWS g5.xlarge instance with batch size of 16:

Model | Original | New |
| --- | --- | --- |
GPT-2 Small, 124M| 3.0GB | 0.6GB |
GPT-2 Medium, 335M | 8.4GB | 1.6GB |
GPT-2 Large, 774M | 18.7GB | 3.5GB |
GPT-2 XL, 1.5B | (Too big to test) | 6.8GB |

Example usage to finetune GPT-2 medium:
```
python3 src/finetune_gpt2.py --gpt2_model="gpt2-medium" --batch_size=16 --lr=5e-5 --epochs=20 --seed=42 --int8_training --lora_training
```

We recommend using a batch size >=8 with this dataset for stable training.
