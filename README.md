Efficiently finetune GPT-2 on TruthfulQA with a single GPU. Uses [wandb](https://wandb.ai/) to log results.

Results on AWS g5.xlarge instance with batch size of one.

Model | Original | New |
| --- | --- | --- |
GPT-2 Small, 124M| 3.0GB | 0.7GB |
GPT-2 Medium, 335M | 8.4GB | 2.2GB |
GPT-2 Large, 774M | 18.7GB | 4.8GB |
GPT-2 XL, 1.5B | (Too big to test) | 9.8GB |

Example usage to finetune GPT-2 medium:
```
python3 src/finetune_gpt2.py --gpt2_model="gpt2-medium" --batch_size=16 --lr=5e-5 --epochs=20 --seed=42 --fp16 --int8_training --lora_training
```