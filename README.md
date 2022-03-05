# Feedback Prize - Evaluating Student Writing
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
> license is chosen based on the kaggle rules [Winner License Type: Open Source - MIT](https://www.kaggle.com/c/feedback-prize-2021/rules)


![](https://storage.googleapis.com/kaggle-media/competitions/The%20Learning%20Agency/Kaggle%20Description%20Image.png)



## 🚀 Getting Started

**What to do to train a model**
> Check `run-gpu0.ipynb` to run on single gpu. DDP multi-GPU will be supported soon

**What am I doing?**
- Code that I wrote will mostly be uploaded here often
- Experiment results could be seen on [Wandb](https://wandb.ai/ducky/feedback_deberta_large?workspace=user-ducky)

**Whatever that will help understanding**
> of the codebase and easily start based on it
```python
python lets_do_this
```

## 🏠 Description

```python
feedback
├── baselinev1
│   ├── data_files                      # necessary data files                 
│   ├── codes         
│   │   ├── longformer                  # Original Longformer Code
│   │   ├── ducky_transformers          # modified transformers code
│   │   └── new_transformers_branch     # for debertav3 fast tokeneizer
│   │
│   ├── model                           # TODO: Add more models 
│   ├── modules                         # All python files for pipeline 
│   │   ├── losses                      # Folder for loss functions
│   │   ├── dataset.py                  # 
│   │   ├── loss.py                     # 
│   │   ├── metric.py                   # 
│   │   ├── optimizer.py                # 
│   │   ├── scheduler.py                # 
│   │   ├── trainer.py                  # 
│   │   └── utils.py                    # 
│   ├── notebook                        # jupyter notebook collections
│   └── result                          # model weight storage
│
├── train.py                            # main python file to run training
└── training.ipynb                      # main notebook file to run training (Not completed)
```

### Models
- `DebertaV3` (Original Hugging Face library)
- `DebertaV3Ducky` (Under local `Ducky Transformer` library)
    - for seq len 512, DebertaV3 position bucket size is 256
    - for seq len 2048, position bucket size should be 384
    - but we are using 256 buckets still, so this model increase it to 384 for finetuning

## ✅ Things that worked
- initial learning rate 1e-5
- max gradient norm 1.0
- batch_size 4
- Plateau (patient=1)
    - never checked till the end, even middle
    - seems to have plenty of rooms for performance increase
- Only Cross-Entropy Loss
- SWA (stabilize valid performance, at least +0.01 boost)

## ⛔️ Not worked
- initial learning rate 3e-5
- gradient accumulation
- batch_size 1, 6, 8
- max gradient norm 10.0
- SAM Optimzier
- Dice Loss / Focal Loss with gamma 2.0

## 😥 Not sure
- [reverse cross entropy](https://www.kaggle.com/c/feedback-prize-2021/discussion/306279)

## 🎁 Didn't checked yet
- class weights
- label smoothing
- Make sure `entities` start from an **alphanumeric** character
- Making sure that tokenization of `xlnet` and `debertav3` preserves newlines, otherwise severe drop in performance
- global attention to `sep`/`cls` token and [.?!] tokens for longformer
- mixup - briefly tried, looks like same results
- cleaning unicode artefacts in data with [ftfy](https://ftfy.readthedocs.io/en/latest/) and regex substitutions
