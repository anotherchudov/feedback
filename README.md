
# Feedback Prize - Evaluating Student Writing
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

![](https://storage.googleapis.com/kaggle-media/competitions/The%20Learning%20Agency/Kaggle%20Description%20Image.png)



## 🚀 Getting Started

**Training LongFormer**

```python
python ???
```

## 🏠 Description

```python
feedback
├── kaggle_inference_notebooks # inference notebooks of each models for kaggle
│   ├── deberta         
│   ├── longformer     
│   ├── xlnet
│   └── ...                    # TODO: Add more models
│
├── models_training            # Test files (alternatively `spec` or `tests`)
│   ├── deberta                 
│   ├── longformer         
│   │   ├── longformer         # Original Longformer Code
│   │   │
│   │   ├── submission         # Group of codes for Logformer model submssion
│   │   │   ├── codes                   # Modified Longformer & Huggingface Code    
│   │   │   ├── pretrained_checkpoints  #         
│   │   │   ├── tokenizer               # 
│   │   │   └── weights                 #
│   │   │
│   │   ├── submission_large   # same as above `submission`
│   │   └── ...  
│   ├── xlnet                  
│   ├── ...                    # TODO: Add more models
│   │
│   ├── oof (out of fold)      #
│ 	└── post processing        #
│
├── train.csv
└── check_and_split_data.ipynb
```

### Data
In code directory **`check_and_split_data.ipynb` was used to make splits**. it is not deterministic due to [rapids umap](https://github.com/rapidsai/cuml), so produced splits also included in that folder.
`train.csv` is slightly cleaner version of public train file.

### Models
- `Longformers` - training scripts in longformer directory are deterministic, but slow
- `Deberta` - not deterministic, yet better results, faster training and faster submission as well.

### Models Training
- **Training scripts are in `models_training`**.
    - Includes some modified import codes in `feedback/models_training/longformer/submission` folder.
    - Training data for `longformer` and for `debertav1` is made by the **script in longformer folder**, as it was assumed that tokenizers are identical.
    - Also, when making that particular data, original `train.csv` was used.
- **`xlnet`** folder contains check_labels.ipynb which is used to sanity check produced data files. Also has a notebook to prepare training data.
- **`deberta`** folder has a notebook to make data for debertav3.

### Kaggle Submission
- submission notebooks in `code/kaggle_inference_notebooks`
- **submission time**
    - `longformer` - 40 minutes for 1 fold
    - `debertav1` - 22 minutes for 1 fold


## ✅ Things that worked
- Make sure `entities` start from an **alphanumeric** character
- class weights
- label smoothing
- global attention to `sep`/`cls` token and [.?!] tokens for longformer
- swa ( sliding windows version of )
- [reverse cross entropy](https://www.kaggle.com/c/feedback-prize-2021/discussion/306279)
- Tokenization of `xlnet` and `debertav3` preserves newlines, otherwise severe drop in performance

## ⛔️ Not worked
- mixup - briefly tried, looks like same results
- cleaning unicode artefacts in data with [ftfy](https://ftfy.readthedocs.io/en/latest/) and regex substitutions



## 🏅 Experiments

[assets]: https://github.com/ultralytics/yolov5/releases

[TTA]: https://github.com/ultralytics/yolov5/issues/303

|Model | Fold | Epochs | Training Time (h) | Val | CV | LB | Special note 
|---   |---   |---     |---                |---  |--- |--- |---      
|Xlnet     |5  | - | rtx3090 x 1 **19h**   | -| - | - | 
|Longformer     |5  | - | rtx3090 x 1 **19h30**  | -| 0.670 | - | with bug entity
|Debertav1      |5  | - | rtx3090 x 1 **13h**  | -| 0.678 | - | with bug entity
|Debertav1      |5  | - | rtx3090 x 1 **13h**   | -| 0.681 | - | partially fixed entity extraction
|Debertav1      |5  | - | rtx3090 x 1 **13h**   | -| 0.69724  | 0.699 | partially fixed entity extraction + adding filtering based on minimal number of words in predicted entity and some confidence thresholds
|Longformer + Debertav1      |5  | - | -   | -| 0.69945 | 0.700 | partially fixed entity extraction + adding filtering based on minimal number of words in predicted entity and some confidence thresholds


- **The code used to find thresholds was `ad-hoc`**, does not optimize correct metric
- The above models were **validated using the bugged entity extraction code**, so the models may be suboptimal.
- Training of xlnet looks deterministic
- **RAM** 
    - Training 4 xlnets in parallel takes 220gb of ram
    - 4 debertav1 barely fit in 256gb
    - 4 debertav3 will likely not fit


## 🎇 Further Work
- Finish training `xlnet` and train a `debertav3`
- Training one more transformer with adding predicted probability weighted embeddings of predicted token types to the word embeddings as a stacking model



## ❓ FAQ

#### Q : `../../data_rev1.csv` file used in `prepare_data_for_longformer_bkpv1.ipynb` (which makes train data for longformer and debertav1), the same file as `train.csv`?
---
Almost same, use train.csv

#### Q : What is bugged entity extraction?
---
```
labels format used was:
0- outside
1 - b-lead
2 - i-lead
3 - b-position
4 - i-position, etc.
```
When scanning the argmaxed prediction, new entity is started when an odd prediction is encountered when it's `0` or `prediction != current category + 1`.
bugged version had only check for odd number. you can see bugged version in train scipt of `longformer` and `debertav1`, function **`extract_entities`**
fixed version in train script of xlnet. **Fixed version checks for an odd prediction, when it's 0 or when prediction != current category + 1.**

i.e. if the prediction was `1 2 2 4 6 8 10 0 0 0 3 4 4 ...`
- old code would extract
```
entities:
1: [0 - 9, ...],
3: [10 - ...]
```
- new code would extract
```
entities:
1: [0 - 2, ...],
3: [10 - ...]
```

#### Q : Why does the performance is similar or better when `newline (\n)` is recognized in the deberta then longformer?
---
In the `longformer` the same tokenizer as in `roberta` is used. that one is also used for `debertav1`, and the tokenizer preserves newlines.
when using `xlnet` tokenizer or `debertav3` tokenizer, the newlines are gone.

**summary**
- `longformer` - `\n` toekn as newline
- `roberta` - `\n` token as newline
- `debertav1` - `\n` token as newline
- `Xlnet` - `<eop>` token as a newline
- `debertav3` - `[MASK]` token as a newline 

Overall `deberta` produces better results all models are trained with max_len 2048


#### Q : Have you tried to ensemble longformer with deberta?
---
Submission with `.700` score has longformer model as well


#### Q :  At longformer repo, name `tvm` was replaced by `te`. Why did you changed the name?
---
Note that `from tvm import te` is different from `import tvm as te`. Library namespace had changed. Few years ago in tvm variable was made with `tvm.var`, in latest release it is `tvm.te.var` but current longformer library still uses `tvm.var`.
`tvm.var` turned out useless as while taking less gpu ram for training it's also slower and not deterministic.

So to use `tvm.te.var` the following had been made
```python
# before
import tvm
b = tvm.var('b')

# after
from tvm import te
b = te.var('b')
```

#### Q :  Why did you modify the longformer attention code in the following link?
---

- [Modified longformer attention code part](https://github.com/anotherchudov/feedback/blob/ducky/models_training/longformer/longformer/longformer/longformer.py#L187-L188)

Other changes to that code ( some indexing modification and attention mask broadcasts ) were to make the code work with `torch.use_deterministic_algorithms(True)` to make training deterministic when using global attention.



## 🔗 Dependency
- [tvm](https://github.com/apache/tvm)
- [RAPIDS - cuML](https://github.com/rapidsai/cuml)