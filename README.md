
# Feedback Prize - Evaluating Student Writing
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
> license is chosen based on the kaggle rules [Winner License Type: Open Source - MIT](https://www.kaggle.com/c/feedback-prize-2021/rules)


![](https://storage.googleapis.com/kaggle-media/competitions/The%20Learning%20Agency/Kaggle%20Description%20Image.png)



## 🚀 Getting Started

**What to do to train a model**
> Models are trained by modifying `val_fold` and `config` on top of `train_script.py` in corresponding folder and then running
```bash
# GPUN being a gpu number
python train_script.py GPUN &
```
Script assumes that there is a `checkpoints` directory in the same location

**Whatever that will help understanding**
> of the codebase and easily start based on it
```python
python lets_do_this
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
│   └── post processing        #
│
├── train.csv
└── check_and_split_data.ipynb
```

### Data
- **`check_and_split_data.ipynb` was used to make splits**.
    - it is not deterministic due to [rapids umap](https://github.com/rapidsai/cuml), so produced splits also included in that folder.
    - rapids umap code is mostly taken from [kaggle notebook - cdeotte/rapids-umap-tfidf-kmeans-discovers-15-topics](https://www.kaggle.com/cdeotte/rapids-umap-tfidf-kmeans-discovers-15-topics)
- **`train.csv`** is slightly cleaner version of public train file.
    - train.csv was made semi-manually after searching for entities where the symbol before first letter of discourse_text was alphanumeric.
    - Has several columns related to the `gt label`, hosts-provided target is `discourse_text`, what been scored is an overlap with `predictionstring`
    - **Those columns are all a `noisy target`**, `discourse_text` worked best in preliminary tests.
- **`data_rev1.csv`**
    - Made in similar process when looking for starts/ends of discourse_text split in `train.csv`
    - For samples where `discourse_text` starts 1 word before `punctuation mark` or ends 1 word after `punctuation mark`
    - `data_rev1.csv` was made with a script in `longformer` directory and new `train.csv` with the same as for debertav3 except for character replacement


### Models
- `Deberta` - not deterministic, yet better results, faster training and faster submission as well.
- `Longformers` - training scripts in longformer directory are deterministic, but slow
- `xlnet` - ...
- **TODO: Add more models**
    - other models with relative positional encoding are [ernie series from baidu](https://medium.com/syncedreview/baidus-knowledge-enhanced-ernie-3-0-7eb37bf098dd)
    - Longformer, BigBird, ETC, are **based on `roberta` checkpoints**

### Models Training
- **Training scripts are in `models_training`**.
    - Includes some modified import codes in `./models_training/longformer/submission` folder.
    - Training data for `longformer` and for `debertav1` is made by the **script in longformer folder**, as it was assumed that tokenizers are identical.
    - Also, when making that particular data, original `train.csv` was used.
- **`deberta`** folder
    - has a notebook to make data for debertav3.
- **`longformer`** folder
    - `./models_training/longformer/sumbission/codes/new_transformers_branch/transformers` is from [mingboiz/transformer](https://github.com/mingboiz/transformers/tree/deberta-v2-fast-tokenizer)
- **`xlnet`** folder
    - contains `check_labels.ipynb` which is used to sanity check produced data files.
    - Also has a notebook to prepare training data.


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
    - reverse cross entropy appears to have **speed up convergence, maybe reduce number of epochs to 7 or less**
- Making sure that tokenization of `xlnet` and `debertav3` preserves newlines, otherwise severe drop in performance

## ⛔️ Not worked
- mixup - briefly tried, looks like same results
- cleaning unicode artefacts in data with [ftfy](https://ftfy.readthedocs.io/en/latest/) and regex substitutions



## 🏅 Experiments

[assets]: https://github.com/ultralytics/yolov5/releases

[TTA]: https://github.com/ultralytics/yolov5/issues/303

|Model | Fold | Epochs | Training Time (h) | Val | CV | LB | Special note 
|---   |---   |---     |---                |---  |--- |--- |---      
|Xlnet     |5  | - | rtx3090 x 1 **19h**   | -| - | - | 
|Longformer     |5  | - | rtx3090 x 1 **19h30**  | -| - | 0.670 | with bug entity
|Debertav1      |5  | - | rtx3090 x 1 **13h**  | -| - | 0.678 | with bug entity
|Debertav1      |5  | - | rtx3090 x 1 **13h**   | -| - | 0.681 | partially fixed entity extraction
|Debertav1      |5  | - | rtx3090 x 1 **13h**   | -| 0.69724  | 0.699 | fixed entity extraction + adding filtering based on minimal number of words in predicted entity and some confidence thresholds
|Longformer + Debertav1      |5  | - | -   | -| 0.69945 | 0.700 | fixed entity extraction + adding filtering based on minimal number of words in predicted entity and some confidence thresholds


- **The code used to find thresholds was `ad-hoc`**, does not optimize correct metric
- The above models were **validated using the bugged entity extraction code**, so the models may be suboptimal.
- Training of xlnet looks deterministic
- **RAM** 
    - 4 xlnets in parallel training takes 220gb of ram
    - 4 debertav1 barely fit in 256gb
    - 4 debertav3 will likely not fit
- **Wandb Logs**
    - [Xlnet Large](https://wandb.ai/schudov/feedback_xlnet_large?workspace=user-schudov)
    - [DebertaV3 Large](https://wandb.ai/schudov/feedback_debertav3_large?workspace=user-schudov)


## 🎇 Further Work
- Finish training `xlnet` and train a `debertav3`
- Training one more transformer with adding predicted probability weighted embeddings of predicted token types to the word embeddings as a stacking model



## ❓ FAQ

#### Q : `../../data_rev1.csv` file used in `prepare_data_for_longformer_bkpv1.ipynb` (which makes train data for longformer and debertav1), the same file as `train.csv`?
Almost same, use train.csv

#### Q : What is bugged entity extraction?
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
Submission with `.700` score has longformer model as well


#### Q :  At longformer repo, name `tvm` was replaced by `te`. Why did you changed the name?
Note that `from tvm import te` is different from `import tvm as te`. Library namespace had changed. Few years ago in tvm variable was made with `tvm.var`, in latest release it is `tvm.te.var` but current longformer library still uses `tvm.var`.

`tvm.var` turned out useless.
- Custom gpu kernel turned out useless as while taking less gpu ram for training it's also slower and not deterministic.
- That file is needed to build and compile custom gpu kernel


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
- [./models_training/longformer/longformer/longformer/longformer.py#L187-L188](https://github.com/anotherchudov/feedback/blob/ducky/models_training/longformer/longformer/longformer/longformer.py#L187-L188)
- [./models_training/longformer/longformer/longformer/longformer.py#L263-L264](https://github.com/anotherchudov/feedback/blob/ducky/models_training/longformer/longformer/longformer/longformer.py#L263-L264)

Other changes to that code ( some indexing modification and attention mask broadcasts ) were to make the code work with `torch.use_deterministic_algorithms(True)` to make training deterministic when using global attention. Also there is a crucial semicolon on line 264.



## 🔗 Dependency
- [tvm](https://github.com/apache/tvm)
- [RAPIDS - cuML](https://github.com/rapidsai/cuml)
