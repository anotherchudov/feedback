
# Feedback Prize - Evaluating Student Writing
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

## Training debertav1-xlarge scripts

* Make `input` directory and put training texts there.
```bash
input
├── test
└── train
└── train.csv
└── sample_submission.csv
```

* If training debertav1, download a transformers library that provides `DebertaV2TokenizerFast`. This [kaggle dataset](https://www.kaggle.com/datasets/sergeichudov/feedbackdebertav2tokenizer) uses code from https://github.com/mingboiz/transformers/tree/deberta-v2-fast-tokenizer


* Make dataset with `models_training/deberta/prepare_data_for_debertav1.ipynb`.
* Modify few lines on top of training script to set a wandb username and project and to add path to appropriate transformers library
  * Train 5 folds with:
  
    ```for fold in {0..4}; do python models_training/deberta/train_script_debertav1_xlarge.py 0 $fold; done```
    
* Then here are models: `models_training/deberta/ckpt/`

* Use above model to create [kaggle notebook](https://www.kaggle.com/code/sergeichudov/8th-place-inference-notebook?scriptVersionId=90185474)
   
