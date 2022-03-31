
# Feedback Prize - Evaluating Student Writing
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

## Training debertav1-xlarge scripts

* Make `train` directory and put training texts there.

* If training debertav2, download a transformers library that provides `DebertaV2TokenizerFast`. This [kaggle dataset](https://www.kaggle.com/datasets/sergeichudov/feedbackdebertav2tokenizer) uses code from https://github.com/mingboiz/transformers/tree/deberta-v2-fast-tokenizer


* Make dataset with `prepare_data_for_debertav1.ipynb` or/and `prepare_data_for_debertav2.ipynb`.
* Modify few lines on top of training script to set a wandb username and project and to add path to appropriate transformers library 
  * Train 5 folds with:
  
    ```for fold in {0..4}; do python train_debertav1_large.py $GPU_TO_USE $fold; done```
    
    or
    
    ```for fold in {0..4}; do python train_debertav2_large.py $GPU1_TO_USE,$GPU2_TO_USE $fold; done```
 ### Optional
 
* Collect oof predictions using `collect_debertav1_ps.ipynb` and/or `collect_debertav2_ps.ipynb`
* Use other branches to train the rest of the models and collect oof predictions from those in the `oof_ps` folder as in notebooks from previous step.
* Use `evaluate_models.ipynb` to find filter thresholds and expansions.
   
