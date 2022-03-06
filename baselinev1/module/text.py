

"""All about text processing for dataset """

import os
import os.path as osp
import sys

import re
import pickle
import numpy as np
import pandas as pd
from collections import deque
from tqdm.auto import tqdm

sys.path.append('./codes/new_transformers_branch/transformers/src')

from new_transformers import DebertaV2TokenizerFast
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForTokenClassification


class TextAugmenter:
    """Text Helper for Online Data Augmentation

    Everything is controlled by `text_id`

    - [ Initializing Step ]
        1. text is preprocessed (text -> list)
            - strip
            - newline replacing
        2. create tokenizer
        3. change text to list

    - [ Data Online Augmentation and Data Loading Step ]
        1. augmentation & noise injection (list -> text)
            - one char changing
            - one char removing
            - back translation
            - [ text augmentation ](https://www.kaggle.com/c/feedback-prize-2021/discussion/295277)
        2. change list to text
            - when heavy augmentaion like backtranslation has been applied
              the text dataframe will be recreated with the new text
        3. calculate entity boundaries
            - use boundary supported by train.csv
            - alternatively could calculate entity boundary with noise elimination
                - assuring discourse text start with alphanumeric character
        4. tokenize the text
            - token id
            - mask
            - offset
        5. calculate the label by using `entity boundary` and `offset` 
    
    """
    def __init__(self, args, text_ids, all_texts, df, verbose=True):
        """Initialize the Text Augmenter
        
        1. text is preprocessed (text -> list)
            - strip
            - newline replacing
        2. create tokenizer
        3. change text to list
         
        Args:
            args (argparse.Namespace): the arguments
            text_ids (list): the list of text_id
            all_texts (dict): the dictionary of all texts
            df (pandas.DataFrame): the dataframe file for each text
            verbose (bool): whether to print the information
        """
        self.verbose = verbose
        if self.verbose:
            print('[ Text Augmenter ] Initializing')

        # TODO: Add the logging for the augmentation status

        self.args = args
        self.text_ids = text_ids
        self.all_texts = all_texts
        self.df = df

        self.initialize_helper()
        self.initialize_regexp()
        self.initialize_cache()
        self.tokenizer = self.initialize_tokenizer(self.args, max_len=2048)
        self.data_dict = self.initialize_data(self.text_ids, self.all_texts, self.df)

    def get_label(self, text_id, cache=False):
        """Main function to get the label

        - Process
            1. augmentation & noise injection (list -> text)
                - one char changing
                - one char removing
                - back translation
                - [ text augmentation ](https://www.kaggle.com/c/feedback-prize-2021/discussion/295277)
            2. change list to text
                - when heavy augmentaion like backtranslation has been applied
                the text dataframe will be recreated with the new text
            3. calculate entity boundaries
                - use boundary supported by train.csv
                - alternatively could calculate entity boundary with noise elimination
                    - assuring discourse text start with alphanumeric character
            4. tokenize the text
                - token id
                - mask
                - offset
            5. calculate the label by using `entity boundary` and `offset`
        """
        if cache:
            if text_id in self.label_cache:
                return self.label_cache[text_id]

        text_list = self.data_dict[text_id]['text_list']
        text_df = self.data_dict[text_id]['text_df']

        # 1. augmentation
        text_list = self.text_augmentation(text_id, text_list)

        # 2. convert list to text
        # TODO: Automatically set the `revise_df` depends on the args setting
        text, text_df = self.list2text(text_list, text_df, revise_df=False)
        
        # 3. get the discourse boundary - text_id is used for cache
        # TODO: Automatically set the `cache` depends on the args setting
        
        # disable the cache when label cache is operating
        discourse_boundary_cache = False if cache else True
        discourse_boundary = self.get_discourse_boundary(text_id, text, text_df, revise=True, cache=discourse_boundary_cache)

        # 4. tokenize the text
        # TODO: truncate token length that exceeds max_len
        tokenizer_outs = self.tokenizer(text, return_offsets_mapping=True)

        # 5. calculate the label
        """labels are consisted with 4 types of data
        1. tokens (max_len)
        2. token labels (max_len, num_labels)
        3. attention mask (max_len)
        4. number of tokens (1)

        label = (token, token_label, attention_mask, num_tokens)
        """
        label = self.token_labeling(text, tokenizer_outs, discourse_boundary, self.cat2id, max_len=2048)

        # store the result to cache
        if cache:
            self.label_cache[text_id] = label

        return label

    def text_augmentation(self, text_id, text_list):
        """Text Augmentation
        
        - noise injection
            - char replacement
            - char remove
        - back translation
            - translate to chinese
            - translate to english
        - grammer correction
        - text augmentation
            - text augmentation
        
        Args:
            text_id (str): the text id for cache
            text_list (list): the text list
        
        Returns:
            text_list (list): the text list
        """
        # TODO: Automatically set the `cache` depends on the args setting
        text_list = self.noise_injection(text_id, text_list, cache=False) if self.args.noise_injection else text_list
        text_list = self.back_translation(text_id, text_list, cache=False) if self.args.back_translation else text_list
        text_list = self.grammer_correction(text_id, text_list, cache=False) if self.args.grammer_correction else text_list
        text_list = self.text_augmentationv1(text_id, text_list, cache=False)

        return text_list

    def noise_injection(self, text_id, text_list, cache=False):
        """Noise Injection
        
        - char replacement
        - char remove
            - currently will not use it due to the performance issue
        
        Args:
            text_id (str): the text id for cache
            text_list (list): the text list
            cache (bool): whether to use cache
        
        Returns:
            text_list (list): the text list
        """
        raise NotImplementedError

        if cache:
            if text_id in self.noise_cache:
                return self.noise_cache[text_id]

        if cache:
            self.noise_cache[text_id] = text_list
        return text_list

    def back_translation(self, text_id, text_list, cache=False):
        """Back Translation
        
        - translate to chinese
        - translate to english
        
        Args:
            text_id (str): the text id for cache
            text_list (list): the text list
            cache (bool): whether to use cache
        
        Returns:
            text_list (list): the text list
        """
        raise NotImplementedError

        if cache:
            if text_id in self.back_translation_cache:
                return self.back_translation_cache[text_id]

        if cache:
            self.back_translation_cache[text_id] = text_list

        return text_list

    def grammer_correction(self, text_id, text_list, cache=False):
        """Grammer Correction
        
        - grammer correction
        
        Args:
            text_id (str): the text id for cache
            text_list (list): the text list
            cache (bool): whether to use cache
        
        Returns:
            text_list (list): the text list
        """
        raise NotImplementedError
        if cache:
            if text_id in self.grammer_correction_cache:
                return self.grammer_correction_cache[text_id]
        
        if cache:
            self.grammer_correction_cache[text_id] = text_list

        return text_list

    def text_augmentationv1(self, text_id, text_list, cache=False):
        """Text Augmentation
        
        - text augmentation
        Whatever you want :)
        
        Args:
            text_id (str): the text id for cache
            text_list (list): the text list
            cache (bool): whether to use cache
        
        Returns:
            text_list (list): the text list
        """
        if cache:
            if text_id in self.text_augmentationv1_cache:
                return self.text_augmentationv1_cache[text_id]

        if cache:
            self.text_augmentationv1_cache[text_id] = text_list
            
        return text_list

    def save_cache(self, cache, save_path):
        """Save the cache
        
        Args:
            cache (dict): the cache you want to save
            save_path (str): the path to save the cache
        """
        if self.verbose:
            print('[ Text Augmenter ] Saving cache...')

        with open(save_path, 'wb') as f:
            pickle.dump(cache, f)

    def load_cache(self, load_path):
        """Load the cache
        
        Args:
            load_path (str): the path to load the cache
        Returns:
            cache (dict): the cache you want to load
        """
        if self.verbose:
            print('[ Text Augmenter ] Loading cache...')

        with open(load_path, 'rb') as f:
            cache = pickle.load(f)

        return cache

    def initialize_helper(self):
        """Initialize Helper """
        self.colors = {
            'Lead': '#8000ff',
            'Position': '#2b7ff6',
            'Evidence': '#2adddd',
            'Claim': '#80ffb4',
            'Concluding Statement': 'd4dd80',
            'Counterclaim': '#ff8042',
            'Rebuttal': '#ff0000'
         }
        self.cat2id = dict(zip(self.colors, range(1, 2 * len(self.colors), 2)))

    def initialize_regexp(self):
        """Initialize the regexp"""
        self.alphanumeric_re = re.compile('[0-9a-zA-z]')

    def initialize_cache(self):
        """Initialize the cache"""
        self.label_cache = {}
        self.discourse_boundary_cache = {}

        # TODO: Whatever cache you want - backtranslation, noise injection, etc.
        self.noise_injection_cache = {}
        self.back_translation_cache = {}
        self.grammer_correction_cache = {}
        self.text_augmentationv1_cache = {}

    def initialize_tokenizer(self, args, max_len=2048):
        """Initialize the tokenizer
        
        Args:
            args (argparse.Namespace): the arguments
            max_len (int): the maximum length of the text
        
        Returns:
            tokenizer (Tokenizer): the tokenizer
        """
        if self.verbose:
            print('[ Text Augmenter ] Initializing tokenizer...')

        if args.model in ['microsoft/deberta-v3-large',
                          'microsoft/deberta-v3-large-ducky']:
            tokenizer = DebertaV2TokenizerFast.from_pretrained('microsoft/deberta-v3-large')
            tokenizer.model_max_length = max_len
        else:
            raise Exception(f'{args.model} is not supported at the moment')

        return tokenizer

    def initialize_data(self, text_ids, all_texts, df):
        """Initialize the text data
        
        # TODO: currenty not supported
        preprocess includes the following
        - strip
        - newline exchange 

        Args:
            text_ids (str): all the train text ids
            all_texts (dict): all the train texts
            df (pandas.DataFrame): the dataframe file for all train meta data
        
        Returns:
            data_dict (dict): a dictionary that stores all the data for each text_id
                              {
                              ...
                              text_id: {'text_list': text_list, 'text_df': text_df},
                              text_id: {'text_list': text_list, 'text_df': text_df}
                              text_id: {'text_list': text_list, 'text_df': text_df}
                              ...
                              }
        """
        if self.verbose:
            print('[ Text Augmenter ] Initializing data...')

        df = df.copy()

        data_dict = {}
        for text_id in tqdm(text_ids, total=len(text_ids)):
            # load data
            text = all_texts[text_id]
            text_df = df.query('id == @text_id').reset_index(drop=True).copy()
            
            # convert to list and clean the text_df
            text_list, text_df = self.text2list(text, text_df, clean_text_df=True)

            # save as dictionary format
            data_dict[text_id] = {}
            data_dict[text_id]['text_list'] = text_list
            data_dict[text_id]['text_df'] = text_df

        return data_dict

    def text2list(self, text, text_df, clean_text_df=True):
        """Convert the text to list
        This is mainly to work on data augmentation and noise injection
        
        I'm working now quark! -> [[Lead, I'm working"],
                                   [Nonez, " "],
                                   [Claim, "now quark!"]]
        
        Args:
            text (str): literally the text of each text_id returns
            text_df (pandas.DataFrame): the dataframe file for each text
            clean_text_df (bool): text files and discourse_text in train.csv file doesn't match
                                fix the text to which is stored in the "{text_id}.txt" files
            
        Returns:
            text_list (list): list that stores the divided text and category of each text
            text_df (pandas.DataFrame): the dataframe file for each text

        """
        text_df = text_df.copy()
        
        text_list = []
        first_sentence = True
        last_end_idx = 0
        for row in text_df.itertuples():
            start_idx = int(row.discourse_start)
            end_idx = int(row.discourse_end)
            cat = row.discourse_type

            # the first sentence that will stored in the list
            if first_sentence:
                # when the first sentence is not the entity
                # 1. store the first sentence with none entity
                # 2. store the entity sentence
                if start_idx != 0:
                    text_list.append(["None", text[:start_idx]])

                # save the entity
                text_list.append([cat, text[start_idx:end_idx]])
                first_sentence = False
                last_end_idx = end_idx
            else:
                # when there is a middle sentence save it also
                if last_end_idx != start_idx:
                    middle_text = text[last_end_idx:start_idx]
                    text_list.append(["None", middle_text])

                # save the entity
                text_list.append([cat, text[start_idx:end_idx]])
                last_end_idx = end_idx

        # when there is sentence left store it
        text_len = len(text)
        if last_end_idx != text_len:
            last_text = text[last_end_idx:text_len]
            text_list.append(["None", last_text])
            
        if clean_text_df:
            discourse_texts = []
            for discourse_type, discourse_text in text_list:
                if discourse_type != 'None':
                    discourse_texts.append(discourse_text)
                    
            text_df.loc[text_df.index, 'discourse_text'] = discourse_texts
            
        return text_list, text_df

    def list2text(self, text_list, text_df, revise_df=False):
        """Convert the text to list
        Convert the list to text after data augmentation and noise injection
        
        [[Lead, I'm working"],
        [None, " "],
        [Claim, "now quark!"]]
        -> I'm working now quark!
        Args:
            text_list (list): list that stores the divided text and category of each text
            text_df (pandas.DataFrame): the dataframe file for each text
            revise_df (bool): If the augmentation is hard enough
                            to change the word or sentence different from original
                            recalculate the text_df totally with prediction string also
        
        Returns:
            text (str): Merged text from text_list
            text_df (optional[pandas.DataFrame]): None, or the dataframe file for each text
                                                if revise_df is True
        """
        text_df = text_df.copy()
        
        # convert to text
        text = ''.join(np.array(text_list)[:, 1])
        
        if not revise_df:
            return text, text_df
        
        # convert to text_df
        text_id = text_df.id.iloc[0]

        last_position = 0
        discourses = []
        for discourse_type, discourse_text in text_list:
            text_len = len(discourse_text)
            if discourse_type != "None":
                discourse_start = last_position
                discourse_end = last_position + text_len
                discourse_rows = {'id': text_id,
                                'discourse_start': discourse_start,
                                'discourse_end': discourse_end,
                                'discourse_text': discourse_text,
                                'discourse_type': discourse_type}
                discourses.append(discourse_rows)

            last_position += text_len
        
        text_df = pd.DataFrame(discourses)
        
        # recalculate prediction string
        def calculate_predictionstring(row):
            """recalculate prediction string for the augmented text data
            
            reference - https://www.kaggle.com/c/feedback-prize-2021/discussion/297591
            """
            word_start = len(text[:row.discourse_start].split())
            word_end = word_start + len(text[row.discourse_start:row.discourse_end].split())
            word_end = min(word_end, len(text.split()))

            predictionstring = " ".join([str(x) for x in range(word_start, word_end)])
            
            return predictionstring

        text_df['predictionstring'] = text_df[['discourse_start', 'discourse_end']].apply(calculate_predictionstring, axis=1)
        
        return text, text_df

    def get_discourse_boundary(self, text_id, text, text_df, revise=True, cache=True):
        """Get the discourse boundary from the text_df
        
        Args:
            text (str): the text of the text_id
            text_df (pandas.DataFrame): the dataframe file for each text
            revise (bool): revise the boundary of the text discourse boundary
            cache (bool): cache the result, but if the augmentation is hard enough set it to False
        
        Returns:
            discourse_boundary (list): list of discourse boundary
        """
        if cache:
            if text_id in self.discourse_boundary_cache:
                return self.discourse_boundary_cache[text_id]

        if revise:
            discourse_boundary = self.recalculate_entity_boundary(text, text_df)
        else:
            discourse_boundary = text_df[['discourse_start', 'discourse_end', 'discourse_type']].values

        # store the result to cache
        if cache:
            self.discourse_boundary_cache[text_id] = discourse_boundary
        
        return discourse_boundary

    def recalculate_entity_boundary(self, text, text_df):
        """recalculate the entity boundary
        Author - sergei chudov

        Args:
            text (str): literally the text of each text_id
            text_df (pandas.DataFrame): the dataframe file for each text

        Returns:
            ent_boundaries (list): list that contains the entity boundary
                                    [(0, 92, 'Lead'),
                                     (93, 130, 'Position'),
                                     (285, 356, 'Claim')]
        """
        ent_boundaries = []
        pointer = 0
        
        for row in text_df.itertuples():
            entity_text = row.discourse_text

            # regex to find text start with alphanumeric (a-zA-Z0-9)
            entity_text = entity_text[next(self.alphanumeric_re.finditer(entity_text)).start():]
            
            # if the first character length is 1, then check the previous text chunk
            if len(entity_text.split()[0]) == 1 and pointer != 0:
                entity_start_ix = text[pointer:].index(entity_text)
                prev_text = text[:pointer + entity_start_ix]
                
                # current text is not the beginning and the previous text last char is alphanumeric
                if pointer + entity_start_ix > 0 and prev_text[-1].isalpha():
                    cut_word_chunk_size = len(prev_text.split()[-1])
                    
                    # if the previous text last word length is not 1
                    if cut_word_chunk_size > 1:
                        entity_text = entity_text[next(self.alphanumeric_re.finditer(entity_text[1:])).start() + 1:]

            offset = text[pointer:].index(entity_text)
            starts_at = offset + pointer
            ent_boundaries.append((starts_at, starts_at + len(entity_text), row.discourse_type))
            pointer = starts_at + len(entity_text)
                
        return ent_boundaries

    def make_one_hot(self, indices, num_labels):
        """Make one hot encoding
        Author - sergei chudov
        """
        array = np.zeros((len(indices), num_labels))
        array[np.arange(len(indices)), indices.astype('i4')] = 1

        return array

    def token_labeling(self, text, tokenizer_outs, ent_boundaries, cat2id, max_len=2048):
        """label the tokens
        Author - sergei chudov

        Args:
            text (str): literally the text of each text_id
            tokenizer_outs (list): list of tokenizer outputs
            ent_boundaries (list): list of entity boundaries
            cat2id (dict): dictionary that maps the category to id
            max_len (int): max length of the text
        Returns:
            label (tuple): label consisted with 3 types of data
                            - tokens (max_len)
                            - token labels (max_len, num_labels)
                            - attention mask (max_len)
                            - number of tokens (1)
        """

        all_boundaries = deque([])
        for ent_boundary in ent_boundaries:
            for position, boundary_type in zip(ent_boundary[:2], ('start', 'end')):
                discourse_type = ent_boundary[-1]
                all_boundaries.append((position, discourse_type, boundary_type))
                
        current_target = 0
        token_len = len(tokenizer_outs['input_ids'])
    
        targets = np.zeros(token_len, dtype='i8')
        for token_ix in range(token_len):
            token_start_ix, token_end_ix = tokenizer_outs['offset_mapping'][token_ix]
            
            cur_pos, cur_cat_type, cur_bound_type = all_boundaries[0]

            if token_end_ix != 0 \
            and (cur_bound_type == 'end' and token_end_ix >= cur_pos) \
            or (cur_bound_type == 'start' and token_end_ix > cur_pos):
                
                if len(all_boundaries) > 1:
                    next_pos, next_dis_type, next_bound_type = all_boundaries[1]
                if cur_bound_type == 'start':
                    # cat2id {'Lead': 1, 'Position': 3, ..., 'Rebuttal': 13}
                    current_target = cat2id[cur_cat_type]
                    targets[token_ix] = current_target
                    
                    if token_end_ix == next_pos:
                        current_target = 0
                        all_boundaries.popleft()
                    else:
                        current_target += 1
                else:
                    # If there is more entity left to consider and current is already on the next pos
                    if len(all_boundaries) > 1 and token_end_ix > next_pos:
                        
                        # can this actually happen?
                        if token_start_ix >= next_pos:
                            assert text[cur_pos - 1] == '¨'

                        all_boundaries.popleft()
                        current_target = cat2id[cur_cat_type]
                        targets[token_ix] = current_target
                        current_target += 1
                    else:
                        if token_start_ix >= cur_pos:
                            current_target = 0

                        targets[token_ix] = current_target
                        current_target = 0

                all_boundaries.popleft()
                if not all_boundaries:
                    break
            else:
                targets[token_ix] = current_target

        # one-hot encoding
        targets = self.make_one_hot(targets, 15)

        # Generate label
        tokens = np.zeros(max_len, dtype='i8')
        token_labels = np.zeros((max_len, 15), dtype='f4')
        attention_mask = np.zeros(max_len, dtype='f4')

        tokens[:token_len] = tokenizer_outs['input_ids']
        token_labels[:token_len] = targets
        attention_mask[:token_len] = tokenizer_outs['attention_mask']

        label = (tokens, token_labels, attention_mask, token_len)

        return label