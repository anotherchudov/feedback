# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for TrOCR.
"""
from contextlib import contextmanager

from new_transformers import AutoFeatureExtractor, AutoTokenizer
from new_transformers.feature_extraction_utils import FeatureExtractionMixin
from new_transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from new_transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from new_transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer
from new_transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast


class TrOCRProcessor:
    r"""
    Constructs a TrOCR processor which wraps a vision feature extractor and a TrOCR tokenizer into a single processor.

    [`TrOCRProcessor`] offers all the functionalities of [`ViTFeatureExtractor`/`DeiTFeatureExtractor`] and
    [`RobertaTokenizer`/`XLMRobertaTokenizer`]. See the [`~TrOCRProcessor.__call__`] and [`~TrOCRProcessor.decode`] for
    more information.

    Args:
        feature_extractor ([`ViTFeatureExtractor`/`DeiTFeatureExtractor`]):
            An instance of [`ViTFeatureExtractor`/`DeiTFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`RobertaTokenizer`/`XLMRobertaTokenizer`]):
            An instance of [`RobertaTokenizer`/`XLMRobertaTokenizer`]. The tokenizer is a required input.
    """

    def __init__(self, feature_extractor, tokenizer):
        if not isinstance(feature_extractor, FeatureExtractionMixin):
            raise ValueError(
                f"`feature_extractor` has to be of type {FeatureExtractionMixin.__class__}, but is {type(feature_extractor)}"
            )
        if not isinstance(
            tokenizer, (RobertaTokenizer, RobertaTokenizerFast, XLMRobertaTokenizer, XLMRobertaTokenizerFast)
        ):
            raise ValueError(
                f"`tokenizer` has to be of type {RobertaTokenizer.__class__} or {RobertaTokenizerFast.__class__} or {XLMRobertaTokenizer.__class__} or {XLMRobertaTokenizerFast.__class__}, but is {type(tokenizer)}"
            )

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        """
        Save a TrOCR feature extractor object and TrOCR tokenizer object to the directory `save_directory`, so that it
        can be re-loaded using the [`~TrOCRProcessor.from_pretrained`] class method.

        <Tip>

        This class method is simply calling [`~PreTrainedFeatureExtractor.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizer.save_pretrained`]. Please refer to the docstrings of the methods
        above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """

        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a [`TrOCRProcessor`] from a pretrained TrOCR processor.

        <Tip>

        This class method is simply calling AutoFeatureExtractor's [`~PreTrainedFeatureExtractor.from_pretrained`] and
        TrOCRTokenizer's [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`]. Please refer to the
        docstrings of the methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~PreTrainedFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both [`PreTrainedFeatureExtractor`] and
                [`PreTrainedTokenizer`]
        """
        feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to AutoFeatureExtractor's
        [`~AutoFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~TrOCRProcessor.as_target_processor`] this method forwards all its arguments to TrOCRTokenizer's
        [`~TrOCRTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.
        """
        return self.current_processor(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor