import json

import numpy as np
from transformers import AutoTokenizer

from alignment import Alignment


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.transformer_model)
        self.ingredient_map = json.load(open(self.config.data_path + 'ingredient_map.json'))

    def preprocess(self, data):
        data = Alignment(self.config).align(data)
        text = data['Input'].tolist()

        model_input = self.tokenizer(text, padding='max_length', truncation=True,
                                     max_length=self.config.max_length)
        model_input['ner_labels'] = data['NER_Labels'].map(self._pad_labels).tolist()
        assert len([True for d in model_input['ner_labels'][0] if d != -100]) == len([True for d in model_input['input_ids'][0] if (d != 3 and d != 4 and d != 0)])
        return model_input


    def _pad_labels(self, labels):
        if not self.config.soft_labels:
            labels = list(map(lambda x: round(x), labels))
        pad_len = self.config.max_length - len(labels) - 2 # for CLS and end of seq
        if pad_len > 0:
            labels = [-100] + labels + [-100] + [-100] * pad_len
        else:
            labels = [-100] + labels[:self.config.max_length-2] + [-100]
        assert len(labels) == self.config.max_length
        return labels

    def _encode_labels(self, labels):
        mapped_labels = [self.ingredient_map[label] for label in labels]
        encoded_labels = np.zeros((len(self.ingredient_map)))
        encoded_labels[mapped_labels] = 1
        return encoded_labels.tolist()
