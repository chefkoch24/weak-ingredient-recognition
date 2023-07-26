import tokenizations
import numpy as np
from transformers import AutoTokenizer


class Alignment:

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.transformer_model)

    def align(self, data):
        aligned_labels = []
        for i, d in data.iterrows():
            spacy_tokens = self.config.nlp(d['Input'])
            wordpiece_tokens = self.tokenizer.tokenize(d['Input'],
                                                       padding=False,
                                                       truncation=True,
                                                       max_length=self.config.max_length,
                                                       add_special_tokens=False)
            labels = d['NER_Labels']
            spacy_tokens = [token.text for token in spacy_tokens]
            label_ids = self._align(spacy_tokens, wordpiece_tokens, labels)
            aligned_labels.append(label_ids)
        data['NER_Labels'] = aligned_labels
        return data



    def _align(self, spacy_tokens, wordpiece_tokens, labels):
        """
        Aligns spaCy tokens to wordpiece tokens.
        """
        a2b, b2a = tokenizations.get_alignments(spacy_tokens, wordpiece_tokens)
        len_of_classification = len(wordpiece_tokens)  # for CLS and end of seq
        label_ids = np.zeros((len_of_classification))
        previous_label_idx = 0
        previous_label = 0
        for j, e in enumerate(b2a):
            if len(e) >= 1:  # Check if not special token
                label_idx = e[0]
                if previous_label != labels[label_idx] or previous_label == 0:
                    label_ids[j] = labels[label_idx]
                else:
                    label_ids[j] = labels[label_idx] + 1
                previous_label_idx = label_idx
                previous_label = labels[label_idx]
            else:
                label_ids[j] = labels[previous_label_idx]
        return label_ids
