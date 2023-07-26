import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class Extractor:
    def __init__(self, config):
        self.config = config

    def extract(self):
        data = pd.read_json(self.config.input_file)
        data = data.dropna()
        data['Input'] = [', '.join(d['Ingredients']) + ' ' + d['Instructions'] for i,d in data.iterrows()]
        if self.config.debug:
            data = data.iloc[:20]
        data['Ingredient_processed'] = data['Ingredients'].apply(lambda x: [self.config.nlp(t) for t in x])
        data['Ingredient_processed'] = data['Ingredient_processed'].apply(lambda x: [self._clean(t.text) for doc in x for t in doc if self._filter(t)])

        flat_ingredients = [item for sublist in data['Ingredient_processed'] for item in sublist]
        ingredient_map = {k: v for v, k in enumerate(np.unique(flat_ingredients))}
        data['Labels'] = data['Ingredient_processed'].apply(lambda x: [ingredient_map[t] for t in x])

        # split into train, dev and tests
        train_data, test_data = train_test_split(data, test_size=self.config.test_size, random_state=self.config.seed)
        train_data, dev_data = train_test_split(train_data, test_size=self.config.test_size, random_state=self.config.seed)

        train_data.to_json(self.config.data_path + "train.json")
        dev_data.to_json(self.config.data_path + "dev.json")
        test_data.to_json(self.config.data_path + "test.json")
        return train_data, dev_data, test_data

    def _filter(self, token):
        if len(token) < 2:
            return False
        if token.is_stop:
            return False
        if token.text[0].islower():
            return False
        if token.is_digit:
            return False
        if token.like_num:
            return False
        for c in token.text:
            if c in "0123456789":
                return False
        if token.text[0] in ['.', ',', ':', ';', '+', '-', '(', ')', '[', ']', '{', '}', '/', '\\', '!', '?', '@', '#', '$', '%', '^', '&', '*', '_', '=', '<', '>', '|', '~', '`', '"', "'"]:
            return False
        if token.text.isupper():
            return False
        return True

    def _clean(self, text):
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("[", " ")
        text = text.replace("]", " ")
        text = text.replace("{", " ")
        text = text.replace("}", " ")
        text = text.replace("/", " ")
        text = text.replace("\\", " ")
        text = text.replace("!", " ")
        text = text.replace("?", " ")
        text = text.replace("@", " ")
        text = text.replace("#", " ")
        text = text.replace("$", " ")
        text = text.replace("%", " ")
        text = text.replace("^", " ")
        text = text.replace("&", " ")
        text = text.replace("*", " ")
        text = text.replace("_", " ")
        text = text.replace("=", " ")
        text = text.replace("<", " ")
        text = text.replace(">", " ")
        text = text.replace("|", " ")
        text = text.replace("~", " ")
        text = text.replace("-", " ")
        text = text.replace("+", " ")
        text = text.replace(":", " ")
        text = text.replace(";", " ")
        text = text.replace(",", " ")
        text = text.replace(".", " ")
        return text
