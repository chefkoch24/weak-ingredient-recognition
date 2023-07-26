import pandas as pd

from config import Config
from preprocessor import Preprocessor


class Tests:

    def test_preprocessor(self):
        # TODO
        config = Config()
        labels = [0, 0, 0, 1]
        input = ['Das ist ein Test']
        ingredients = ['100 g Mehl', '1l Wasser', 'Salz', 'Pfeffer']
        ingredients_processed = ['Mehl']
        data = pd.DataFrame({
                'Instructions': input,
                'Ingredients': ingredients,
                'NER_Labels': [labels],
                'Ingredient_processed': [ingredients_processed]
            })
        preprocessor = Preprocessor(config)
        data = preprocessor.preprocess(data)
        assert len(data['ner_labels'][0]) == config.max_length
        assert len([True for d in data['ner_labels'][0] if d != -100]) == len(labels)
        assert len([True for d in data['ner_labels'][0] if d != -100]) == len(
            [True for d in data['input_ids'][0] if (d != 3 and d != 4 and d != 0)])



