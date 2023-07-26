import json

import numpy as np
import spacy
import re
from skweak import heuristics, gazetteers, generative, utils, aggregation
from skweak.base import CombinedAnnotator
from skweak.heuristics import SpanEditorAnnotator, SpanConstraintAnnotator
from spacy.matcher import PhraseMatcher


class WeakSupervision:
    def __init__(self, config, aggregation_method='hmm'):
        self.config = config
        self.ingredients_map = json.load(open(self.config.data_path + 'ingredient_map.json'))
        self.ingredients_patterns = [self.config.nlp(k) for k in self.ingredients_map.keys()]
        self.ingredients_trie = self._ingredients_trie()
        self.units = {"el", "tl", "esslöffel", 'teelöfel','g', 'mg' 'kg', 'gramm', 'milligramm', 'kilogramm', 'ml', 'l', 'liter', 'milliliter', 'stück', 'stücke', 'st', 'stk', 'stueck', 'dose', 'dosen', 'packung', 'packungen', 'pack', 'pck', 'tasse'}
        self.hmm = aggregation.HMM("hmm", ["INGREDIENT", "QUANTITY", "UNIT"], prefixes='BIO')
        self.voter = aggregation.MajorityVoter("voter", ["INGREDIENT", "QUANTITY", "UNIT"],  prefixes='BIO')
        self.aggregation_method = aggregation_method
        self.combined_annotator = CombinedAnnotator()
        self.combined_annotator.add_annotator(self.lf_ingredient_gazetteer_match())
        self.combined_annotator.add_annotator(self.lf_ingredient_lemma_match())
        #self.combined_annotator.add_annotator(self.lf_ingredient_regex_match())
        self.combined_annotator.add_annotator(self.lf_quantity_match())
        self.combined_annotator.add_annotator(self.lf_quantity_with_unit_match())
        self.combined_annotator.add_annotator(self.lf_unit_match())



    def _ingredients_trie(self):
        ingredients = []
        for k, v in self.ingredients_map.items():
            ingredients.append((k,))
        return gazetteers.Trie(ingredients)

    def lf_ingredient_gazetteer_match(self):
        return gazetteers.GazetteerAnnotator("gazetteer", {"INGREDIENT": self.ingredients_trie})

    def lf_ingredient_lemma_match(self):
        def lemma_match(doc):
            matcher = PhraseMatcher(self.config.nlp.vocab, attr="LEMMA")
            matcher.add("Lemma_Match", self.ingredients_patterns)
            for match_id, start, end in matcher(doc):
                yield start, end, "INGREDIENT"

        return heuristics.FunctionAnnotator("lemma_match", lemma_match)

    def lf_ingredient_regex_match(self):
        def regex(doc):
            for ingredient in self.ingredients_patterns:
                i = ingredient.text.lower()
                for t in doc:
                    if i in t.text.lower() and t.tag in ["NN", 'NNP']:
                        yield t.i, t.i+1, "INGREDIENT"

        return heuristics.FunctionAnnotator("regex_match", regex)

    def lf_quantity_match(self):
        def quantity(doc):
            for t in doc:
                if t.is_digit:
                    yield t.i, t.i + 1, "QUANTITY"
        return heuristics.FunctionAnnotator("quantity", quantity)

    def lf_quantity_with_unit_match(self):
        def quantity_unit(span):
            last_token = span[-1]
            try:
                next_token = last_token.nbor(1)
            except IndexError:
                return False
            if next_token.text.lower() in self.units: #maybe the frist is wrong
                return True
            else:
                return False
        return SpanConstraintAnnotator("quantity_unit", "quantity", quantity_unit)


    def lf_unit_match(self):
        def unit(doc):
            for t in doc:
                if t.text.lower() in self.units:
                    yield t.i, t.i + 1, "UNIT"
        return heuristics.FunctionAnnotator("unit", unit)


    def fit(self, data):
        docs = data['Input'].apply(lambda x: self.config.nlp(x)).tolist()
        docs = list(self.combined_annotator.pipe(docs))
        if self.aggregation_method == 'hmm':
            docs = self.hmm.fit_and_aggregate(docs)
        elif self.aggregation_method == 'voter':
            docs = self.voter.fit_and_aggregate(docs)
        return docs

    def annotate(self,data):
        combined_docs = [self.config.nlp(d['Input']) for _, d in data.iterrows()]
        docs = list(self.combined_annotator.pipe(combined_docs))
        if self.aggregation_method == 'hmm':
            docs = list(self.hmm.pipe(docs))
        elif self.aggregation_method == 'voter':
            docs = list(self.voter.pipe(docs))
        return docs, self._get_labels(docs)

    def _get_labels(self, docs):
        annotated_data = []
        for doc in docs:
            label = np.zeros((len(doc)))
            probs = doc.spans[self.aggregation_method].attrs['probs']
            # if two probabilites are given the last one above 0.5 is taken
            for k, v in probs.items():
                if v.get('B-INGREDIENT', 0) > 0.5:
                    label[int(k)] = 1
                elif v.get('I-INGREDIENT', 0) > 0.5:
                    label[int(k)] = 2
                elif v.get('B-UNIT', 0) > 0.5:
                    label[int(k)] = 3
                elif v.get('I-UNIT', 0) > 0.5:
                    label[int(k)] = 4
                elif v.get('B-QUANTITY', 0) > 0.5:
                    label[int(k)] = 5
                elif v.get('I-QUANTITY', 0) > 0.5:
                    label[int(k)] = 6
            annotated_data.append(label.tolist())
        return annotated_data


