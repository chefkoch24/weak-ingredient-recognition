import json
import logging

import numpy as np
import pandas as pd
import skweak
import torch
from config import Config
from extraction import Extractor
from preprocessor import Preprocessor
from scraper.scraper import Scraper
from tests.tests import Tests
from trainer import Trainer
from weak_supervision import WeakSupervision

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def __main__():
    logger.info("Start procedure ...")
    config = Config()
    if not config.skip_scraper:
        scraper = Scraper(config)
        lebensmittel = scraper.scrape_wiktionary(
            url='https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Essen_und_Trinken/Lebensmittel')
        getraenke = scraper.scrape_wiktionary(
            url='https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Essen_und_Trinken/Getr%C3%A4nke')
        obst_gemuese = scraper.scrape_wiktionary(
            url='https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Essen_und_Trinken/Obst_und_Gem%C3%BCse')
        speisen = scraper.scrape_wiktionary(
            url='https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Essen_und_Trinken/Speisen')
        full_list = np.unique(lebensmittel + getraenke + obst_gemuese + speisen).tolist()
        ingredient_map = {k: v for v, k in enumerate(full_list)}
        with open(config.data_path + "ingredient_map.json", "w") as outfile:
            json.dump(ingredient_map, outfile)
    if not config.skip_extraction:
        logger.info("Extracting data ... ")
        extractor = Extractor(config)
        train, dev, test = extractor.extract()
    else:
        logger.info("Load data ... ")
        if config.debug:
            train = pd.read_json(config.data_path + 'train.json')[:10]
            dev = pd.read_json(config.data_path + 'dev.json')[:10]
            test = pd.read_json(config.data_path + 'test.json')[:10]
        else:
            train = pd.read_json(config.data_path + 'train.json')
            dev = pd.read_json(config.data_path + 'dev.json')
            test = pd.read_json(config.data_path + 'test.json')
    weak_supervision = WeakSupervision(config, aggregation_method=config.aggregation_method)
    if not config.skip_weak_supervision:
        logger.info("Weak Supervision Fitting ... ")
        docs = weak_supervision.fit(train)
        for doc in docs:
            doc.ents = doc.spans[config.aggregation_method]
        train_labels = weak_supervision._get_labels(docs)
        skweak.utils.docbin_writer(docs, config.data_path + "training_corpus.spacy")
        train['NER_Labels'] = train_labels
        logger.info("Weak Supervision Annotating ... ")
        docs, dev_labels = weak_supervision.annotate(dev)
        for doc in docs:
            doc.ents = doc.spans[config.aggregation_method]
        skweak.utils.docbin_writer(docs, config.data_path + "dev_corpus.spacy")
        docs, test_labels = weak_supervision.annotate(test)
        for doc in docs:
            doc.ents = doc.spans[config.aggregation_method]
        skweak.utils.docbin_writer(docs, config.data_path + "test_corpus.spacy")

    else:
        train_corpus = skweak.utils.docbin_reader(config.data_path + "training_corpus.spacy", spacy_model_name=config.spacy_model)
        dev_corpus = skweak.utils.docbin_reader(config.data_path + "dev_corpus.spacy", spacy_model_name=config.spacy_model)
        test_corpus = skweak.utils.docbin_reader(config.data_path + "test_corpus.spacy", spacy_model_name=config.spacy_model)
        train_labels = weak_supervision._get_labels(train_corpus)
        dev_labels = weak_supervision._get_labels(dev_corpus)
        test_labels = weak_supervision._get_labels(test_corpus)
    if config.debug:
        train['NER_Labels'] = train_labels[:len(train)]
        dev['NER_Labels'] = dev_labels[:len(dev)]
        test['NER_Labels'] = test_labels[:len(test)]
    else:
        train['NER_Labels'] = train_labels
        dev['NER_Labels'] = dev_labels
        test['NER_Labels'] = test_labels
    logger.info("Preprocessing ... ")
    preprocessor = Preprocessor(config)
    train_data = preprocessor.preprocess(train)
    val_data = preprocessor.preprocess(dev)
    test_data = preprocessor.preprocess(test)

    trainer = Trainer(config, train_data, val_data, test_data=test_data)
    if config.mode == 'train':
        logger.info("Training ... ")
        trainer.train()
    elif config.mode == 'prediction':
        logger.info("Predicting ... ")
        version = config.checkpoint.split('/')[-1]
        predictions = trainer.predict(test_data, path=config.checkpoint)
        # save predictions
        input_ids = torch.cat([batch['input_ids'] for batch in predictions]).detach().cpu().numpy().tolist()
        ner_labels = torch.cat([batch['ner_labels'] for batch in predictions]).detach().cpu().numpy().tolist()
        token_predictions = torch.cat([batch['token_pred'] for batch in predictions]).detach().cpu().numpy().tolist()
        attention_masks = torch.cat([batch['attention_mask'] for batch in predictions]).detach().cpu().numpy().tolist()
        token_type_ids = torch.cat([batch['token_type_ids'] for batch in predictions]).detach().cpu().numpy().tolist()
        pd.DataFrame({
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'token_type_ids': token_type_ids,
            'ner_labels': ner_labels,
            'token_predictions': token_predictions,
        }).to_csv(f'results/predictions_{version}.csv', index=False)

    elif config.mode == 'evaluate':
        logger.info("Testing ... ")
        metrics = trainer.evaluate(path=config.checkpoint)
        logger.info(f"Evaluation results: {metrics}")
    elif config.mode == 'save':
        logger.info("Saving ... ")
        trainer.save(path=config.checkpoint)
    elif config.mode == 'test':
        logger.info("Testing ... ")
        tests = Tests()
        tests.test_preprocessor()
    logger.info("Done!")




if __name__ == '__main__':
    __main__()