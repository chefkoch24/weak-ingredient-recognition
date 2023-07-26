import spacy

# https://www.kaggle.com/datasets/sterby/german-recipes-dataset
import torch
import argparse

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected boolean value.')



class Config:
    def __init__(self):
        args = self.get_params()
        self.input_file = args.input_file
        self.data_path = args.data_path
        self.test_size = args.test_size
        self.seed = args.seed
        self.spacy_model = args.spacy_model
        self.transformer_model = args.transformer_model
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.learning_rate = args.learning_rate
        self.soft_labels = args.soft_labels
        self.dense_layer_size = args.dense_layer_size
        self.debug = args.debug
        self.task = args.task
        self.skip_extraction = args.skip_extraction
        self.skip_weak_supervision = args.skip_weak_supervision
        self.skip_scraper = args.skip_scraper
        self.mode = args.mode
        self.aggregation_method = args.aggregation_method
        self.checkpoint = args.checkpoint
        self.version = args.version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nlp = spacy.load(self.spacy_model)
        self.label2idx = {
            'O': 0,
            'B-INGREDIENT': 1,
            'I-INGREDIENT': 2,
            'B-UNIT': 3,
            'I-UNIT': 4,
            'B-QUANTITY': 5,
            'I-QUANTITY': 6
        }
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def to_dict(self):
        dict = {}
        for key, value in self.__dict__.items():
            if not callable(value):
                if isinstance(value, torch.device):
                    dict[key] = value.type
                else:
                    dict[key] = value
        return dict


    def get_params(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--input_file", type=str, default="input/recipes.json")
        arg_parser.add_argument("--data_path", type=str, default="data/")
        arg_parser.add_argument("--test_size", type=float, default=0.2)
        arg_parser.add_argument("--seed", type=int, default=42)
        arg_parser.add_argument("--spacy_model", type=str, default="de_core_news_md")
        arg_parser.add_argument("--transformer_model", type=str, default="bert-base-german-cased")
        arg_parser.add_argument("--max_length", type=int, default=512)
        arg_parser.add_argument("--batch_size", type=int, default=8)
        arg_parser.add_argument("--max_epochs", type=int, default=15)
        arg_parser.add_argument("--dense_layer_size", type=int, default=2000)
        arg_parser.add_argument("--learning_rate", type=float, default=2e-5)
        arg_parser.add_argument("--soft_labels", type=str2bool, nargs='?', const=True, default=False) # not fully implemented yet
        arg_parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False)
        arg_parser.add_argument("--task", type=str, default="multitask")
        arg_parser.add_argument("--mode", type=str, default="train")
        arg_parser.add_argument("--checkpoint", type=str, default="checkpoints/")
        arg_parser.add_argument("--skip_extraction", type=str2bool, nargs='?', const=True, default=False)
        arg_parser.add_argument("--skip_weak_supervision", type=str2bool, nargs='?', const=True, default=False)
        arg_parser.add_argument("--skip_scraper",  type=str2bool, nargs='?', const=True, default=False)
        arg_parser.add_argument("--aggregation_method", type=str, default="hmm")
        arg_parser.add_argument("--version", type=str, default="v0.0.1")
        args, _ = arg_parser.parse_known_args()

        return args




