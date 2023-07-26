import json
import os

import huggingface_hub
import torch
import pytorch_lightning as pl
from dotenv import load_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import MultiTaskDataset
from model import MultiTaskModel, TokenClassificationModel


class Trainer:
    def __init__(self, config, train_data, val_data, test_data=None):
        self.config = config
        torch.manual_seed(config.seed)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.checkpoint_callback = ModelCheckpoint(
            filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
        )

        self.early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=3,
            mode='min'
        )
        self.trainer = pl.Trainer(max_epochs=self.config.max_epochs,
                               deterministic=True,
                               fast_dev_run = self.config.debug,
                               logger=CSVLogger("logs", name=self.config.task),
                               callbacks=[self.checkpoint_callback, self.early_stop_callback],
                               accelerator="gpu" if torch.cuda.is_available() else 'cpu',
        )



    def train(self):
        training_dataset = MultiTaskDataset(self.train_data)
        val_dataset = MultiTaskDataset(self.val_data)
        test_dataset = MultiTaskDataset(self.test_data) if self.test_data else None
        train_dataloader = DataLoader(training_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False) if test_dataset else None
        if self.config.task == "multitask":
            self.model = MultiTaskModel(self.config, num_token_classes=training_dataset.num_token_classes(), num_classification_classes=training_dataset.num_classification_classes())
        elif self.config.task == "token":
            self.model = TokenClassificationModel(self.config, num_token_classes=training_dataset.num_token_classes())
        self.trainer.fit(self.model, train_dataloader, val_dataloader)
        with open(f"logs/{self.config.task}/config.json", 'w') as file:
            json.dump(self.config.to_dict(), file)


    def predict(self, data, path=''):
        dataset = MultiTaskDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        hparams = path.split('/')[:-2]
        hparams = "/".join(hparams) + '/hparams.yaml'
        if self.config.task == "multitask":
            self.model = MultiTaskModel.load_from_checkpoint(config=self.config, hparams_file=hparams,checkpoint_path=path, map_location=self.config.device)
        elif self.config.task == "token":
            self.model = TokenClassificationModel.load_from_checkpoint(config=self.config, hparams_file=hparams,checkpoint_path=path, map_location=self.config.device)
        predictions = self.trainer.predict(self.model, dataloader)
        return predictions

    def evaluate(self, path=''):
        hparams = path.split('/')[:-2]
        hparams = "/".join(hparams) + '/hparams.yaml'
        test_dataset = MultiTaskDataset(self.test_data)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size,
                                     shuffle=False)
        if self.config.task == "multitask":
            self.model = MultiTaskModel.load_from_checkpoint(config=self.config, hparams_file=hparams,
                                                             checkpoint_path=path, map_location=self.config.device)
        elif self.config.task == "token":
            self.model = TokenClassificationModel.load_from_checkpoint(config=self.config, hparams_file=hparams,
                                                                       checkpoint_path=path,
                                                                       map_location=self.config.device)
        return self.trainer.test(self.model, test_dataloader)

    def save(self, path=''):
        hparams = path.split('/')[:-2]
        hparams = "/".join(hparams) + '/hparams.yaml'
        model = TokenClassificationModel.load_from_checkpoint(config=self.config, hparams_file=hparams, checkpoint_path=path,
                                                              map_location=self.config.device)
        transfomer_model = model.model
        transfomer_model.revision = self.config.version
        transfomer_model.config.id2label = self.config.idx2label
        transfomer_model.config.label2id = {str(k): v for k, v in self.config.label2idx.items()}
        save_path = path.split('/')[:-1]
        save_path = "/".join(save_path)
        transfomer_model.save_pretrained(save_path)
        tokenizer = AutoTokenizer.from_pretrained(model.transformer_model)
        tokenizer.save_pretrained(save_path)
        load_dotenv()
        # Access the Hugging Face token
        hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')
        huggingface_hub.login(token=hugging_face_token)
        transfomer_model.push_to_hub("chefkoch24/weak-ingredient-recognition-bert-base-uncased-german")
        tokenizer.push_to_hub("chefkoch24/weak-ingredient-recognition-bert-base-uncased-german")
