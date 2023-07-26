import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoModel, BertForTokenClassification, BertModel, AutoModelForTokenClassification, Adafactor

from metrics import Metrics


class MultiTaskModel(pl.LightningModule):
    def __init__(self, config, num_token_classes, num_classification_classes):
        super().__init__()
        self.config = config
        self.model = BertModel.from_pretrained(self.config.transformer_model)
        self.token_classification_head = TokenClassificationHead(self.config, self.model.config, num_labels=num_token_classes)
        self.classification_head = MultiLabelClassificationHead(self.config, self.model.config, num_labels=num_classification_classes)
        self.sigmoid = nn.Sigmoid().to(self.config.device)
        self.metrics = Metrics(self.config, num_token_classes=num_token_classes,
                               num_classification_classes=num_classification_classes)
        self.model.post_init()
        self.save_hyperparameters(ignore=["config"])



    def forward(self, input_ids, attention_mask, ner_labels, classification_labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        classification_output = outputs[1]
        sequence_logits = self.token_classification_head.forward(sequence_output)
        classification_logits = self.classification_head.forward(classification_output)
        sequence_loss = self.token_classification_head.loss_function(sequence_logits, ner_labels)
        classification_loss = self.classification_head.loss_function(classification_logits, classification_labels)
        # try different loss aggregation methods
        loss = sequence_loss + classification_loss
        return loss, classification_logits, sequence_logits


    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ner_labels = batch['ner_labels']
        classification_labels = batch['classification_labels']
        loss, classification_logits, sequence_logits = self.forward(input_ids, attention_mask, ner_labels, classification_labels)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ner_labels = batch['ner_labels']
        classification_labels = batch['classification_labels']
        loss, classification_logits, sequence_logits = self.forward(input_ids, attention_mask, ner_labels, classification_labels)
        self.log('val_loss', loss,  on_epoch=True, prog_bar=True, on_step=False)
        classification_pred = self.sigmoid(classification_logits)
        sequence_pred = torch.argmax(sequence_logits, dim=-1)
        batch['classification_pred'] = classification_pred
        batch['token_pred'] = sequence_pred
        batch['loss'] = loss
        metrics = self.metrics.multilabel_classification(batch)
        metrics.update(self.metrics.token_classification(batch))
        self.log_dict(metrics, on_epoch=True)
        return metrics

    def predict_step(self, batch, batch_idx, **kwargs):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ner_labels = batch['ner_labels']
        classification_labels = batch['classification_labels']
        loss, classification_logits, sequence_logits = self.forward(input_ids, attention_mask, ner_labels, classification_labels)
        classification_pred = self.sigmoid(classification_logits)
        sequence_pred = torch.argmax(sequence_logits, dim=-1)
        batch['classification_pred'] = classification_pred
        batch['token_pred'] = sequence_pred
        batch['loss'] = loss
        return batch


    def configure_optimizers(self):
        model_params = list(self.model.parameters())  # Parameters of the BERT model
        head_params = list(self.token_classification_head.classifier.parameters()) + list(self.classification_head.output_classifier.parameters()) + list(self.classification_head.classifier.parameters())
        optimizer = torch.optim.Adam(model_params + head_params, lr=self.config.learning_rate)
        # optimizer = Adafactor(self.model.parameters(), lr=None, relative_step=True)
        return optimizer


class MultiLabelClassificationHead:
    def __init__(self, config, bert_config, num_labels=2):
        classifier_dropout = (
            bert_config.classifier_dropout if bert_config.classifier_dropout is not None else bert_config.hidden_dropout_prob
        )
        self.num_labels = num_labels
        self.dropout = nn.Dropout(classifier_dropout).to(config.device)
        self.classifier = nn.Linear(bert_config.hidden_size, config.dense_layer_size).to(config.device)
        self.output_classifier = nn.Linear(config.dense_layer_size, self.num_labels).to(config.device)
        self.loss = nn.BCEWithLogitsLoss().to(config.device)
        self.relu = nn.ReLU().to(config.device)

    def forward(self, output):
        output = self.dropout(output)
        logits = self.classifier(output)
        logits = self.relu(logits)
        logits = self.output_classifier(logits)
        return logits

    def loss_function(self, logits, labels):
        return self.loss(logits, labels)


class TokenClassificationHead:
    def __init__(self, config, bert_config, num_labels=2):
        classifier_dropout = (
            bert_config.classifier_dropout if bert_config.classifier_dropout is not None else bert_config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout).to(config.device)
        self.num_labels = num_labels
        self.classifier = nn.Linear(bert_config.hidden_size, self.num_labels).to(config.device)
        self.loss = nn.CrossEntropyLoss().to(config.device)

    def forward(self, output):
        sequence_output = self.dropout(output)
        logits = self.classifier(sequence_output)
        return logits

    def loss_function(self, logits, labels):
        return self.loss(logits.view(-1, self.num_labels), labels.view(-1).long())


class TokenClassificationModel(pl.LightningModule):
    def __init__(self, config, num_token_classes):
        super().__init__()
        self.config = config
        self.num_token_classes = num_token_classes
        self.learning_rate = config.learning_rate
        self.seed = config.seed
        self.transformer_model = config.transformer_model
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.max_epochs = config.max_epochs
        self.soft_labels = config.soft_labels
        self.model = AutoModelForTokenClassification.from_pretrained(self.transformer_model, num_labels=self.num_token_classes)
        self.metrics = Metrics(self.config, num_token_classes=self.num_token_classes)
        self.save_hyperparameters(ignore=["config"])

    def forward(self, input_ids, attention_mask, ner_labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=ner_labels)
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ner_labels = batch['ner_labels']
        loss, logits = self.forward(input_ids, attention_mask, ner_labels)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ner_labels = batch['ner_labels']
        loss, logits = self.forward(input_ids, attention_mask, ner_labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        pred = torch.argmax(logits, dim=-1)
        batch['token_pred'] = pred
        batch['loss'] = loss
        metrics = self.metrics.token_classification(batch)
        self.log_dict(metrics, on_epoch=True)
        return metrics

    def predict_step(self, batch, batch_idx, **kwargs):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ner_labels = batch['ner_labels']
        loss, logits = self.forward(input_ids, attention_mask, ner_labels)
        pred = torch.argmax(logits, dim=-1)
        batch['token_pred'] = pred
        batch['loss'] = loss
        return batch

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #optimizer = Adafactor(self.model.parameters(), lr=None, relative_step=True, warmup_init=True)
        return optimizer