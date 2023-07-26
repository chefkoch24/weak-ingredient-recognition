import torch
from torchmetrics import F1Score, Precision, Recall, ExactMatch, Accuracy
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall, \
    MultilabelHammingDistance, MultilabelAccuracy, MultilabelExactMatch


class Metrics:
    def __init__(self, config, num_token_classes=2, num_classification_classes=10):
        self.config = config
        self.multilabel_f1 = MultilabelF1Score(num_labels=num_classification_classes).to(self.config.device)
        self.multilabel_precision = MultilabelPrecision(num_labels=num_classification_classes).to(self.config.device)
        self.multilabel_recall = MultilabelRecall(num_labels=num_classification_classes).to(self.config.device)
        self.hamming_distance = MultilabelHammingDistance(num_labels=num_classification_classes).to(self.config.device)
        self.multilabel_accuracy = MultilabelAccuracy(num_labels=num_classification_classes).to(self.config.device)
        self.multilabel_em = MultilabelExactMatch(num_labels=num_classification_classes).to(self.config.device)

        self.token_f1 = F1Score(task='multiclass', num_classes=num_token_classes, ignore_index=-100).to(self.config.device)
        self.token_precision = Precision(task='multiclass', num_classes=num_token_classes, ignore_index=-100).to(self.config.device)
        self.token_recall = Recall(task='multiclass', num_classes=num_token_classes, ignore_index=-100).to(self.config.device)
        self.token_accuracy = Accuracy(task='multiclass', num_classes=num_token_classes, ignore_index=-100).to(self.config.device)



    def multilabel_classification(self, outputs):
        labels = outputs['classification_labels']
        preds = outputs['classification_pred']
        f1 = self.multilabel_f1(preds, labels)
        precision = self.multilabel_precision(preds, labels)
        recall = self.multilabel_recall(preds, labels)
        hamming_distance = self.hamming_distance(preds, labels)
        accuracy = self.multilabel_accuracy(preds, labels)
        em = self.multilabel_em(preds, labels)
        return {'f1_class': f1.item(), 'precision_class': precision.item(), 'recall_class': recall.item(), 'hamming_distance_class': hamming_distance.item(), 'accuracy_class': accuracy.item(), 'em_class': em.item()}

    def token_classification(self, outputs):
        labels = outputs['ner_labels'].long()
        preds = outputs['token_pred']
        f1 = self.token_f1(preds, labels)
        precision = self.token_precision(preds, labels)
        recall = self.token_recall(preds, labels)
        accuracy = self.token_accuracy(preds, labels)
        return {'f1_token': f1.item(), 'precision_token': precision.item(), 'recall_token': recall.item(), 'accuracy_token': accuracy.item()}
