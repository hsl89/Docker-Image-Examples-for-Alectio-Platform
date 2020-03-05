import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data
from torchtext import datasets
import time

from dataset import TEXT, LABEL, BasicDS 

import os
from utils import compute_class_weight
from collections import Counter
import envs # environment variables


class Flow:
    def __init__(self, model, 
            labeled, unlabeled, batch_size=64, cap=None, resume_from=None):
        self.model = model
        self.data_root = envs.DATA_DIR
        self.device = envs.CUDA_DEVICE 

        self.model.to(self.device)
        # compute class weights

        train_set = BasicDS(
            path=os.path.join(self.data_root, 'train.json'),
            text_field=TEXT,
            label_field=LABEL,
            samples=labeled,
            cap=cap)

        test_set = BasicDS(
            path=os.path.join(self.data_root, 'test.json'),
            text_field=TEXT,
            label_field=LABEL,
            samples=None,
            cap=cap)

        infer_set = BasicDS(
            path=os.path.join(self.data_root, 'train.json'),
            text_field=TEXT,
            label_field=LABEL,
            samples=unlabeled,
            cap=cap)
        
        self.train_iterator = data.BucketIterator(
            train_set,
            batch_size=batch_size,
            device=self.device,
            shuffle=True,
            sort_key=lambda x:len(x.text),
            sort_within_batch=True)

        self.test_iterator, self.infer_iterator = data.BucketIterator.splits(
            (test_set, infer_set),
            batch_size=batch_size,
            device=self.device,
            shuffle=False,
            sort_key=lambda x:len(x.text),
            sort_within_batch=True)


        labels = []
        for i in range(len(train_set)):
            labels.append(train_set[i].label)
        
        class_weight = compute_class_weight(
                Counter(labels), num_classes=10, min_count=1)

        class_weight = torch.Tensor(class_weight).to(self.device)

        self.criterion = nn.CrossEntropyLoss(class_weight)
        self.optimizer = optim.Adam(self.model.parameters())

        if envs.RESUME_FROM:
            ckpt = torch.load(os.path.join(
                envs.EXPT_DIR, envs.RESUME_FROM))
            self.model.load_state_dict(
                ckpt['model'])
            self.optimizer.load_state_dict(
                ckpt['optimizer'])

            for state in self.optimizer.state_dict():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(envs.CUDA_DEVICE)
            

    def accuracy(self, preds, y):
        #round predictions to the closest integer
        preds = torch.argmax(preds, dim=1)
        correct = (preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc

    def train(self):
        self.model.train()
        # create a dataset object 
        step=0
        total_loss, acc = 0.0, 0.0
        for batch in self.train_iterator:
            self.optimizer.zero_grad()
            text, text_lengths = batch.text
        
            text, text_lengths = text.to(self.device), text_lengths.to(self.device)
            
            predictions = self.model(text, text_lengths)

            loss = self.criterion(predictions, batch.label.long())
            loss.backward()
            self.optimizer.step()

            total_loss+=loss.item(); acc+=self.accuracy(predictions, batch.label.long())
            if step % 10 == 0:
                print('Loss: {}'.format(loss.item()))
            step+=1
        # save ckpt 
        return total_loss / len(self.train_iterator), acc / len(self.train_iterator) 

    def test(self):
        self.model.eval() 
        lbs, prd, total_loss, acc = [], [], 0.0, 0.0
        with torch.no_grad():
            for batch in self.test_iterator:
                text, text_lengths = batch.text
                output = self.model(text, text_lengths)
                # total_loss+=self.loss(output, batch.label.long()).item()
                acc+=self.accuracy(output, batch.label.long()).item()
                preds = torch.argmax(output, dim=1)
                # lbs.extend(batch.label.long().cpu().numpy().tolist())
                prd.extend(preds.cpu().numpy().tolist())
                lbs.extend(batch.label.long().cpu().numpy().tolist())
        lbs = {i: v for i, v in enumerate(lbs)}
        prd = {i: v for i, v in enumerate(prd)}

        return acc / len(self.test_iterator), prd, lbs

    def infer(self):
        outputs=[]
        self.model.eval()
        with torch.no_grad():
            for batch in self.infer_iterator:
                text, text_lengths = batch.text
                output = self.model(text, text_lengths)
                outputs.extend(output.cpu().numpy().tolist())
        
        return outputs
