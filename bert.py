import torch
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score
from tqdm import tqdm
import pickle
import os

tqdm.pandas()
label_path = 'HCC/lab/'

class TranscriptData(torch.utils.data.Dataset):
    def __init__(self, df, le = None,target='complaint',transcript_path='HCC/transcripts/',test=False):
        self.test = test
        self.df = df
        self.df['trans0'] = self.df.apply(lambda x: open(os.path.join(transcript_path, x['filename'].replace('.wav','0.txt')),'r').read(), axis=1)
        self.df['trans1'] = self.df.apply(lambda x: open(os.path.join(transcript_path, x['filename'].replace('.wav','1.txt')),'r').read(), axis=1)
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        self.df['tokenized_0'] = self.df.progress_apply(lambda x: self.tokenizer(x['trans0'], padding='max_length', truncation=True, max_length=512, return_tensors='pt'), axis=1)
        self.df['tokenized_1'] = self.df.progress_apply(lambda x: self.tokenizer(x['trans1'], padding='max_length', truncation=True, max_length=512, return_tensors='pt'), axis=1)

        #Squeeze the tensors
        self.df['tokenized_0'] = self.df.progress_apply(lambda x: {k: v.squeeze() for k, v in x['tokenized_0'].items()}, axis=1)
        self.df['tokenized_1'] = self.df.progress_apply(lambda x: {k: v.squeeze() for k, v in x['tokenized_1'].items()}, axis=1)

        self.df[target] = self.df[target].apply(lambda x: x.lower().strip())
        if le is None:
            self.le = LabelEncoder()
            self.le.fit(self.df[target])
        else:
            self.le = le
        self.df['label'] = self.le.transform(self.df[target].values)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.test:
            return self.df.iloc[idx]['tokenized_0'], self.df.iloc[idx]['tokenized_1']
        return self.df.iloc[idx]['tokenized_0'], self.df.iloc[idx]['tokenized_1'], self.df.iloc[idx]['label']

class Classifier(pl.LightningModule):
    def __init__(self,num_classes=2):
        super().__init__()
        self.model0 = AutoModel.from_pretrained('distilbert-base-multilingual-cased')
        self.model1 = AutoModel.from_pretrained('distilbert-base-multilingual-cased')
        self.classifier = torch.nn.Linear(768*2, num_classes)
        self.preds = np.array([])
        self.labels = np.array([])

    
    def forward(self, x0, x1):
        x0 = self.model0(**x0)
        x0 = x0.last_hidden_state[:,0,:]

        x1 = self.model1(**x1)
        x1 = x1.last_hidden_state[:,0,:]
        x = torch.cat([x0,x1], dim=1)
        x = self.classifier(x)
        x = torch.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x0, x1, y = batch
        y_hat = self(x0, x1)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1, y = batch
        y_hat = self(x0, x1)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_hat, y)
        self.log('val_loss', loss)
        self.preds = np.append(self.preds, y_hat.argmax(dim=1).cpu().numpy())
        self.labels = np.append(self.labels, y.cpu().numpy())
        return loss
    
    def on_validation_epoch_end(self):
        self.log('val_recall', recall_score(self.labels, self.preds, average='macro'))
        self.preds = np.array([])
        self.labels = np.array([])
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1,total_iters=100),
                'interval': 'step',
            },
            'monitor': 'val_recall',
            'interval': 'epoch'
        }

if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join(label_path, 'train.csv'))
    dev_df = pd.read_csv(os.path.join(label_path, 'devel.csv'))

    train_dataset = TranscriptData(train_df,target='request')
    dev_dataset = TranscriptData(dev_df, train_dataset.le,target='request')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=8, shuffle=False)

    checkpoints = pl.callbacks.ModelCheckpoint(dirpath='distilbert_req',monitor='val_recall', mode='max', save_top_k=1)
    logger = pl.loggers.WandbLogger(project='requests',name='distilbert')
    model = Classifier()
    trainer = pl.Trainer(
        devices=3,
        accelerator='gpu',
        logger=logger,
        max_epochs=20,
        callbacks=[checkpoints],
        precision=16,
    )
    trainer.fit(model, train_loader, dev_loader)


