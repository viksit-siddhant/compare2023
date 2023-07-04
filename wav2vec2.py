import lightning.pytorch as pl
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, WavLMModel
from scipy.stats import spearmanr
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
import librosa
import pickle

tqdm.pandas()


data_path = 'HCC/wav/'

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, df, data_path,test=False):
        self.df = df
        self.data_path = data_path
        self.files = df['filename'].copy()
        if test:
            self.labels = None
        else:
            self.labels = df.drop('filename', axis=1).values
        print('Loading and processing audio')
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.files['audio'] = self.files.progress_apply(lambda x: librosa.load(self.data_path + x,sr=16000)[0])
        self.files['audio'] = self.files['audio'].progress_apply(lambda x: self.processor(x, sampling_rate=16000, return_tensors="pt", padding=True).input_values.squeeze(0))
        maxlen = max(self.files['audio'].apply(lambda x: x.shape[0]))
        self.files['audio'] = self.files['audio'].progress_apply(lambda x: torch.cat((x, torch.zeros(maxlen - x.shape[0]))))


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.files['audio'][idx]
        return self.files['audio'][idx], self.labels[idx]
    
class RequestsDataset(torch.utils.data.Dataset):
    @classmethod
    def get_le(cls,df,target='complaint'):
        df[target] = df[target].apply(lambda x: x.lower().strip())
        le = LabelEncoder()
        le.fit(df[target])
        return le

    def get_labels(self):
        return self.labels

    def __init__(self,df, data_path,target='complaint',max_sec=10,sr=16000, le = None,truncate=True,test=False):
        self.test = test
        self.truncate = truncate
        self.files = df['filename'].copy()
        #Sanitize labels
        df[target] = df[target].apply(lambda x: x.lower().strip())
        if le is None:
            self.le = LabelEncoder()
            self.labels = self.le.fit_transform(df[target].values)
        else:
            self.le = le
            self.labels = self.le.transform(df[target].values)
        self.maxlen = max_sec * sr
        print('Loading and processing audio')
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french")
        self.files['audio'] = self.files.progress_apply(lambda x: librosa.load(data_path + x,sr=sr)[0])
        self.files['audio'] = self.files['audio'].progress_apply(lambda x: self.processor(x, sampling_rate=sr, return_tensors="pt", padding=True).input_values.squeeze(0))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        audio = self.files['audio'][idx]
        if not self.truncate:
            return audio, self.labels[idx]
        if (audio.shape[0] > self.maxlen):
            start = np.random.randint(audio.shape[0] - self.maxlen)
            audio = audio[start:start+self.maxlen]
        else:
            audio = torch.cat((audio, torch.zeros(self.maxlen - audio.shape[0])))
        if not self.test:
            return audio, self.labels[idx]
        else:
            return audio

class AudioModel(pl.LightningModule):
    def __init__(self,num_classes,task='requests',ckpt='facebook/wav2vec2-large-xlsr-53'):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(ckpt)
        self.model.feature_extractor._freeze_parameters()
        self.layer_weights = torch.nn.Parameter(torch.ones(25))
        self.linear = torch.nn.Linear(1024*2, num_classes)
        self.dropout = torch.nn.Dropout(0.2)
        self.task = task
        if task == 'requests':
            self.preds = []
            self.labels = []

    def compute_features(self, x):
        x = self.model(input_values=x, output_hidden_states=True).hidden_states
        x = torch.stack(x,dim=1)
        weights = torch.nn.functional.softmax(self.layer_weights, dim=-1)
        mean_x = x.mean(dim = 2)
        std_x = x.std(dim = 2)
        x = torch.cat((mean_x, std_x), dim=-1)
        x = (x * weights.view(-1,25,1)).sum(dim=1)
        return x

    def forward(self, x):
        x = self.compute_features(x)
        x = self.dropout(x)
        x = self.linear(x)
        if self.task == 'requests':
            x = torch.softmax(x,dim=-1)
        return x

    def training_step(self, batch,batch_idx):
        x,y = batch
        logits = self.forward(x)
        if self.task == 'requests':
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits,y)
        self.log('train_loss', loss,sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self.forward(x)
        if self.task == 'requests':
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits,y)
        self.log('val_loss', loss,sync_dist=True)
        logits = torch.sigmoid(logits)
        # Metric for emotions
        if self.task == 'emotions':
            coeffs = []
            for i in range(logits.shape[0]):
                coeffs.append(spearmanr(logits[i].detach().cpu().numpy(),y[i].detach().cpu().numpy()).statistic)
            self.log('val_spearman', np.mean(coeffs), sync_dist=True)
        else:
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            self.preds.append(preds)
            self.labels.append(y.detach().cpu().numpy())
        return loss
    
    def on_validation_epoch_end(self):
        if self.task == 'emotions':
            super().on_validation_epoch_end()
            return
        self.preds = np.concatenate(self.preds)
        self.labels = np.concatenate(self.labels)
        self.log('val_recall', recall_score(self.labels,self.preds,average='macro'), sync_dist=True)
        self.preds = []
        self.labels = []

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
    train_df = pd.read_csv('HCC/lab/train.csv')
    dev_df = pd.read_csv('HCC/lab/devel.csv')
    model = AudioModel(2)
    train_dataset = RequestsDataset(train_df, data_path,max_sec = 10,target='request')
    le = train_dataset.le
    dev_dataset = RequestsDataset(dev_df, data_path,max_sec = 30,le=le,target='request')
    checkpoint_callback = ModelCheckpoint(dirpath='com_ckpts',monitor='val_recall',save_top_k=1,mode='max')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=8, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=4, num_workers=8, shuffle=False)
    logger = pl.loggers.WandbLogger(project='requests', name='wav2vec2-large-xlsr-53')
    trainer = pl.Trainer(
        devices=[1,2],
        accelerator='gpu',
        max_epochs=20,
        logger=logger,
        callbacks=[checkpoint_callback],
        strategy='ddp_find_unused_parameters_true',
        precision=16
    )
    trainer.fit(model, train_loader, dev_loader)