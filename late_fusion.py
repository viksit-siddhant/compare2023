# %%
import torch
import pandas as pd
import lightning.pytorch as pl
import numpy as np

from wav2vec2 import RequestsDataset, EmotionDataset, AudioModel
from sklearn.preprocessing import LabelEncoder
from bert import TranscriptData, Classifier
from sklearn.metrics import recall_score

# %%
class MultiModalData(torch.utils.data.Dataset):
    def __init__(self,df, le = None, target = 'complaint'):
        if le is None:
            self.le = LabelEncoder()
            self.le.fit(df[target])
        else:
            self.le = le
        self.audio_dataset = RequestsDataset(df, data_path='HCC/wav/',le=le,target=target)
        self.text_dataset = TranscriptData(df, le=le,target=target)
    
    def __len__(self):
        return len(self.audio_dataset)
    
    def __getitem__(self, idx):
        audio,labels = self.audio_dataset[idx]
        text0, text1, label_text = self.text_dataset[idx]
        assert (label_text == labels).all()
        return audio, text0, text1, labels

# %%
class LateFusion(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.audio_model = AudioModel.load_from_checkpoint('req_ckpts/wav2vec2-com.ckpt',num_classes=2)
        self.text_model = Classifier.load_from_checkpoint('distilbert_com/distil-com.ckpt')
        self.proj = torch.nn.Linear(1024*2+768*2,1024)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(1024, 2)
        self.preds = np.array([])
        self.labels = np.array([])

    def freeze(self):
        for param in self.audio_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

    def forward(self, audio_input, text_input0, text_input1):
        audio_emb = self.audio_model.compute_features(audio_input)
        text_emb0 = self.text_model.model0(**text_input0)
        text_emb0 = text_emb0.last_hidden_state[:,0,:]
        text_emb1 = self.text_model.model1(**text_input1)
        text_emb1 = text_emb1.last_hidden_state[:,0,:]
        text_emb = torch.cat([text_emb0,text_emb1], dim=1)
        x = torch.cat([audio_emb,text_emb], dim=1)
        x = self.dropout(x)
        x = self.proj(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = torch.softmax(x, dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        audio_input, text_input0, text_input1, y = batch
        y_hat = self(audio_input, text_input0, text_input1)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio_input, text_input0, text_input1, y = batch
        y_hat = self(audio_input, text_input0, text_input1)
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
        return optimizer

if __name__ == '__main__':
    train_df = pd.read_csv('HCC/lab/train.csv')
    dev_df = pd.read_csv('HCC/lab/devel.csv')

    train_dataset = MultiModalData(train_df)
    dev_dataset = MultiModalData(dev_df, le=train_dataset.le)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=4, shuffle=False)

    model = LateFusion()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='late_ckpts',monitor='val_recall',save_top_k=1,mode='max')
    logger = pl.loggers.WandbLogger(project='requests',name='late_fusion')
    trainer = pl.Trainer(
        devices = 3,
        accelerator='gpu',
        callbacks=[checkpoint_callback],
        max_epochs=20,
        strategy='ddp_find_unused_parameters_true',
        precision=16,
        logger=logger
    )

    model.freeze()
    trainer.fit(model, train_loader, dev_loader)