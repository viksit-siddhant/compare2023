'''
Generating transcripts
'''

import librosa
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from tqdm import tqdm

device = torch.device('cuda:0')
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french").to(device)
debug_file = open("debug.txt","w")

def get_files(folder):
    file_dict = {}
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if not filename.endswith('.wav'):
                continue
            filepath = os.path.join(root,filename)
            file_dict[filename] = filepath
    return file_dict

def transcribe(signal):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            signal = torch.from_numpy(signal).to(device)
            inputs = processor(signal, sampling_rate=16000, return_tensors="pt",padding=True).to(device)
            logits = model(inputs.input_values.type(torch.float16),attention_mask=inputs.attention_mask).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
    return transcription

def load_audio(raw_folder,trans_folder):
    files = get_files(raw_folder)
    for file,file_path in tqdm(files.items()):
        trans_path_0 = os.path.join(trans_folder,file.replace('.wav','0.txt'))
        trans_path_1 = os.path.join(trans_folder,file.replace('.wav','1.txt'))
        if os.path.exists(trans_path_0):
            continue
        audio_input, _ = librosa.load(file_path, sr=16000,mono=False)
        channel_0 = audio_input[0]
        channel_1 = audio_input[1]
        trans_0 = transcribe(channel_0)
        trans_1 = transcribe(channel_1)
        with open(trans_path_0,'w') as f:
            f.write(trans_0)
        with open(trans_path_1,'w') as f:
            f.write(trans_1)

if __name__ == "__main__":
    raw_folder = "HCC/raw"
    trans_folder = "HCC/transcripts"
    load_audio(raw_folder,trans_folder)