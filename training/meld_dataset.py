import os
import cv2
import torch
import torchaudio
import subprocess
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir): 
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6,
        }
        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2,
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
        video_path = os.path.join(self.video_dir, video_filename)

        video_path_exists = os.path.exists(video_path)
        if not video_path_exists:
            print(f"Video {video_path} not found")
            raise FileNotFoundError(f"Video {video_path} not found")
        
        # Tokenize text
        text_inputs = self.tokenizer(
            row['Utterance'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt',
        )

        # Video frames
        # video_frames = self._load_video_frames(video_path)

        # Audio
        audio_features = self._extract_audio_features(video_path)

    def _extract_audio_features(self, video_path):
        audio_path = video_path.replace('.mp4', '.wav')
        try:
            subprocess.run(
                [
                    'ffmpeg', 
                    '-i', video_path, 
                    '-vn', 
                    '-acodec', 'pcm_s16le', 
                    '-ar', '16000',
                    '-ac', '1',
                    audio_path,
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            waveform, sample_rate = torchaudio.load(audio_path)

            # common speech rate is 16000
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512,
            )
            mel_spec = mel_spectrogram(waveform)

            # fix the input size to 300
            # channel, frequency bin, timestamp
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]


            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error extracting audio features from video {video_path}: {e}")

        except Exception as e:
            raise ValueError(f"Error extracting audio features from video {video_path}: {e}")
        
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Error opening video {video_path}")
            
            # Check the first frame is validate
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Error reading video {video_path}")
            
            # Reset cap to first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0 # normalized to 0~1
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Error loading video {video_path}: {e}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"No valid frames found in video {video_path}")
        
        # Pad or truncate frames
        if len(frames) < 30:
            frames = [
                np.zeros_like(
                    frames[0]
                )
            ] * (30 - len(frames))
        else:
            frames = frames[:30]

        # before permute: [frames, height, width, channels]
        # after permute: [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)



if __name__ == '__main__':
    meld = MELDDataset(
        csv_path='../dataset/dev/dev_sent_emo.csv',
        video_dir='../dataset/dev/dev_splits_complete',
    )
    # get item
    print(meld[0])
