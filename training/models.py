import torch
import torch.nn as nn

from transformers import BertModel
from torchvision import models as vision_models
from sklearn.metrics import precision_score, accuracy_score

from training.meld_dataset import MELDDataset

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # freeze bert parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # 768: the output dimension of BERT
        # 128: the input dimension of fusion model
        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extract BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(
            pretrained=True,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        # num_fts: the output dimension of r3d_18
        # 128: the input dimension of fusion model
        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        # [batch_size, frames, channels, height, width] -> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)

        return self.backbone(x)

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64), # normalize the output of convolutional layer
            nn.ReLU(), # activation function
            nn.MaxPool1d(kernel_size=2), # reduce the data, keep the strongest feature

            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128), # normalize the output of convolutional layer
            nn.ReLU(), # activation function
            nn.AdaptiveAvgPool1d(1), # reduce the data, smooth the feature
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # drop channel information
        x = x.squeeze(1)

        features = self.conv_layers(x)
        # feature outputs: [batch_size, 128, 1]

        # squeeze the last dimension which is 1
        # features: [batch_size, 128]
        return self.projection(features.squeeze(-1))
    

class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),    # compress
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classification heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7),  # 7 emotions output: sadness, neutral, joy, fear, disgust, anger, surprise
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),  # 3 sentiment output: negative, neutral, positive
        )

    def forward(self, text_inputs, video_frames, audio_features):
        text_feature = self.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
        )
        video_feature = self.video_encoder(video_frames)
        audio_feature = self.audio_encoder(audio_features)

        # Concatenate features
        combined_features = torch.cat(
            (text_feature, video_feature, audio_feature), dim=1
        ) # [batch_size, 128*3]

        fused_feature = self.fusion_layer(combined_features)

        # classifier
        emotion_output = self.emotion_classifier(fused_feature)
        sentiment_output = self.sentiment_classifier(fused_feature)

        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output,
        }


class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # log dataset sized
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)

        print("\nDataset sizes:")
        print(f'Training set size: {train_size}')
        print(f'Validation set size: {val_size}')
        print(f"Batches per epoch: {len(train_loader):,}")

        # optimizer
        self.optimizer = torch.optim.Adam(
            [
                {'params': model.text_encoder.parameters(), 'lr': 8e-6},
                {'params': model.video_encoder.parameters(), 'lr': 8e-5},
                {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
                {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
                {'params': model.emotion_classifier.parameters(), 'lr': 5e-4},
                {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4},
            ], 
            weight_decay=1e-5,
        )

        # scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            model="min", 
            factor=0.1, 
            patience=2, # every 2 epochs
        )

        # loss function
        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )
        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

    def train_epoch(self):
        self.model.train()
        running_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}

        for batch in self.train_loader:
            device = next(self.model.parameters()).device
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device),
            }
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)
            
            # zero gradient
            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(
                text_inputs=text_inputs,
                video_frames=video_frames,
                audio_features=audio_features,
            )

            # calculate loss using raw logits
            emotion_loss = self.emotion_criterion(
                outputs['emotions'],
                emotion_labels
            )
            sentiment_loss = self.sentiment_criterion(
                outputs['sentiments'],
                sentiment_labels
            )
            total_loss = emotion_loss + sentiment_loss

            # backward pass. Calculate gradients
            total_loss.backward()

            # gradient clipping: prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            self.optimizer.step()

            # track losses
            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()

        return {
            k: v/len(self.train_loader) for k, v in running_loss.items()
        }

    def evaluate(self, dataloader, phrase='val'):
        self.model.eval()
        losses = {
            'total': 0,
            'emotion': 0,
            'sentiment': 0,
        }
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.no_grad():
            for batch in dataloader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device),
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_labels'].to(device)
                sentiment_labels = batch['sentiment_labels'].to(device)

                # forward pass
                outputs = self.model(
                    text_inputs=text_inputs,
                    video_frames=video_frames,
                    audio_features=audio_features,
                )

                # calculate loss using raw logits
                emotion_loss = self.emotion_criterion(
                    outputs['emotions'],
                    emotion_labels
                )
                sentiment_loss = self.sentiment_criterion(
                    outputs['sentiments'],
                    sentiment_labels
                )
                total_loss = emotion_loss + sentiment_loss

                # Update metrics
                all_emotion_preds.extend(
                    outputs['emotions'].argmax(dim=1).cpu().numpy()
                )
                all_emotion_labels.extend(
                    emotion_labels.cpu().numpy()
                )
                all_sentiment_preds.extend(
                    outputs['sentiments'].argmax(dim=1).cpu().numpy()
                )
                all_sentiment_labels.extend(
                    sentiment_labels.cpu().numpy()
                )

                # Track losses
                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()

        avg_loss = {k: v / len(dataloader) for k, v in losses.items()}

        # Compute the precision and accuracy
        emotion_precision = precision_score(
            all_emotion_labels,
            all_emotion_preds,
            average='weighted',
        )
        emotion_accuracy = accuracy_score(
            all_emotion_labels,
            all_emotion_preds,
        )
        sentiment_precision = precision_score(
            all_sentiment_labels,
            all_sentiment_preds,
            average='weighted',
        )
        sentiment_accuracy = accuracy_score(
            all_sentiment_labels,
            all_sentiment_preds,
        )

        if phrase == 'val':
            self.scheduler.step(avg_loss['total'])

        return avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy,
        }


# cd video-sentiment && python3 -m training.models

if __name__ == '__main__':
    dataset = MELDDataset(
        csv_path='dataset/train/train_sent_emo.csv',
        video_dir='dataset/train/train_splits'
    )
    sample = dataset[0]

    model = MultimodalSentimentModel()
    model.eval()

    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),   #插入一个大小为 1 的维度
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0),
    }
    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            text_inputs=text_inputs,
            video_frames=video_frames,
            audio_features=audio_features,
        )
        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

    emotion_map = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'joy',
        4: 'neutral',
        5: 'sadness',
        6: 'surprise',
    }
    sentiment_map = {
        0: 'negative',
        1: 'neutral',
        2: 'positive',
    }
    
    for i, prob in enumerate(emotion_probs):
        print(f'{emotion_map[i]}: {prob:.4f}')
    
    for i, prob in enumerate(sentiment_probs):
        print(f'{sentiment_map[i]}: {prob:.4f}')

# if __name__ == '__main__':
#     batch_size = 2
#     # 1: channel
#     # 64: frequency bin
#     # 300: time step
#     x = torch.randn(batch_size, 1, 64, 300)
#     print("input shape: ", x.shape)
#     # torch.Size([2, 1, 64, 300])

#     x_squeezed = x.squeeze(1)
#     print("output shape: ", x_squeezed.shape)
#     # torch.Size([2, 64, 300])

