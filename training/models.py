import torch
import torch.nn as nn

from transformers import BertModel
from torchvision import models as vision_models

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
        x = self.backbone(x)

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
            nn.BatchNorm1d(64), # normalize the output of convolutional layer
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

    def forward(self, text_input, video_frames, audio_features):
        text_feature = self.text_encoder(
            text_input["input_ids"],
            text_input["attention_mask"],
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



if __name__ == '__main__':
    batch_size = 2
    # 1: channel
    # 64: frequency bin
    # 300: time step
    x = torch.randn(batch_size, 1, 64, 300)
    print("input shape: ", x.shape)
    # torch.Size([2, 1, 64, 300])

    x_squeezed = x.squeeze(1)
    print("output shape: ", x_squeezed.shape)
    # torch.Size([2, 64, 300])

