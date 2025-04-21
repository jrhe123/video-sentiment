## Dataset

- MELD: Multimodal EmotionLines Dataset
    - https://affective-meld.github.io/
    - curl -O https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
    - tar -xzvf MELD.Raw.tar.gz
    - cd MELD.Raw && tar -xzvf dev.tar.gz && tar -xzvf test.tar.gz && tar -xzvf train.tar.gz
    - mv MELD.Raw dataset

## Architecture
1. Video encoder: ResNet3D 18 layer
2. Text encoder: BERT
3. Audio encoder: Raw spectrogram

## Fusion
1. Late fusion:
```
modality 1 -> ml model 1
                        -> fusion layer -> prediction
modality 2 -> ml model 2
```

2. Early fusion:
```
modality 1 -> feature extraction 
                                -> concatenated vector -> ml model -> prediction
modality 2 -> feature extraction
```

3. Intermediate fusion:
```
modality 1 -> feature extraction -> ml model 1
                                              -> fusion layer  -> ml model -> prediction
modality 2 -> feature extraction -> ml model 2
```

## Architecture 2
```
Video encoder (ResNet3D 18 layer) [batch_size, 128]

Text encoder (BERT) [batch_size, 128]               -> concatenate data [batch_size, 384] -> fusion [learn relation between modalities] -> emotion classifer [batch_size, 7] (e.g., 7 emotions joy/sadness/etc..) & sentiment classifier [batch_size, 3] (e.g., positive/negative/neutral)

Audio encoder (Raw spectrogram) [batch_size, 128]

```

## Mel spectrogram
- 梅尔倒频谱

## Dataset

1. Feature Text: load text from row -> tokenize utterance -> return dict
2. Feature Video: find video in folder -> extact frames -> normalize RGB values -> one tensor with all frames -> return dict (normalize RGB values to 0~1)
3. Feature Audio: -> .mp4 to .wav -> extract audio from video -> create mel spectrogram -> normalize mel spectrogram -> return dict
4. Label: load label from row 0 -> map emotion and sentiment to numbers -> return dict

{
    "text": xyz,
    "video_frames": xyz,
    "audio_features": xyz,",
    "emotion_label": "sadness",
    "sentiment_label": "negative"
}

## Fusion
- Dimension: 128

## Test models
- python3 -m training.models

## Test logger
- python3 -m training.test_logging
- tensorboard --logdir runs
- http://localhost:6006/

## Count model parameters
- python3 -m training.count_parameters
