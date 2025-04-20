## Dataset

- MELD: Multimodal EmotionLines Dataset
    - https://affective-meld.github.io/
    - curl -O https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

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

Text encoder (BERT) [batch_size, 128]

Audio encoder (Raw spectrogram) [batch_size, 128]

```