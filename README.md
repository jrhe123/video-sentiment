## Dataset

- MELD: Multimodal EmotionLines Dataset
    - https://affective-meld.github.io/
    - curl -O https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
    - tar -xzvf MELD.Raw.tar.gz
    - cd MELD.Raw && tar -xzvf dev.tar.gz && tar -xzvf test.tar.gz && tar -xzvf train.tar.gz

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