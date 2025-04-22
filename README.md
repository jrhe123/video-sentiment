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

## SageMaker
- EC2:
    - download dataset to S3
- S3:
    - datasets
    - tensorboard runs & download back to local machine
    - setup CORS for allowed origins
- SageMaker:
    - training jobs: instance type `ml.g5.xlarge`
    - run endpoint -> inference
- IAM:
    - roles to access S3 / SageMaker deploy & invoke endpoint

## AWS console
1. SageMaker
- request quota
    - ml.g5.xlarge for training job usage -> increase quota value to 1
    - ml.g5.xlarge for endpoint usage -> increase quota value to 1

2. S3 bucket
- create bucket
    - general purpose
    - public access
- create folders
    - dataset/
    - tensorboard/
- edit bucket's permission
    - CORS
    ```
    [
        {
            "AllowedHeaders": ["*"],
            "AllowedMethods": ["GET", "PUT", "POST"],
            "AllowedOrigins": ["*"],
            "ExposeHeaders": []
        }
    ]
    ```

3. IAM
- create role
    - search -> sagemaker -> `AmazonSageMakerFullAccess`
    - name `sentiment-analysis-execution-role`
    - copy the ARN `arn:aws:s3:::sentiment-analysis-saas`
- create inline policy
    - get object
    - delete object
    - put object
    - list bucket
    - specify bucket with above ARN
    - apply to ANY object
    - create policy `sentiment-analysis-execution-s3-policy`

- create role
    - search -> sagemaker -> `AmazonSageMakerFullAccess`
    - name `sentiment-analysis-deploy-endpoint-role`
- attach policy
    - `cloudWatchLogsFullAccess`
- create inline policy
    - same as above, so we can copy & paste it as JSON

- create policy
    - name `sentiment-analysis-deploy-endpoint-s3-policy`


## AWS cli
- aws configure
- start the training job
    - python3 train_sagemaker.py
    - check the logs in cloudwatch
- download the model
    - aws s3 cp s3://sentiment-analysis-saas/tensorboard/runs/2022-04-04_09-08-07/events.out.tfevents.1649168487.ip-172-31-13-250.ec2.internal.tar.gz .
- sync tensorboard logs to local folder
    - aws s3 sync s3://sentiment-analysis-saas/tensorboard/ ./tensorboard_logs
    - tensorboard --logdir tensorboard_logs


## References
- https://www.youtube.com/watch?v=Myo5kizoSk0
- BE: https://github.com/Andreaswt/ai-video-sentiment-model/tree/main
- FE: https://github.com/Andreaswt/ai-video-sentiment-saas
- Pre-trained models: https://drive.google.com/drive/folders/1f5tOlIixDUeYtzzIdctQRb_-qllzAMQd
