# Barretts Esophagus Segmentation
## Initial Setup
 - Open the Src folder and download the checkpoints directory from the given link - https://drive.google.com/drive/folders/1-PepsDxRdAj3duLPw7mrNfL59486m48A?usp=sharing
 (Optional : Not needed for training)

 - Create and activate conda environment
 ``` 
    conda create --name BE
    conda activate BE
 ```
 - Install all packages
 ```
    pip3 install -r requirements.txt
 ```
### Trianing and Testing
- Download the dataset from the following link to the project directory directory - https://drive.google.com/drive/folders/1TH7XBzgwcEavYkLif82P6r5EvMhaMdSr?usp=sharing
- Remove the output directory and all other files and folders from the tmp directory
- Go into Src directory and run
```
    python3 preprocess.py
```
- Preprocess.py contains the augmentations for the dataset and the prepared test, train dataset will be under /tmp/training. 
- Now run train.py
```
    python3 train.py --model <model name> --epochs <no of epochs>
    eg: python3 train.py --model unet --epoch 10
```
- Available models - unet, vgg_unet, resnet50_unet, segnet
- After training outputs and checkpoints are stored under tmp directory 
- Run postPrediction.py. This does post prediction on the output generated and creates the final output in output directory
```
    python3 postPrediction.py --model <model name>
    eg: python3 postPrediction.py --model unet
```

- Available models - unet, vgg_unet, resnet50_unet, segnet, compare(compares all models)

### Using Colab File

- Run the first 2 steps of training and testing step (Preprocessing the dataset) and upload the Whole project into google drive

- Run the colab file