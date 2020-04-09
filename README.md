# microscopy
Quality Assessment and Restoration of High-Content Microscopy Images using Deep Learning

## Data availabilty
Dataset and pre-trained models can be downloaded from Google Drive
https://drive.google.com/open?id=1fAr5SKCC1NLpz5rDr34tOqbw0-Q9DxtF

pre-trained models:
* AlexNet_224x224.h5
* VGG16_unbalanced_224x224.h5
* VGG16_balanced_600x600.h5

datasets:
* dataset_orig
* dataset1 (used in training above pre-trained models)
* autoencoder_dataset (used in training autoencoder)

## Image evaluation
**script: evaluation/evaluate_microscopy_images.py**

This script is a CLI which allows user to select a pre-trained model and use it to classify images. To run the script,
 follow the steps below

1. Create a conda environment and activate it
    ```bash
    $ conda env create -n deeplearning_evaluate -f evaluation/environment.yml
    $ conda activate deeplearning_evaluate
    ```
 
2. Download the pre-trained models and save to 'evaluation' folder
   <br/>[provide Google Drive link to the two models]
   
   model_AlexNet.h5 (image size = 224)
   <br/> 
   model_VGG16.h5 (image size = 150)
      
3. Run the script
    
    ```bash
    $ python evaluation/evaluate_microscopy_images.py -m </path/to/model.h5> -i </path/to/image/directory>
      -s <image_size> -o </path/to/output/directory>
    ```
   example (AlexNet model):
   ```bash
   $ python evaluation/evaluate_microscopy_images.py -m evaluation/model_AlexNet.h5 -i /home/myra/DeepLearning/dataset_orig/good -s 224 -o /home/myra/DeepLearning
   ```
   
   example (VGG16 model):
   ```bash
   $ python evaluation/evaluate_microscopy_images.py -m evaluation/model_VGG16.h5 -i /home/myra/DeepLearning/dataset_orig/good -s 150 -o /home/myra/DeepLearning
   ```

4. Check result file
    
    The output CSV file will be saved in the specified output directory. It is a spreadsheet with columns:
    * Image Path (image file location)
    * Top Prediction (class with highest prediction score)
    * Good (prediction score)
    * Blurred (prediction score)
    * Empty (prediction score)
    
    See example output file: evaluation/image_classification.csv