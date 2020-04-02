import tensorflow as tf
from keras.models import load_model
from skimage.io import imread
from skimage import transform
import pandas as pd
import numpy as np
import os

# hide warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Microscopy Image Classification: good, empty or blurred')
    parser.add_argument('-m', '--model-path', help='Path to the model which will be used for evaluation')
    parser.add_argument('-i', '--image-dir', help='Path to the image directory')
    parser.add_argument('-o', '--output-dir', help='Path where to save output file')
    parser.add_argument('-s', '--image-size', type=int, help='Image size used during the model training')

    args = parser.parse_args()
    model_path = args.model_path
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = args.image_size

    # Basic checks
    if not os.path.isfile(model_path):
        parser.error("Model '%s' does not exist" % model_path)
    if not os.path.isdir(image_dir):
        parser.error("Image path '%s' does not exist" % image_dir)
    if not os.path.isdir(output_dir):
        parser.error("Output directory '%s' does not exist" % output_dir)

    # Check image files
    image_files = sorted(filter(lambda x: (x.endswith('.jpeg') or x.endswith('.png') or x.endswith('.tif') or
                                           x.endswith('.tiff') or x.endswith('.flex')), os.listdir(image_dir)))
    image_count = len(image_files)
    if image_count == 0:
        parser.error("No images found from directory '%s'" % image_dir)

    # Load the model
    model = load_model(model_path)

    # Loop through all images and make an array stack
    classes = ['good', 'empty', 'blurred']
    to_concat = []
    for image in image_files:
        imagepath = os.path.join(image_dir, image)
        imagearray = transform.resize(imread(imagepath), (image_size, image_size, 3))
        to_concat.append(imagearray)
    stacked_images = np.stack(to_concat)
    y_pred = model.predict(stacked_images)

    # Create a dataframe with prediction results
    image_pred = {'Image Path': [], 'Top Prediction': [], 'Good': [], 'Blurred': [], 'Empty': []}
    for y_idx, pred in enumerate(y_pred):
        pred_max = int(np.argmax(pred))
        pred_class = classes[pred_max].upper()
        image_pred['Image Path'].append(os.path.join(image_dir, image_files[y_idx]))
        image_pred['Top Prediction'].append(pred_class)
        image_pred['Good'].append(pred[0])
        image_pred['Blurred'].append(pred[2])
        image_pred['Empty'].append(pred[1])
    out_df = pd.DataFrame.from_dict(image_pred)

    # Save dataframe to a CSV file
    out_df.to_csv(os.path.join(output_dir, 'image_classification.csv'), index=False)


if __name__ == '__main__':
    main()
