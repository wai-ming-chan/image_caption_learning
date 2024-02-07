# Image-Caption-Learning Project

## Project Overview
  This project (on the course EECS738 Machine Learning 2023 Spring) focuses on generating descriptive captions for images using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The main goal is to bridge the gap between visual understanding and natural language processing by training a model that can accurately describe the content of an image in English.

## Features
- Utilizes a pre-trained ResNet50 model as the encoder to extract features from images.
- Employs an LSTM-based decoder to generate descriptive captions based on the features extracted by the CNN.
- Includes functionality to evaluate the model's performance using BLEU scores for a comprehensive assessment of the generated captions' quality.

## Dataset
  The project uses the Flickr8k dataset, which is not included in the repository. Please download the dataset and place it in the `data/flickr8k` directory with subdirectories for `images` and `captions.txt`.

## Usage
  - Train the model by runnig the file `resnet2RNN-train.ipynb` or `resnet2RNN-train.py`
  
  - Result is stored in the file `resnet-output_3e4.txt`

  Make sure to adjust the paths in the code to match your dataset's location.

## How It Works
- The `EncoderCNN` class uses ResNet50 to extract feature vectors from input images.
- The `DecoderRNN` class generates captions based on the features provided by the encoder.
- The `CNNtoRNN` class combines the encoder and decoder for end-to-end training and caption generation.
- The script includes functions for saving and loading model checkpoints, printing example captions, and calculating BLEU scores for evaluation.

## License
MIT

## Acknowledgments
- The project utilizes the transformers library by Hugging Face for tokenization.
- The CNN architecture is based on the ResNet50 model, and the RNN utilizes LSTM for sequence generation.




