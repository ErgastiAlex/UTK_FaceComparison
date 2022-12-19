A.A. 2022/2023
Deep Learning and Generative Models
Project assignment #6

## Project objective:

- Train a model that, given two face images, outputs who is younger and who is older.

## Dataset:

- UTK-Face: https://susanqq.github.io/UTKFace/

## Network model:

- A standard CNN such as VGG or ResNet should work

## Detailed information:

- The model should not predict the age, but can be trained to do so if useful;
- Use UTK-face to build a custom dataset with relation labels (younger, older) associated to the face pairs.
- Many different solutions are possible. Two options are:
  - Modify the CNN to receive as input two images and train it as binary classifier;
  - Train the CNN as age estimator and use it as feature extractor to compare 2 latent vectors corresponding to two faces by training a simple MLP.

## Additional notes:

- Two images can be concatenated in the channel dimension to form a single tensor
