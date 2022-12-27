# Project Assignment

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

# Project solution

## Dataset

The UTKFace dataset isn't provided inside the repo, but `split_dataset.py` can be used to split the dataset in 3:

- Train set
- Validation set
- Test set

This script must be executed before the `main.py`

## UTKDataset class

The UTKDataset is higly configurable with this input parameters:

- `root_dir` is a string representing the directory containing the images.
- `transform` is an optional transform to be applied to each image in the dataset.
- `seed` is an optional seed for the random number generator.
- `year_diff` is the minimum age difference between images in the dataset.
- `data_size` is the size of the dataset. If data_size is `None` or negative, the dataset will be the maximum possible size. If `unique_images` is `True`, the required size of the dataset may not be reached.
- `duplicate_probability` is the probability of duplicating a combination of images by switching the order. The duplication will result in a dataset that is larger by a factor of 1 | duplicate_probability.
- `unique_images` is a boolean indicating whether or not to use each image only once in the dataset.

The model proposed are trained with:

- `unique_images` to false, to be able to propose more combination example to the model
- `year_diff` equal to 1, to put the model in the most difficult case
- `duplicate_probability` equal to 0, to avoid putting to many duplicates
- `data_size` equal to 100k, to keep the training time reasonable

This parameters has been chosen to improve the number of combination examples shown to the ResNet

## UTKAgeDataset class

TODO

# Solver

## Basic Solver

This is a basic solver that train the model with the given loss, it has the following input parameters:

- train_loader,
- test_loader,
- device,
- model,
- writer,
- args

it is able to recover a training of a model.

It implements a mechanism of early stopping using a parameter called `patience`, if the model hasn't improved on the validation set for `patience` epochs, it stops the search for a better model.

## AutoSolver

This solver is able to search for hyperparameters autonomously. To use this solver the package ray[tune] must be installed.
It has the following input parameters:
training_set, test_set,val_set , model_class, writer, args

# Model

## SiameseResNet

This model consists of two identical ResNet use to extract a feature map from each image, the feature map are then flatten, concatenated and passed to a FC layers to classify if the left image is older or not.

## ResNetClassifier

This model consists of a ResNet that takes as an input a image with 6 channels, this image is composed of the two images concatenated along the channel axis, at the end of the resnet there are FC layers to classifyt if the left image is older or not.

## SiameseResNetAge Classifier

This model is very similar to the SiameseResNet, but it uses a pretrained ResNet that classify the age of a given face.
During the training the model will one train the FC layers and not the ResNet.

# Ablation Experiment

This experiment has been produced by the AutoSolver using Ray Tune.
All the models have been trained using the following dataset dimension:

| Dataset        | Dim  |
| -------------- | ---- |
| Training set   | 100k |
| Validation set | 5k   |
| Test set       | 5k   |

## SiameseResNet

| Trial name            | batch_size | dropout_prob | hidden_layers   | lr          | resnet_type | use_dropout | weight_decay | iter | total time (s) | loss         | accuracy   | AUC         |
| --------------------- | ---------- | ------------ | --------------- | ----------- | ----------- | ----------- | ------------ | ---- | -------------- | ------------ | ---------- | ----------- |
| \_\_train_d46fc_00000 | 64         | 0.200681     | [512, 256, 128] | 0.00528511  | resnet18    | &cross;     | 5.06796e-05  | 10   | 7187.98        | 0.416078     | 0.825      | 0.914002    |
| \_\_train_d46fc_00001 | 32         | 0.419782     | [512, 256, 128] | 0.000180967 | resnet50    | &check;     | 1.34209e-05  | 1    | 868.455        | 0.445902     | 0.786      | 0.87629     |
| \_\_train_d46fc_00002 | 64         | 0.205812     | [512, 256, 128] | 0.008269    | resnet50    | &check;     | 7.60351e-06  | 1    | 761.974        | 0.680393     | 0.5606     | 0.593323    |
| \_\_train_d46fc_00003 | 32         | 0.269371     | [512, 256, 128] | 0.0169636   | resnet50    | &check;     | 4.10899e-05  | 1    | 865.805        | 0.694148     | 0.4972     | 0.491982    |
| \_\_train_d46fc_00004 | 16         | 0.532978     | [64, 32, 16]    | 0.000643042 | resnet18    | &check;     | 5.92368e-06  | 1    | 766.598        | 0.604223     | 0.7224     | 0.775198    |
| \_\_train_d46fc_00005 | 32         | 0.340338     | []              | 0.00247932  | resnet18    | &check;     | 5.58359e-06  | 4    | 3031.08        | **0.387461** | **0.8294** | **0.91727** |
| \_\_train_d46fc_00006 | 32         | 0.255038     | []              | 0.00310665  | resnet50    | &check;     | 0.00173744   | 1    | 851.608        | 0.538291     | 0.7292     | 0.805117    |
| \_\_train_d46fc_00007 | 4          | 0.264651     | [512, 256, 128] | 0.00170083  | resnet18    | &check;     | 0.000681391  | 1    | 1547.35        | 0.693506     | 0.5        | 0.487549    |
| \_\_train_d46fc_00008 | 64         | 0.433996     | [64, 32, 16]    | 0.000605203 | resnet50    | &cross;     | 0.00189003   | 2    | 1473.61        | 0.536957     | 0.7412     | 0.828701    |

The best model is a simple resnet18.
Best trial config:

```json
{
  "lr": 0.002479324241152825,
  "batch_size": 32,
  "hidden_layers": [],
  "use_dropout": true,
  "dropout_prob": 0.3403376632221325,
  "weight_decay": 5.58358886230025e-6,
  "resnet_type": "resnet18"
}
```

Best trial final validation loss: 0.38746090527552707
Best trial final validation accuracy: 0.8294
Best trial test set accuracy: 0.8234, AUC_score: 0.9140750400000001

The accuracy is calculated using a threshold of 0.5

The model was retrained from scratch and gets this performances:

| Dataset    | Loss                | Accuracy           | AUC        |
| ---------- | ------------------- | ------------------ | ---------- |
| Validation | 0.3589838809648137  | 0.8411624203821656 | 0.92645456 |
| Test       | 0.36990593430722596 | 0.8234474522292994 | 0.92036864 |

The model is saved as `models\SiameseResNetClassifier_ResNet18\best_model.pth`

## ResNetClassifier

| Trial name            | batch_size | dropout_prob | hidden_layers   | lr          | resnet_type | use_dropout | weight_decay | iter | total time (s) | loss         | accuracy | AUC          |
| --------------------- | ---------- | ------------ | --------------- | ----------- | ----------- | ----------- | ------------ | ---- | -------------- | ------------ | -------- | ------------ |
| \_\_train_78f21_00000 | 64         | 0.200681     | [512, 256, 128] | 0.00528511  | resnet18    | &cross;     | 5.06796e-05  | 10   | 4071.29        | **0.438381** | 0.7966   | 0.883987     |
| \_\_train_78f21_00001 | 32         | 0.419782     | [512, 256, 128] | 0.000180967 | resnet50    | &check;     | 1.34209e-05  | 2    | 1094.55        | 0.510653     | 0.7504   | 0.838142     |
| \_\_train_78f21_00002 | 64         | 0.205812     | [512, 256, 128] | 0.008269    | resnet50    | &check;     | 7.60351e-06  | 1    | 457.527        | 0.665431     | 0.5982   | 0.642916     |
| \_\_train_78f21_00003 | 32         | 0.269371     | [512, 256, 128] | 0.0169636   | resnet50    | &check;     | 4.10899e-05  | 1    | 552.325        | 0.697621     | 0.5      | 0.494179     |
| \_\_train_78f21_00004 | 16         | 0.532978     | [64, 32, 16]    | 0.000643042 | resnet18    | &check;     | 5.92368e-06  | 1    | 420.208        | 0.627728     | 0.6724   | 0.704657     |
| \_\_train_78f21_00005 | 32         | 0.340338     | []              | 0.00247932  | resnet18    | &check;     | 5.58359e-06  | 10   | 4119.39        | 0.483091     | **0.8**  | **0.897777** |
| \_\_train_78f21_00006 | 32         | 0.255038     | []              | 0.00310665  | resnet50    | &check;     | 0.00173744   | 1    | 539.234        | 0.601481     | 0.683    | 0.745532     |
| \_\_train_78f21_00007 | 4          | 0.264651     | [512, 256, 128] | 0.00170083  | resnet18    | &check;     | 0.000681391  | 1    | 1196.29        | 0.693234     | 0.5      | 0.499012     |
| \_\_train_78f21_00008 | 64         | 0.433996     | [64, 32, 16]    | 0.000605203 | resnet50    | &cross;     | 0.00189003   | 1    | 498.209        | 0.667643     | 0.5944   | 0.629829     |
| \_\_train_78f21_00009 | 128        | 0.597436     | [512, 256, 128] | 0.0891902   | resnet50    | &check;     | 0.000687784  | 1    | 466.556        | 50.4337      | 0.4998   | 0.50039      |

A simple resnet18 has a great accuracy and AUC, but an higher loss compared to a resnet18 with some hidden layers.

The model with the best loss has this configuration:

```json
{
  "lr": 0.005285108213178958,
  "batch_size": 64,
  "hidden_layers": [512, 256, 128],
  "use_dropout": false,
  "dropout_prob": 0.20068111676973313,
  "weight_decay": 5.067959425940853e-5,
  "resnet_type": "resnet18"
}
```

Best trial final validation loss: 0.43838050320178645
Best trial final validation accuracy: 0.7966
Best trial test set accuracy: 0.767, AUC_score: 0.8535112

The best accuracy and AUC score is achived by a simple resnet without any hidden layers, this proves that a resnet is enough to achieve a good performance on the task. This is probably because a resnet18 is able to extract a meaningful feature vector that can be classified with a simple MLP.

# Conclusion

Overall, between the Siamese and the modified ResNet, the best model is the first one, this is probably because the Siamese extract 2 feature map instead of a single one, as in the ResNet. Hence it is able to provide to the MLP more information.
