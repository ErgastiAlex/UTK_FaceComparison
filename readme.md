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

# Project description

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

TODO: Add some prouf

The model proposed are trained with:

- `unique_images` to false, to be able to propose more combination example to the model
- `year_diff` equal to 1, to put the model in the most difficult case
- `duplicate_probability` equal to 0, to avoid putting to many duplicates
- `data_size` equal to 100k, to keep the training time reasonable

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

## ResNetClassifier

# Ablation Experiment

This experiment has been produced by the AutoSolver using Ray Tune

## SiameseResNet

| Trial name            | batch_size | dropout_prob | hidden_layers   | lr          | resnet_type | use_dropout | weight_decay | iter | total time (s) | loss         | accuracy  |
| --------------------- | ---------- | ------------ | --------------- | ----------- | ----------- | ----------- | ------------ | ---- | -------------- | ------------ | --------- |
| \_\_train_176e7_00000 | 64         | 0.200681     | [512, 256, 128] | 0.00528511  | resnet18    | False       | 5.06796e-05  | 10   | 3671.92        | 0.644346     | **0.824** |
| \_\_train_176e7_00001 | 32         | 0.419782     | [512, 256, 128] | 0.000180967 | resnet50    | True        | 1.34209e-05  | 1    | 851.986        | 0.43919      | 0.7796    |
| \_\_train_176e7_00002 | 64         | 0.205812     | [512, 256, 128] | 0.008269    | resnet50    | True        | 7.60351e-06  | 1    | 741.517        | 0.581454     | 0.678     |
| \_\_train_176e7_00003 | 32         | 0.269371     | [512, 256, 128] | 0.0169636   | resnet50    | True        | 4.10899e-05  | 1    | 854.674        | 50.0597      | 0.5       |
| \_\_train_176e7_00004 | 16         | 0.532978     | [64, 32, 16]    | 0.000643042 | resnet18    | True        | 5.92368e-06  | 1    | 522.538        | 0.693192     | 0.5028    |
| \_\_train_176e7_00005 | 32         | 0.340338     | []              | 0.00247932  | resnet18    | True        | 5.58359e-06  | 2    | 717.701        | **0.402612** | 0.8062    |
| \_\_train_176e7_00006 | 32         | 0.255038     | []              | 0.00310665  | resnet50    | True        | 0.00173744   | 1    | 845.759        | 0.514387     | 0.7518    |
| \_\_train_176e7_00007 | 4          | 0.264651     | [512, 256, 128] | 0.00170083  | resnet18    | True        | 0.000681391  | 2    | 3479           | 0.468881     | 0.7974    |
| \_\_train_176e7_00008 | 64         | 0.433996     | [64, 32, 16]    | 0.000605203 | resnet50    | False       | 0.00189003   | 2    | 1477.42        | 0.407465     | 0.7964    |

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

Performance of the best model:
| validation loss | validation accuracy |
| ------------------- | ------------------- |
| 0.40261170211111663 | 0.8062 |

## ResNetClassifier

| Trial name            | batch_size | dropout_prob | hidden_layers   | lr          | resnet_type | use_dropout | weight_decay | iter | total time (s) | loss         | accuracy   |
| --------------------- | ---------- | ------------ | --------------- | ----------- | ----------- | ----------- | ------------ | ---- | -------------- | ------------ | ---------- |
| \_\_train_5ea5d_00000 | 64         | 0.200681     | [512, 256, 128] | 0.00528511  | resnet18    | &cross;     | 5.06796e-05  | 10   | 3646.47        | 0.733621     | **0.7942** |
| \_\_train_5ea5d_00001 | 32         | 0.419782     | [512, 256, 128] | 0.000180967 | resnet50    | &check;     | 1.34209e-05  | 4    | 2143.92        | **0.442943** | 0.7858     |
| \_\_train_5ea5d_00002 | 64         | 0.205812     | [512, 256, 128] | 0.008269    | resnet50    | &check;     | 7.60351e-06  | 1    | 445.046        | 0.66125      | 0.6238     |
| \_\_train_5ea5d_00003 | 32         | 0.269371     | [512, 256, 128] | 0.0169636   | resnet50    | &check;     | 4.10899e-05  | 1    | 535.922        | 4.77258      | 0.504      |
| \_\_train_5ea5d_00004 | 16         | 0.532978     | [64, 32, 16]    | 0.000643042 | resnet18    | &check;     | 5.92368e-06  | 1    | 378.15         | 0.650033     | 0.644      |
| \_\_train_5ea5d_00005 | 32         | 0.340338     | []              | 0.00247932  | resnet18    | &check;     | 5.58359e-06  | 2    | 716.849        | 0.521344     | 0.7452     |
| \_\_train_5ea5d_00006 | 32         | 0.255038     | []              | 0.00310665  | resnet50    | &check;     | 0.00173744   | 2    | 1059.43        | 0.535556     | 0.718      |
| \_\_train_5ea5d_00007 | 4          | 0.264651     | [512, 256, 128] | 0.00170083  | resnet18    | &check;     | 0.000681391  | 1    | 1173.96        | 0.693327     | 0.5        |
| \_\_train_5ea5d_00008 | 64         | 0.433996     | [64, 32, 16]    | 0.000605203 | resnet50    | &cross;     | 0.00189003   | 4    | 1778.04        | 0.49628      | 0.7682     |
| \_\_train_5ea5d_00009 | 128        | 0.597436     | [512, 256, 128] | 0.0891902   | resnet50    | &check;     | 0.000687784  | 1    | 392.492        | 49.707       | 0.5        |

Best trial config:

```json
{
  "lr": 0.00018096700138086342,
  "batch_size": 32,
  "hidden_layers": [512, 256, 128],
  "use_dropout": true,
  "dropout_prob": 0.41978191409759136,
  "weight_decay": 1.342085547606527e-5,
  "resnet_type": "resnet50"
}
```

Performance of the best model:
| validation loss | validation accuracy |
| ------------------- | ------------------- |
| 0.44294281541162234 | 0.7858 |

# Conclusion
