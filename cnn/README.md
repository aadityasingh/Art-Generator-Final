This is a fork of RASTA made in the fall of 2018, with fixes, improvements, and additional parameters that supplement the research and results found by the original paper.

Note that the overall project (Art-Generator-Final) worked only with the ResNet50 section of the code, and that the functionality of the remainder of the code cannot be guaranteed (i.e. the original code source may still have bugs that we, the authors of Art-Generator-Final are unaware of).

We also have a folder `old/` that contains an old implementation of transfer learning from ResNet50 for art style classification in PyTorch. No guarantees that any of it works as expected, as we switched to building on top of the original Tensorflow/Keras implementation. We mostly incorporated for pure documentation, and because it might be useful as a starting point for building a pytorch implementation in the future to create allow for easier interaction with the rest of the project.


# Setup

The data folder should be organized with a separate folders train, validation, and test directory. You may use `python/split_data.py` to easily do so. In each directory should be a series of directories, one per art style, that contain the images of artwork of that style.

# Usage

To train a model using data, use `python/run.py`. You can visualize the loss and accuracy over time for the training and validation dataset via `python/graph_history.py`. The actual code for training is found in `python/models/processing.py`.

To test a model using data, use `python/evaluation.py`. 

## Options for Training

* Training directory `--train_path`: path to training data
* Validation directory `--val_path`: path to validiation data
* Model `-m`: we've currently only leveraged the `resnet` option, as the original RASTA project found this model, ResNet50 to work the best among the models tested. Other options include AlexNet, Inception, and various other sizes of ResNet. New models should be written in the `python/models` directory and imported into `run.py` for use.
* Batch size `-b`: batch size
* Epochs `-e`: number of epochs
* Horizontal flips `-f`: whether to augment the dataset with horizontal flips or not
* Optimizer `--optim`: optimizer to choose
    * Momentum `--mom`: momentum for SGD momentum optimizer

*Training Layer Options:*

* Number of retrainable layers `-n`: number of retrainable layers in the model
* Start of retrainable layers `--start-layer`: first retrainable layer. If None, by default the last layers are retrainable.

*Currently haven't unused and left as default:*

* Dropout `-d`: dropout rate (unused)
* Distortions `--distortions`: distortions
* ImageNet preprocessing `-p`: use ImageNet preprocessing or not

## Options for Testing

* Type of evaluation `-t`: accuracy calculation or prediction
* Top-k numbers `-k`: In accuracy mode, this can calculate the accuracy for a list of top-k accuracies. In prediction mode, only expects one number (takes the first number if an array) and predicts the top k predictions.
* Path to data `--data_path`: path to test data
* Path to model `--model_path`: path to saved model
* Save to JSON `-j`: only for prediction mode
* Save to results file `-s`: only for accuracy mode

*Currently haven't unused and left as default:*

* Bagging `-b`
* Type of preprocessing `-p`

## TODO: Confusion Matrix

Currently not working :(


If you are looking for the original README.md of RASTA, please navigate to RASTA_README.md



