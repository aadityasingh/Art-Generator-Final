This is a GAN trained using cycle consistency loss. 


# Setup

The data folder should be organized with a separate folders organized as  /data/train/[art_movement] and /data/test/[art_movement] where art_movement is the name of the Art Movement in question. These are directories with the art.

# Usage

To train a model using data, use `python cycle_gan.py`. 

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






