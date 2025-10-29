<h1 align='center'>gradient-weight alignment</h1>

## Preparing Datasets
If you want to use our dataloading plug and play, you need to preprocess some of the datasets.
The files in this directory do the following:
- `check_jpeg.py`: There are a few files with corrupted headers in standard ImageNet-1k that can lead to problems when using the `tf.image.decode_and_crop_jpeg` augmentation. This script should find those files and fix the headers.
- `cifar.py`: Create a train-val split based on the train set of CIFAR-10/100 such that the val set can be used as train set. The script also allows one to do a corresponding split for the noisy labels in CIFAR-N.
- `cifar_c.py`: Prepare CIFAR-C to test model robustness.
- `imagenet.py`: Prepare ImageNet-V2 and ImageNet-R to test model robustness.
- `inat.py`: Prepare the iNaturalist18 dataset for fine-tuning.