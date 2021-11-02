# learning discrete geometry operators

This repo include the code for the paper

“Learning Geometric Operators on Meshes”
International Conference on Learning Representations (ICLR) Workshop 2019
Yu Wang, Vladimir Kim, Michael Bronstein and Justin Solomon

https://rlgm.github.io/papers/28.pdf


Download the data and unzip:

MNIST dataset:
`https://www.dropbox.com/s/lnsuu8xyqlcaqr8/mnist-mesh.zip?dl=0`
unzip files to `./data/mnist-mesh`

ARAP animation dataset is from:
`https://github.com/jiangzhongshi/SurfaceNetworks`
unzip files to `./data/data_plus`

To run the code:

`cd example`

`python3 train.py  --model=model_mnist_cls_241zld-48 --batch_size=32 --gpu=0`

`python3 train.py --batch_size=16 --model=model_surfnet_arap_dir_lr_101 --gpu=0`

Tested with tensorflow-1.9.0 and python 3.5

For questions and comments, email: wangyu9@mit.edu
