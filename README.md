# resnet-food-101-theano

## Downloads

Download cifar-10 dataset :

```bash
cd /sharedfiles
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xvzf cifar-10-python.tar.gz
```

Download the Food 101 dataset :

```bash
cd /sharedfiles
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar xvzf food-101.tar.gz
```

and resize dataset so that all images have same width and height :

```bash
for file in food-101/images/*; do
  mogrify "$file/*.jpg[!320x320>]"
done
```

## Train

Train model :

```bash
PATH=/usr/local/cuda-8.0-cudnn-5.1/bin:$PATH THEANO_FLAGS="device=gpu,floatX=float32" python train.py
```


X_train (50000, 1, 28, 28)
y_train (50000,)
X_val (10000, 1, 28, 28)
y_val (10000,)
X_test (10000, 1, 28, 28)
y_test (10000,)
