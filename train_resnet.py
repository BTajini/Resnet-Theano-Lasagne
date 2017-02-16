#!/usr/bin/env python
from __future__ import print_function
import sys, os, time, string
import numpy as np
np.random.seed(1234)
import theano
import theano.tensor as T
import lasagne
from utils import *
sys.setrecursionlimit(10000)

print("Config mode " + theano.config.mode)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=500)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--n", type=int, default=18, help="The total of residual blocks will be 3 times this value n.")
parser.add_argument("--width", type=int, default=1, help="For normal block only. Converting Resnet to Wide ResNet.")
parser.add_argument("--num_filters", type=int, default=16, help="Number of filter units for first convolution")
parser.add_argument('--block', default="normal", choices=["normal","bottleneck","dense", "dense_fast"])
parser.add_argument('--dataset', default="mnist", choices=['mnist', 'cifar10', 'food101'])
args = parser.parse_args()

num_classes = 10
if args.dataset == "mnist" :
    from dataloader.mnist import Dataloader
elif args.dataset == "cifar10":
    from dataloader.cifar10 import Dataloader
else:
    from models.resnet101 import model
    num_classes = 101
train_data_loader = Dataloader(args.batch_size, "train")
test_data_loader = Dataloader(args.batch_size, "test")

print("Data shape:")
print(train_data_loader.shape())

print("ResNet with {} residual blocks.".format(args.n))
if (args.dataset == "mnist") or (args.dataset == "cifar10"):
    print("Bottleneck block. Depth {}".format(9*args.n+2))
elif (args.dataset == "mnist") or (args.dataset == "cifar10"):
    print("Width factor {}, no bottleneck. Depth {}.".format(args.width,6*args.n+2))
else:
    print("Dense block. Depth {}.".format(3*args.n+1))
from models.resnets import model
network = model(train_data_loader.shape(), n=args.n, num_classes=num_classes, num_filters=args.num_filters, width=args.width, block=args.block)
describe(network)

print("Compiling")
input_var = T.tensor4()
target_var = T.ivector()
prediction = lasagne.layers.get_output(network, input_var)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)

learning_rate_schedule = {
0: 0.0001, # low initial learning rate as described in paper
2: 0.01,
100: 0.001,
150: 0.0001
}

learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=learning_rate, momentum=0.9)

test_prediction = lasagne.layers.get_output(network, input_var, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

print("Starting training...")
test_acc, test_err = [], []
for epoch in range(args.num_epochs):
    if epoch in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[epoch])
        print(" setting learning rate to %.7f" % lr)
        learning_rate.set_value(lr)

    train_err = 0
    train_batches = 0
    start_time = time.time()
    for inputs, targets in progress(train_data_loader.next_minibatch(),
            desc='Epoch %d/%d, Batch ' % (epoch + 1, args.num_epochs),
            total=train_data_loader.nb_batches):
        train_err += train_fn(inputs, targets)
        train_batches += 1

    val_err = 0
    val_acc = 0
    val_batches = 0
    for inputs, targets in test_data_loader.next_minibatch():
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    test_acc += [val_acc / val_batches]
    test_err += [val_err / val_batches]

    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, args.num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))


np.savez("res_test_3",test_acc=test_acc,test_err=test_err)
np.savez("weights", *lasagne.layers.get_all_param_values(network))

test_err = 0
test_acc = 0
test_batches = 0
for inputs, targets in test_data_loader.next_minibatch():
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))
