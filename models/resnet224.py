import sys
sys.setrecursionlimit(10000)
import lasagne
from lasagne.nonlinearities import rectify, softmax, sigmoid
from lasagne.layers import InputLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, helper
from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer, ElemwiseSumLayer, NonlinearityLayer, PadLayer, GlobalPoolLayer, ExpressionLayer, MaxPool2DLayer
from lasagne.init import Orthogonal, HeNormal, GlorotNormal
from residual_block import residual_bottleneck_block as residual_block

he_norm = HeNormal(gain='relu')

def model(num_classes=101):
    l_in = InputLayer(shape=(None, 3, 224, 224))
    l = NonlinearityLayer(BatchNormLayer(ConvLayer(l_in, num_filters=64, filter_size=(7,7), stride=(2,2), nonlinearity=None, pad='same', W=he_norm)),nonlinearity=rectify)
    l = MaxPool2DLayer(l, 3, stride=2, pad='same')

    l = residual_block(l, filters=256,first = True)
    for _ in range(1,3):
        l = residual_block(l, filters=256)

    l = residual_block(l, filters=512,transition = True)
    for _ in range(1,4):
        l = residual_block(l, filters=512)

    l = residual_block(l, filters=1024,transition = True)
    for _ in range(1,23):
        l = residual_block(l, filters=1024)

    l = residual_block(l, filters=2048,transition = True)
    for _ in range(1,3):
        l = residual_block(l, filters=2048)





    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)
    avg_pool = GlobalPoolLayer(bn_post_relu)
    return DenseLayer(avg_pool, num_units=num_classes, W=HeNormal(), nonlinearity=softmax)   #lasagne.init.HeNormal(gain=1)



