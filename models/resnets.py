import sys
sys.setrecursionlimit(10000)
import lasagne
from lasagne.nonlinearities import rectify, softmax, sigmoid
from lasagne.layers import InputLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, helper
from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer, ElemwiseSumLayer, NonlinearityLayer, PadLayer, GlobalPoolLayer, ExpressionLayer
from lasagne.init import Orthogonal, HeNormal, GlorotNormal

he_norm = HeNormal(gain='relu')

def model(shape, n=18, num_filters=16, num_classes=10, width=1, block='normal'):

    if block == "normal":
        from residual_block import residual_block
        n_filters = {0:num_filters, 1:num_filters*width, 2:num_filters*2*width, 3:num_filters*4*width}
    elif block == "dense":
        from residual_block import dense_block as residual_block
        growth_rate=12
        n_filters = {0:num_filters,1:growth_rate,2:growth_rate, 3:growth_rate}
    elif block == "dense_fast":
        from residual_block import dense_fast_block as residual_block
        growth_rate=12
        n_filters = {0:num_filters,1:growth_rate,2:growth_rate, 3:growth_rate}
    else:
        from residual_block import residual_bottleneck_block as residual_block
        n_filters = {0:num_filters, 1:num_filters*4, 2:num_filters*8, 3:num_filters*16}

    l_in = InputLayer(shape=(None, shape[1], shape[2], shape[3]))
    l = NonlinearityLayer(BatchNormLayer(ConvLayer(l_in, num_filters=n_filters[0], filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)),nonlinearity=rectify)

    l = residual_block(l, first=True, filters=n_filters[1])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[1])

    l = residual_block(l, transition=True, filters=n_filters[2])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[2])

    l = residual_block(l, transition=True, filters=n_filters[3])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[3])

    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)
    avg_pool = GlobalPoolLayer(bn_post_relu)
    return DenseLayer(avg_pool, num_units=num_classes, W=HeNormal(), nonlinearity=softmax)   #lasagne.init.HeNormal(gain=1)
