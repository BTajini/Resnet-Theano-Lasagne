from lasagne.nonlinearities import rectify
from lasagne.layers import NonlinearityLayer, ElemwiseSumLayer, Pool2DLayer, ConcatLayer, BiasLayer, ScaleLayer
from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.init import HeNormal

he_norm = HeNormal(gain='relu')

def residual_block(l, transition=False, first=False, filters=16):
    if transition:
        first_stride = (2,2)
    else:
        first_stride = (1,1)

    if first:
        bn_pre_relu = l
    else:
        bn_pre_conv = BatchNormLayer(l)
        bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

    conv_1 = NonlinearityLayer(BatchNormLayer(ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(3,3), stride=first_stride, nonlinearity=None, pad='same', W=he_norm)),nonlinearity=rectify)

    #dropout = DropoutLayer(conv_1, p=0.3)
    conv_2 = ConvLayer(conv_1, num_filters=filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)

    # add shortcut connections
    if transition:
        # projection shortcut, as option B in paper
        projection = ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
    elif conv_2.output_shape == l.output_shape:
        projection=l
    else:
        projection = ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None)

    return ElemwiseSumLayer([conv_2, projection])


def residual_bottleneck_block(l, transition=False, first=False, filters=16):
    if transition:
        first_stride = (2,2)
    else:
        first_stride = (1,1)

    if first:
        bn_pre_relu = l
    else:
        bn_pre_conv = BatchNormLayer(l)
        bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

    bottleneck_filters = filters / 4

    conv_1 = NonlinearityLayer(BatchNormLayer(ConvLayer(bn_pre_relu, num_filters=bottleneck_filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)),nonlinearity=rectify)

    conv_2 = NonlinearityLayer(BatchNormLayer(ConvLayer(conv_1, num_filters=bottleneck_filters, filter_size=(3,3), stride=first_stride, nonlinearity=None, pad='same', W=he_norm)),nonlinearity=rectify)

    conv_3 = ConvLayer(conv_2, num_filters=filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)

    if transition:
        projection = ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
    elif first:
        projection = ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None)
    else:
        projection = l

    return ElemwiseSumLayer([conv_3, projection])


def dense_block(network, transition=False, first=False, filters=16):
    if transition:
        network = NonlinearityLayer(BatchNormLayer(network), nonlinearity=rectify)
        network = ConvLayer(network,network.output_shape[1], 1, pad='same', W=he_norm,b=None, nonlinearity=None)
        network = Pool2DLayer(network, 2, mode='average_inc_pad')

    network = NonlinearityLayer(BatchNormLayer(network), nonlinearity=rectify)
    conv = ConvLayer(network,filters, 3, pad='same', W=he_norm,b=None, nonlinearity=None)
    return ConcatLayer([network, conv], axis=1)


def dense_fast_block(network, transition=False, first=False, filters=16):
    if transition:
        network = NonlinearityLayer(BiasLayer(ScaleLayer(network)), nonlinearity=rectify)
        network = ConvLayer(network,network.output_shape[1], 1, pad='same', W=he_norm,b=None, nonlinearity=None)
        network = BatchNormLayer(Pool2DLayer(network, 2, mode='average_inc_pad'))

    network = NonlinearityLayer(BiasLayer(ScaleLayer(network)), nonlinearity=rectify)
    conv = ConvLayer(network,filters, 3, pad='same', W=he_norm,b=None, nonlinearity=None)
    return ConcatLayer([network, BatchNormLayer(conv)], axis=1)
