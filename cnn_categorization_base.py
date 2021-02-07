from torch import nn


def cnn_categorization_base(netspec_opts):
    """
    Constructs a network for the base categorization model.

    Arguments
    --------
    netspec_opts: (dictionary), the network's architecture. It has the keys
                 'kernel_size', 'num_filters', 'stride', and 'layer_type'.
                 Each key holds a list containing the values for the
                corresponding parameter for each layer.
    Returns
    ------
     net: (nn.Sequential), the base categorization model
    """
    # instantiate an instance of nn.Sequential
    net = nn.Sequential()

    # add layers as specified in netspec_opts to the network
    kernal_size = netspec_opts['kernel_size']
    num_filters = netspec_opts['num_filters']
    stride = netspec_opts['stride']
    layer_type = netspec_opts['layer_type']
    previous = 0
    for index, (size_kernal,filters_num,strid,type_layer) in enumerate(zip(kernal_size,num_filters, stride,layer_type)):
        if type_layer == "conv":
            padding = (size_kernal-1)/2
            if index == 0:
                previous = filters_num
                net.add_module("conv" + str(index),nn.Conv2d(3, filters_num, size_kernal, strid,int(padding)))
            else:
                net.add_module("conv"+str(index),nn.Conv2d(previous,filters_num,size_kernal,strid,int(padding)))
                previous = filters_num
        if type_layer == "bn":
            net.add_module("bn"+str(index),nn.BatchNorm2d(num_filters[index-1]))
        if type_layer == "relu":
            net.add_module("relu"+str(index),nn.ReLU())
        if type_layer == "pool":
            net.add_module("pool"+str(index),nn.AvgPool2d(size_kernal,strid,0))

    return net
