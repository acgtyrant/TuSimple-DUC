#!/usr/bin/env python
import mxnet as mx


def residual_unit(
        data, num_filter, stride, dim_match, dilate, name,
        bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same,
        otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    bn1 = mx.sym.BatchNorm(
            data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom,
            name=name + '_bn1')
    act1 = mx.sym.Activation(
            data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(
            data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1),
            stride=(1, 1), pad=(0, 0), no_bias=True, workspace=workspace,
            name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(
            data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom,
            name=name + '_bn2')
    act2 = mx.sym.Activation(
            data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(
            data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3),
            stride=stride, pad=dilate, dilate=dilate, no_bias=True,
            workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(
            data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom,
            name=name + '_bn3')
    act3 = mx.sym.Activation(
            data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(
            data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1),
            pad=(0, 0), no_bias=True, workspace=workspace,
            name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(
                data=act1, num_filter=num_filter, kernel=(1, 1),
                stride=stride, no_bias=True, workspace=workspace,
                name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut


def resnet(units, filter_list, bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    filter_list : list
        Channel size of each stage
    workspace : int
        Workspace used in convolution operator
    """
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(
            data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom,
            name='bn_data')
    body = mx.sym.Convolution(
            data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2),
            pad=(3, 3), no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(
            data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(
            data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
            pool_type='max')
    for i, unit in enumerate(units):
        for j in range(unit):
            if j == 0:
                stride = (1 if i != 1 else 2, 1 if i != 1 else 2)
                dim_match = False
            else:
                stride = (1, 1)
                dim_match = True
            if i in (0, 1):
                dilate = (1, 1)
            elif i == 2:
                dilate = (1, 2, 5, 9)[j % 4]
                dilate = (dilate, dilate)
            elif i == 3:
                dilate = (5, 9, 17)[j]
                dilate = (dilate, dilate)
            body = residual_unit(
                    body,
                    filter_list[i + 1],
                    stride,
                    dim_match,
                    dilate,
                    name='stage%d_unit%d' % (i + 1, j + 1),
                    workspace=workspace,
                    memonger=memonger)
    return body


def main():
    from pathlib import Path
    if not Path('resnet-152-symbol.json').exists():
        path = 'http://data.mxnet.io/models/imagenet/resnet/101-layers/'
        mx.test_utils.download(path + 'resnet-101-0000.params')
        mx.test_utils.download(path + 'resnet-101-symbol.json')
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-101', 0)
    print(arg_params.keys())
    sym = resnet(units=(3, 4, 23, 3), filter_list=(64, 256, 512, 1024, 2048))
    module = mx.mod.Module(sym, label_names=None)
    module.bind([('data', (1, 3, 800, 800))])
    module.set_params(arg_params, aux_params)


if __name__ == '__main__':
    main()
