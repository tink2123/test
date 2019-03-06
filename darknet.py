#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.regularizer import L2Decay
from config import cfg

def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='leaky',
                  name='conv',
                  i=0):
    conv1 = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02),
                name=name + str(i)+"_weights"),
        bias_attr=False,
        name=name + '.conv2d.output.1')
    if name == "conv":
        bn_name = "bn" + str(i)
    else:
        bn_name = "bn" + name[4:]
    print(bn_name)

    out = fluid.layers.batch_norm(
        input=conv1,
        name=bn_name + '.output.1',
        param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02),
                regularizer=L2Decay(0.),
                name=bn_name + '_scale'),
        bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=L2Decay(0.),
                name=bn_name + '_offset'),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_var',
        is_test=False)

    if act == 'leakey':
        out = fluid.layers.leakey_relu(x=out, alpha=0.1)
    return out

def conv_affine_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      name=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=False,
        name=name + '.conv2d.output.1')
    if name == "conv1":
        bn_name = "bn_" + name
    else:
        bn_name = "bn" + name[3:]

    scale = fluid.layers.create_parameter(
        shape=[conv.shape[1]],
        dtype=conv.dtype,
        attr=ParamAttr(
            name=bn_name + '_scale', learning_rate=0.),
        default_initializer=Constant(1.))
    scale.stop_gradient = True
    bias = fluid.layers.create_parameter(
        shape=[conv.shape[1]],
        dtype=conv.dtype,
        attr=ParamAttr(
            bn_name + '_offset', learning_rate=0.),
        default_initializer=Constant(0.))
    bias.stop_gradient = True

    out = fluid.layers.affine_channel(x=conv, scale=scale, bias=bias)
    if act == 'relu':
        out = fluid.layers.relu(x=out)
        return out

def shortcut(input, ch_out, stride, name):
    ch_in = input.shape[1]  # if args.data_format == 'NCHW' else input.shape[-1]
    if ch_in != ch_out:
        return conv_affine_layer(input, ch_out, 1, stride, 0, act=None, name=name)
    else:
        return input

def basicblock(input, ch_out, stride, name,i):
    """
    channel: convolution channels for 1x1 conv
    """
    short = shortcut(input, ch_out*2, stride, name=name)
    conv1 = conv_bn_layer(input, ch_out, 1, 1, 0, name="conv", i=i)
    conv2 = conv_bn_layer(conv1, ch_out*2, 3, 1, 1, name="conv", i=i+1)
    out = fluid.layers.elementwise_add(x=short, y=conv2, act=None, name=name+"short")
    return out

def layer_warp(block_func, input, ch_out, count, stride, name,i):
    res_out = block_func(input, ch_out, stride, name="conv", i=i)
    for j in range(1, count):
        res_out = block_func(res_out, ch_out, 1, name="conv" ,i=i+j*2)
    return res_out

DarkNet_cfg = {
        53: ([1,2,8,8,4],basicblock)
}

# num_filters = [32, 64, 128, 256, 512, 1024]

def add_DarkNet53_conv_body(body_input,layers):

    stages, block_func = DarkNet_cfg[53]
    stages = stages[0:5]
    conv1 = conv_bn_layer(
            body_input, ch_out=32, filter_size=3, stride=1, padding=1, act="leaky", name="conv",i=0)
    conv2 = conv_bn_layer(
            conv1, ch_out=64, filter_size=3, stride=2, padding=1, act="leaky", name="conv",i=1)
    block3 = layer_warp(block_func, conv2, 32, stages[0], 1, name="conv",i=2)
    downsample3 = conv_bn_layer(
            block3, ch_out=128, filter_size=3, stride=2, padding=1,name="conv", i=5)
    """
    do we use freeze_at ?
    """
    block4 = layer_warp(block_func, downsample3, 64, stages[1], 1, name="conv",i=6)
    downsample4 = conv_bn_layer(
            block4, ch_out=256, filter_size=3, stride=2, padding=1, name="conv",i=12)
    block5 = layer_warp(block_func, downsample4, 128, stages[2], 1, name="conv", i=13)
    if layers == "scale3":
        return block5
    downsample5 = conv_bn_layer(
            block5, ch_out=512, filter_size=3, stride=2, padding=1, name="conv",i=37)
    block6 = layer_warp(block_func, downsample5, 256, stages[3], 1, name="conv", i=38)
    if layers == "scale2":
        return block6
    downsample6 = conv_bn_layer(
            block6, ch_out=1024, filter_size=3, stride=2, padding=1, name="conv", i=62)
    block7 = layer_warp(block_func, downsample6, 512, stages[4], 1, name="conv",i=63)
    if layers == "scale1":
        return block7

