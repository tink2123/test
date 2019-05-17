from __future__ import division
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D,  Conv2DTranspose , BatchNorm ,Pool2D
import os

# cudnn is not better when batch size is 1.
use_cudnn = False
if 'ce_mode' in os.environ:
    use_cudnn = False


class conv2d(fluid.dygraph.Layer):
    """docstring for Conv2D"""
    def __init__(self, 
                name_scope,
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=1,
                stddev=0.02,
                padding=0,
                norm=True,
                relu=True,
                relufactor=0.0,
                use_bias=False):
        super(conv2d, self).__init__(name_scope)

        self.conv_bias = Conv2D(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            use_cudnn=use_cudnn,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(0.0,0.02)),
            bias_attr=None)

        self.conv = Conv2D(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            use_cudnn=use_cudnn,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(0.0,0.02)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0)))

        self.bn = BatchNorm(self.full_name(),
            num_channels=num_filters,
            param_attr=fluid.ParamAttr(
                name="what's_weight",
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name="what's",
                initializer=fluid.initializer.Constant(0.0)),
            )

        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.relu = relu

    
    def forward(self,inputs):
        if self.use_bias == False:
            conv = self.conv(inputs)
        else:
            conv = self.conv_bias(inputs)
##        if self.norm:
##            conv = self.bn(conv)
        if self.relu:
            conv = fluid.layers.leaky_relu(conv,alpha=self.relufactor)
       
        return conv


class DeConv2D(fluid.dygraph.Layer):
    def __init__(self,
            name_scope,
            num_filters=64,
            filter_size=7,
            stride=1,
            stddev=0.02,
            padding=[0,0],
            outpadding=[0,0,0,0],
            relu=True,
            norm=True,
            relufactor=0.0,
            use_bias=False
            ):
        super(DeConv2D,self).__init__(name_scope)

        self._deconv = Conv2DTranspose(self.full_name(),
                                        num_filters,
                                        filter_size=filter_size,
                                        stride=stride,
                                        padding=padding,
                                        param_attr=fluid.ParamAttr(
                                            initializer=fluid.initializer.TruncatedNormal(scale=stddev)),
                                        bias_attr=fluid.ParamAttr(
                			    initializer=fluid.initializer.Constant(0.0)))

        self.bn = BatchNorm(self.full_name(),
            num_channels=num_filters,
            param_attr=fluid.ParamAttr(
                name="de_wights",
                initializer=fluid.initializer.TruncatedNormal(1.0, 0.02)),
            bias_attr=fluid.ParamAttr(
                name="de_bias",
                initializer=fluid.initializer.Constant(0.0)),
            )        
        self.outpadding = outpadding
        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.relu = relu

    def forward(self,inputs):
        #todo: add use_bias
        if self.use_bias==False:
            conv = self._deconv(inputs)
        else:
            conv = self._deconv(inputs)
        conv = fluid.layers.pad2d(conv, paddings=self.outpadding, mode='Constant', pad_value=0.0)

        if self.norm:
            conv = self.bn(conv)
        if self.relu:
            conv = fluid.layers.leaky_relu(conv,alpha=self.relufactor)
        return conv

