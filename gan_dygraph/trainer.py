from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model import *
import paddle.fluid as fluid
step_per_epoch = 2974
lambda_A = 10.0
lambda_B = 10.0
lambda_identity = 0.5


class G_A(fluid.dygraph.Layer):
    """docstring for GATrainer"""
    def __init__(self, name_scope):
        super (G_A, self).__init__(name_scope)
        self.build_gen_discriminator = build_gen_discriminator(self.full_name())
        self.build_generator_resnet_9blocks = build_generator_resnet_9blocks(self.full_name())

    def build_once(self,input_A,input_B):
        print('---------------', input_A.shape)
        print('---------------', input_B.shape)
      

    def forward(self,input_A,input_B):

        fake_B = self.build_generator_resnet_9blocks(input_A)
        fake_A = self.build_generator_resnet_9blocks(input_B)
        cyc_A = self.build_generator_resnet_9blocks(fake_B)
        cyc_B = self.build_generator_resnet_9blocks(fake_A)
        diff_A = fluid.layers.abs(
            fluid.layers.elementwise_sub(
                x=input_A,y=cyc_A))
        diff_B = fluid.layers.abs(
            fluid.layers.elementwise_sub(
                x=input_B, y=cyc_B))
        fake_rec_A = self.build_gen_discriminator(fake_B)
        fake_rec_B = self.build_gen_discriminator(fake_A)
        idt_A = self.build_generator_resnet_9blocks(input_B)
        idt_B = self.build_generator_resnet_9blocks(input_A)
        return fake_A,fake_B,cyc_A,cyc_B,diff_A,diff_B,fake_rec_A,fake_rec_B,idt_A,idt_B

    
class G_B(fluid.dygraph.Layer):
    """docstring for g_B"""
    def __init__(self, name_scope):
        super (G_B, self).__init__(name_scope)
        self.build_gen_discriminator = build_gen_discriminator(self.full_name())
        self.build_generator_resnet_9blocks = build_generator_resnet_9blocks(self.full_name())
    def forward(self,input_A,input_B):
        fake_B = self.build_generator_resnet_9blocks(input_A)
        fake_A = self.build_generator_resnet_9blocks(input_B)
        cyc_A = self.build_generator_resnet_9blocks(fake_B)
        cyc_B = self.build_generator_resnet_9blocks(fake_A)
        diff_A = fluid.layers.abs(
            fluid.layers.elementwise_sub(
                x=input_A,y=cyc_A))
        diff_B = fluid.layers.abs(
            fluid.layers.elementwise_sub(
                x=input_B, y=self.cyc_B))
        cyc_loss = (
                fluid.layers.reduce_mean(diff_A) +
                fluid.layers.reduce_mean(diff_B)) * cycle_loss_factor
        fake_rec_A = self.build_gen_discriminator(fake_A)
        disc_loss_A = fluid.layers.reduce_mean(
            fluid.layers.square(self.fake_rec_A - 1))
        g_loss_B = fluid.layers.elementwise_add(cyc_loss,disc_loss_A)
        return g_loss_B,fake_A
        

class D_A(fluid.dygraph.Layer):
    """docstring for g_B"""
    def __init__(self, name_scope):
        super (D_A, self).__init__(name_scope)
        self.build_gen_discriminator = build_gen_discriminator(self.full_name())
    def forward(self,input_A,fake_pool_A):
        rec_A = self.build_gen_discriminator(input_A)
        fake_pool_rec_A = self.build_gen_discriminator(fake_pool_A)
        d_loss_A = (fluid.layers.square(self.fake_pool_rec_A) +
            fluid.layers.square(rec_A - 1)) / 2.0
        d_loss_A = fluid.layers.reduce_mean(d_loss_A)
        return d_loss_A


class D_B(fluid.dygraph.Layer):
    """docstring for g_B"""
    def __init__(self, name_scope):
        super (D_B, self).__init__(name_scope)
        self.build_gen_discriminator = build_gen_discriminator(self.full_name())
    def forward(self,input_A,input_B):
        rec_B = self.build_gen_discriminator(input_B)
        fake_pool_rec_B = self.build_gen_discriminator(fake_pool_B)
        d_loss_B = (fluid.layers.square(self.fake_pool_rec_B) +
            fluid.layers.square(rec_B - 1)) / 2.0
        d_loss_B = fluid.layers.reduce_mean(d_loss_B)
        return d_loss_B


