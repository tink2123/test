from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model import *
import paddle.fluid as fluid

step_per_epoch = 2974
lambda_A = 10.0
lambda_B = 10.0
lambda_identity = 0.5


class Cycle_Gan(fluid.dygraph.Layer):
    """docstring for GATrainer"""
    def __init__(self, name_scope,istrain=True):
        super (Cycle_Gan, self).__init__(name_scope)

        self.build_generator_resnet_9blocks_a = build_generator_resnet_9blocks(self.full_name())
        self.build_generator_resnet_9blocks_b = build_generator_resnet_9blocks(self.full_name())
        if istrain:
            self.build_gen_discriminator_a = build_gen_discriminator(self.full_name())
            self.build_gen_discriminator_b = build_gen_discriminator(self.full_name())
        #self.is_G = is_G
        #self.is_DA = is_DA
        #self.is_DB = is_DB

    ###def build_once(self,input_A,input_B):
    ###    print('---------------', input_A.shape)
    ###    print('---------------', input_B.shape)
    ###  

    def forward(self,input_A,input_B,is_G,is_DA,is_DB):

        if is_G:
            fake_B = self.build_generator_resnet_9blocks_a(input_A)
            fake_A = self.build_generator_resnet_9blocks_b(input_B)
            cyc_A = self.build_generator_resnet_9blocks_b(fake_B)
            cyc_B = self.build_generator_resnet_9blocks_a(fake_A)

            diff_A = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_A,y=cyc_A))
            diff_B = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_B, y=cyc_B))
            #print("fake_B:",fake_B.numpy()[0][0][0][:10])
            fake_rec_A = self.build_gen_discriminator_a(fake_B)
            #print("fake_Rec_A:",fake_rec_A.numpy()[0][0][0][:10])            
            fake_rec_B = self.build_gen_discriminator_b(fake_A)
            
            idt_A = self.build_generator_resnet_9blocks_a(input_B)

            idt_B = self.build_generator_resnet_9blocks_b(input_A)
            return fake_A,fake_B,cyc_A,cyc_B,diff_A,diff_B,fake_rec_A,fake_rec_B,idt_A,idt_B

        if is_DA:

            ### D
            rec_B = self.build_gen_discriminator_a(input_A)
            fake_pool_rec_B = self.build_gen_discriminator_a(input_B)
            #print("dy:fake_pool_rec_B=",fake_pool_rec_B.numpy()[0][0][0][:10])
            return rec_B, fake_pool_rec_B

        if is_DB:

            rec_A = self.build_gen_discriminator_b(input_A)

            fake_pool_rec_A = self.build_gen_discriminator_b(input_B)


        return rec_A, fake_pool_rec_A

