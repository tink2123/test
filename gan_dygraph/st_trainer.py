from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model import *
import paddle.fluid as fluid

step_per_epoch = 2974
lambda_A = 10.0
lambda_B = 10.0
lambda_identity = 0.5


class GTrainer():
    def __init__(self, input_A, input_B):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            self.fake_B = build_generator_resnet_9blocks(input_A, name="g_A")
            #FIXME set persistable explicitly to pass CE
            self.fake_B.persistable = True
            self.fake_A = build_generator_resnet_9blocks(input_B, name="g_B")
            self.fake_A.persistable = True
            self.cyc_A = build_generator_resnet_9blocks(self.fake_B, "g_B")
            self.cyc_B = build_generator_resnet_9blocks(self.fake_A, "g_A")
            self.infer_program = self.program.clone()
            print('---------------', input_A.shape)
            print('---------------', input_B.shape)
            print('---------------', self.fake_B.shape)
            print('---------------', self.fake_A.shape)
            print('---------------', self.cyc_A.shape)
            print('---------------', self.cyc_B.shape)
            # Cycle Loss
            diff_A = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_A, y=self.cyc_A))
            diff_B = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_B, y=self.cyc_B))
            self.cyc_A_loss = fluid.layers.reduce_mean(diff_A) * lambda_A
            self.cyc_B_loss = fluid.layers.reduce_mean(diff_B) * lambda_B
            self.cyc_loss = self.cyc_A_loss + self.cyc_B_loss
            # GAN Loss D_A(G_A(A))
            self.fake_rec_A = build_gen_discriminator(self.fake_B, "d_A")
            self.G_A = fluid.layers.reduce_mean(
                fluid.layers.square(self.fake_rec_A - 1))
            # GAN Loss D_B(G_B(B))
            self.fake_rec_B = build_gen_discriminator(self.fake_A, "d_B")
            self.G_B = fluid.layers.reduce_mean(
                fluid.layers.square(self.fake_rec_B - 1))
            self.G = self.G_A + self.G_B
            # Identity Loss G_A
            self.idt_A = build_generator_resnet_9blocks(input_B, name="g_A")
            self.idt_loss_A = fluid.layers.reduce_mean(fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x = input_B, y = self.idt_A))) * lambda_B * lambda_identity
            # Identity Loss G_B
            self.idt_B = build_generator_resnet_9blocks(input_A, name="g_B")
            self.idt_loss_B = fluid.layers.reduce_mean(fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x = input_A, y = self.idt_B))) * lambda_A * lambda_identity

            self.idt_loss = fluid.layers.elementwise_add(self.idt_loss_A, self.idt_loss_B)

###            self.g_loss_A = fluid.layers.elementwise_add(self.cyc_loss,
###                                                         self.disc_loss_B)
            self.g_loss = self.cyc_loss + self.G + self.idt_loss

            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and (var.name.startswith("g_A") or var.name.startswith("g_B")):
                    vars.append(var.name)
            self.param = vars
            lr = 0.0002
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        12 * step_per_epoch, 32 * step_per_epoch,
                        52 * step_per_epoch, 72 * step_per_epoch
                    ],
                    values=[
                        lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5,
                name="gen")
###            for param in self.program.global_block().all_parameters():
###                 print(param.name)
            optimizer.minimize(self.g_loss, parameter_list=vars)

class DATrainer():
    def __init__(self, input_B, fake_pool_B):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            self.rec_B = build_gen_discriminator(input_B, "d_A")
            self.fake_pool_rec_B = build_gen_discriminator(fake_pool_B, "d_A")
            self.d_loss_A = (fluid.layers.square(self.fake_pool_rec_B) +
                             fluid.layers.square(self.rec_B - 1)) / 2.0
            self.d_loss_A = fluid.layers.reduce_mean(self.d_loss_A)

            optimizer = fluid.optimizer.Adam(learning_rate=0.0002, beta1=0.5)
            optimizer._name = "d_A"
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("d_A"):
                    vars.append(var.name)

            self.param = vars
            lr = 0.0002
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        12 * step_per_epoch, 32 * step_per_epoch,
                        52 * step_per_epoch, 72 * step_per_epoch
                    ],
                    values=[
                        lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5,
                name="d_A")

###            for param in self.program.global_block().all_parameters():
###                 print(param.name)
###
            optimizer.minimize(self.d_loss_A, parameter_list=vars)


class DBTrainer():
    def __init__(self, input_A, fake_pool_A):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            self.rec_A = build_gen_discriminator(input_A, "d_B")
            self.fake_pool_rec_A = build_gen_discriminator(fake_pool_A, "d_B")
            self.d_loss_B = (fluid.layers.square(self.fake_pool_rec_A) +
                             fluid.layers.square(self.rec_A - 1)) / 2.0
            self.d_loss_B = fluid.layers.reduce_mean(self.d_loss_B)
            optimizer = fluid.optimizer.Adam(learning_rate=0.0002, beta1=0.5)
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("d_B"):
                    vars.append(var.name)
            self.param = vars
            lr = 0.0002
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        12 * step_per_epoch, 32 * step_per_epoch,
                        52 * step_per_epoch, 72 * step_per_epoch
                    ],
                    values=[
                        lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5,
                name="d_B")
###            for param in self.program.global_block().all_parameters():
###                 print(param.name)
            optimizer.minimize(self.d_loss_B, parameter_list=vars)
