import unittest
import numpy as np
import six
import sys

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid import Conv2D, Pool2D, FC
from test_imperative_base import new_program_scope
from paddle.fluid.dygraph.base import to_variable
import data_reader
from utility import add_arguments, print_arguments, ImagePool
from trainer import *
from st_model import *


lambda_A = 10.0
lambda_B = 10.0
lambda_identity = 0.5
step_per_epoch = 2974


class GTrainer():
    def __init__(self, input_A, input_B):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            self.fake_B = st_build_generator_resnet_9blocks(input_A, name="g_A")
            #FIXME set persistable explicitly to pass CE
            self.fake_B.persistable = True
            self.fake_A = st_build_generator_resnet_9blocks(input_B, name="g_B")
            self.fake_A.persistable = True
            # build_generatro_res_9block is wrong
            #fluid.layers.Print(input="g_A_c0_w@GRAD",print_tensor_name=True,summarize=10)
            #fluid.layers.Print(input=self.fake_A,print_tensor_name=True,summarize=10)
            self.cyc_A = st_build_generator_resnet_9blocks(self.fake_B, "g_B")
            self.cyc_B = st_build_generator_resnet_9blocks(self.fake_A, "g_A")
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
            self.fake_rec_A = st_build_gen_discriminator(self.fake_B, "d_A")
            #print("__________________________ fake_rec_A_________________________")
            #fluid.layers.Print(input=self.fake_A,print_tensor_name=True,summarize=10)
            g_A_tmp = fluid.layers.square(self.fake_rec_A - 1)
            self.G_A = fluid.layers.reduce_mean(
                fluid.layers.square(self.fake_rec_A - 1))
            #fluid.layers.Print(input=g_A_tmp,print_tensor_name=True,summarize=10)
            # GAN Loss D_B(G_B(B))
            self.fake_rec_B = st_build_gen_discriminator(self.fake_A, "d_B")
            self.G_B = fluid.layers.reduce_mean(
                fluid.layers.square(self.fake_rec_B - 1))
            self.G = self.G_A + self.G_B
            # Identity Loss G_A
            self.idt_A = st_build_generator_resnet_9blocks(input_B, name="g_A")
#            print("~~~~~~~~~~~~~~~~~~~`idt_A~~~~~~~~~~~~~")
#            fluid.layers.Print(input=self.idt_A,print_tensor_name=True)
            self.idt_loss_A = fluid.layers.reduce_mean(fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x = input_B, y = self.idt_A))) * lambda_B * lambda_identity
            # Identity Loss G_B
            self.idt_B = st_build_generator_resnet_9blocks(input_A, name="g_B")
            self.idt_loss_B = fluid.layers.reduce_mean(fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x = input_A, y = self.idt_B))) * lambda_A * lambda_identity
            #print("idt_B")
            #fluid.layers.Print(self.idt_B,print_tensor_name=True,summarize=10)
            #print("idt_A")
            #fluid.layers.Print(self.idt_A,print_tensor_name=True,summarize=10)
            self.idt_loss = fluid.layers.elementwise_add(self.idt_loss_A, self.idt_loss_B)

###            self.g_loss_A = fluid.layers.elementwise_add(self.cyc_loss,
###                                                         self.disc_loss_B)
            self.g_loss = self.cyc_loss + self.G + self.idt_loss
            #print("__________________________ g_loss _________________________")
            #fluid.layers.Print(input=self.g_loss,print_tensor_name=True)
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and (var.name.startswith("g_A") or var.name.startswith("g_B")):
                    vars.append(var.name)
            self.param = vars
            lr = 0.0002
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        100 * step_per_epoch, 120 * step_per_epoch,
                        140 * step_per_epoch, 160 * step_per_epoch,
                        180 * step_per_epoch
                    ],
                    values=[
                        lr , lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5)
###            for param in self.program.global_block().all_parameters():
###                 print(param.name)
            optimizer.minimize(self.g_loss, parameter_list=vars)

class DATrainer():
    def __init__(self, input_B, fake_pool_B):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):

            self.rec_B = st_build_gen_discriminator(input_B, "d_A")
            #print("~~~~~~~~~~~~~~~~~~~~~~~~~ fake_pool_rec_B~~~~~~~~~~~~~~~`")
            #fluid.layers.Print(input=self.rec_B,print_tensor_name=True,summarize=10)
            self.fake_pool_rec_B = st_build_gen_discriminator(fake_pool_B, "d_A")
            #fluid.layers.Print(self.fake_pool_rec_B,print_tensor_name=True,summarize=10)
            self.d_loss_A_tmp = fluid.layers.square(self.fake_pool_rec_B)
            self.d_loss_A = (fluid.layers.square(self.fake_pool_rec_B) +
                             fluid.layers.square(self.rec_B - 1)) / 2.0
            #print("~~~~~~~~~~~~~~~~~~~~~~~d_loss_A_tmp~~~~~~~~~~~~~~~~~~~~~~~")
            #fluid.layers.Print(self.d_loss_A_tmp,print_tensor_name=True)
            self.d_loss_A = fluid.layers.reduce_mean(self.d_loss_A)

            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("d_A"):
                    vars.append(var.name)
                    #fluid.layers.Print(var.name,summarize=10)

            self.param = vars
            lr = 0.0002
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        100 * step_per_epoch, 120 * step_per_epoch,
                        140 * step_per_epoch, 160 * step_per_epoch,
                        180 * step_per_epoch
                    ],
                    values=[
                        lr , lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5)
###            for param in self.program.global_block().all_parameters():
###                if param.name=="d_A_c3_w": 
###                    np.array(fluid.scope.find_var(param.name).get_tensor())
###
            optimizer.minimize(self.d_loss_A, parameter_list=vars)


class DBTrainer():
    def __init__(self, input_A, fake_pool_A):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            self.rec_A = st_build_gen_discriminator(input_A, "d_B")
            self.fake_pool_rec_A = st_build_gen_discriminator(fake_pool_A, "d_B")
            self.d_loss_B = (fluid.layers.square(self.fake_pool_rec_A) +
                             fluid.layers.square(self.rec_A - 1)) / 2.0
            self.d_loss_B = fluid.layers.reduce_mean(self.d_loss_B)
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("d_B"):
                    vars.append(var.name)
            self.param = vars
            lr = 0.0002
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        100 * step_per_epoch, 120 * step_per_epoch,
                        140 * step_per_epoch, 160 * step_per_epoch,
                        180 * step_per_epoch
                    ],
                    values=[
                        lr , lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5)
###            for param in self.program.global_block().all_parameters():
###                 print(param.name)
            optimizer.minimize(self.d_loss_B, parameter_list=vars)


class TestDygraphGAN(unittest.TestCase):
    def test_gan_float32(self):
        seed = 90

        startup = fluid.Program()
        startup.random_seed = seed
        discriminate_p = fluid.Program()
        generate_p = fluid.Program()
        discriminate_p.random_seed = seed
        generate_p.random_seed = seed
        scope = fluid.core.Scope()
        with new_program_scope(
                main=discriminate_p, startup=startup, scope=scope):

            input_A = fluid.layers.data(
                name="input_A", shape=[1,3, 256,256], append_batch_size=False)
            input_B = fluid.layers.data(
                name="input_B", shape=[1,3,256,256], append_batch_size=False)
            fake_pool_A = fluid.layers.data(
                name="fake_pool_A", shape=[1,3,256,256], append_batch_size=False)
            fake_pool_B = fluid.layers.data(
                name="fake_pool_B", shape=[1,3,256,256], append_batch_size=False)

            gen_trainer = GTrainer(input_A, input_B)
            d_A_trainer = DATrainer(input_B, fake_pool_B)
            d_B_trainer = DBTrainer(input_A, fake_pool_A)            

            
        exe = fluid.Executor(fluid.CPUPlace() if not core.is_compiled_with_cuda(
        ) else fluid.CUDAPlace(0))
        gen_trainer_program = fluid.CompiledProgram(
            gen_trainer.program).with_data_parallel(
                loss_name=gen_trainer.g_loss.name)
    ###    g_B_trainer_program = fluid.CompiledProgram(
    ###        g_B_trainer.program).with_data_parallel(
    ###            loss_name=g_B_trainer.g_loss_B.name)
        d_A_trainer_program = fluid.CompiledProgram(
            d_A_trainer.program).with_data_parallel(
                loss_name=d_A_trainer.d_loss_A.name)
        d_B_trainer_program = fluid.CompiledProgram(
            d_B_trainer.program).with_data_parallel(
                loss_name=d_B_trainer.d_loss_B.name)
        static_params = dict()
        with fluid.scope_guard(scope):
            self.data_A= np.random.random(size=[1,3,256,256]).astype("float32")
            self.data_B= np.random.random(size=[1,3,256,256]).astype("float32")
            #self.data_A=np.ones([1,3,256,256], np.float32)
            #self.data_B=np.ones([1,3,256,256], np.float32)
            A_pool = ImagePool()
            B_pool = ImagePool()

            #self.data_pool_B = B_pool.pool_image(fake_B_tmp)
            #self.data_pool_A = A_pool.pool_image(fake_A_tmp)
            self.data_pool_B= np.random.random(size=[1,3,256,256]).astype("float32")
            self.data_pool_A= np.random.random(size=[1,3,256,256]).astype("float32")
            exe.run(startup)
            for i in range(10):
                g_A_loss, g_A_cyc_loss, g_A_idt_loss, g_B_loss, g_B_cyc_loss, g_B_idt_loss, fake_A_tmp, fake_B_tmp = exe.run(
                    gen_trainer_program,
                    fetch_list=[
                        gen_trainer.G_A, gen_trainer.cyc_A_loss, gen_trainer.idt_loss_A, gen_trainer.G_B, gen_trainer.cyc_B_loss,
                        gen_trainer.idt_loss_B, gen_trainer.fake_A, gen_trainer.fake_B
                    ],
                    feed={"input_A": self.data_A,
                          "input_B": self.data_B})
                sta_g_loss = g_A_loss + g_B_loss + g_A_cyc_loss + g_B_cyc_loss + g_A_idt_loss + g_B_idt_loss
                print("sta:g_A_loss:{},g_B_loss:{},g_A_cyc_loss:{},g_B_cyc_loss:{},g_A_idt_loss:{},g_B_idt_loss:{}".format(g_A_loss, g_B_loss,g_A_cyc_loss, g_B_cyc_loss,g_A_idt_loss,g_B_idt_loss))
                
                #A_pool = ImagePool()
                #B_pool = ImagePool()
    
                #self.data_pool_B = B_pool.pool_image(fake_B_tmp)
                #self.data_pool_A = A_pool.pool_image(fake_A_tmp)
                #self.data_pool_B= np.random.random(size=[1,3,256,256]).astype("float32")
                #self.data_pool_A= np.random.random(size=[1,3,256,256]).astype("float32")
		#self.data_pool_B = np.ones([1,3,256,256], np.float32)
                #self.data_pool_A = np.ones([1,3,256,256], np.float32)
                 
    
                d_A_loss = exe.run(
                    d_A_trainer_program,
                    fetch_list=[d_A_trainer.d_loss_A],
                    feed={"input_B": self.data_B,
                          "fake_pool_B": self.data_pool_B})[0]
               ## for param in d_A_trainer_program.parameters():
               ##     print param.name
               ## print(d_A_params)
    
                d_B_loss = exe.run(
                    d_B_trainer_program,
                    fetch_list=[d_B_trainer.d_loss_B],
                    feed={"input_A": self.data_A,
                          "fake_pool_A": self.data_pool_A})[0]
                print("st:g_loss={},d_A_loss={},d_B_loss={}".format(sta_g_loss,d_A_loss,d_B_loss))
                # generate_p contains all parameters needed.
                #for param in d_A_trainer_program.global_block().all_parameters():
                #    static_params[param.name] = np.array(
                #        scope.find_var(param.name).get_tensor())
    
        dy_params = dict()

        with fluid.dygraph.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            cycle_gan = Cycle_Gan("cycle_gan",istrain=True)
            #D_A = Cycle_Gan("d_a",istrain=True,is_G=False,is_DA=True,is_DB=False)
            #D_B = Cycle_Gan("d_b",istrain=True,is_G=False,is_DA=False,is_DB=True)
            

            data_A = to_variable(self.data_A)
            data_B = to_variable(self.data_B)
            fake_pool_B = to_variable(self.data_pool_B)
            fake_pool_A = to_variable(self.data_pool_A)
            lr = 0.0002
            optimizer1 = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        100 * step_per_epoch, 120 * step_per_epoch,
                        140 * step_per_epoch, 160 * step_per_epoch,
                        180 * step_per_epoch
                    ],
                    values=[
                        lr , lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5)

            optimizer2 = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        100 * step_per_epoch, 120 * step_per_epoch,
                        140 * step_per_epoch, 160 * step_per_epoch,
                        180 * step_per_epoch
                    ],
                    values=[
                        lr , lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5)

            optimizer3 = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        100 * step_per_epoch, 120 * step_per_epoch,
                        140 * step_per_epoch, 160 * step_per_epoch,
                        180 * step_per_epoch
                    ],
                    values=[
                        lr , lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5)            
            for i in range(10):
###                fake_A,fake_B,cyc_A,cyc_B,diff_A,diff_B,\
###                fake_rec_A,fake_rec_B,idt_A,idt_B = cycle_gan(data_A,data_B,True,False,False)
###                cyc_A_loss = fluid.layers.reduce_mean(diff_A) * lambda_A
###                cyc_B_loss = fluid.layers.reduce_mean(diff_B) * lambda_B
###                cyc_loss = cyc_A_loss + cyc_B_loss
###    
###                #print("dy:fake_A=",fake_A.numpy()[0][:10])
###                #gan loss D_B(G_B(B))
###                g_A_tmp = fluid.layers.square(fake_rec_A - 1)
###                g_A_loss = fluid.layers.reduce_mean(fluid.layers.square(fake_rec_A-1))
###                g_B_loss = fluid.layers.reduce_mean(fluid.layers.square(fake_rec_B-1))
###    
###                g_loss = g_A_loss + g_B_loss
###                
###                # identity loss G_A
###                idt_loss_A = fluid.layers.reduce_mean(fluid.layers.abs(
###                    fluid.layers.elementwise_sub(
###                    x = data_B, y = idt_A))) * lambda_B * lambda_identity
###    
###                #print("dy_idt_loss_A",idt_loss_A.numpy())
###                idt_loss_B = fluid.layers.reduce_mean(fluid.layers.abs(
###                    fluid.layers.elementwise_sub(
###                    x = data_A, y = idt_B))) * lambda_A * lambda_identity
###    
###                idt_loss = fluid.layers.elementwise_add(idt_loss_A, idt_loss_B)
###    	        #print("cyc_loss:",cyc_loss.numpy())
###                #print("idt_loss:",idt_loss.numpy())
###                g_loss = cyc_loss + g_loss + idt_loss
###    
###                g_loss_out = g_loss.numpy()
###                print("dy:g_A_loss:{},g_B_loss:{},g_A_cyc_loss:{},g_B_cyc_loss:{},g_A_idt_loss:{},g_B_idt_loss:{}".format(g_A_loss.numpy(),g_B_loss.numpy(),cyc_A_loss.numpy(),cyc_B_loss.numpy(),idt_loss_A.numpy(),idt_loss_B.numpy()))    
                fake_A,fake_B,cyc_A,cyc_B,g_A_loss,g_B_loss,idt_loss_A,idt_loss_B,cyc_A_loss,cyc_B_loss,g_loss = cycle_gan(data_A,data_B,True,False,False)
                print("dy:g_A_loss:{},g_B_loss:{},g_A_cyc_loss:{},g_B_cyc_loss:{},g_A_idt_loss:{},g_B_idt_loss:{}".format(g_A_loss.numpy(),g_B_loss.numpy(),cyc_A_loss.numpy(),cyc_B_loss.numpy(),idt_loss_A.numpy(),idt_loss_B.numpy()))
                g_loss_out = g_loss.numpy()
                g_loss.backward()
                vars_G = []
                for param in cycle_gan.parameters():
                    if param.name[:52]=="cycle_gan/Cycle_Gan_0/build_generator_resnet_9blocks": 
                        vars_G.append(param)
                    #if param.name == "cycle_gan/Cycle_Gan_0/build_gen_discriminator_0/conv2d_0/Conv2D_0.conv2d_weights":
                    #    print ("G",param.name,param.gradient()[0][0][0][:10])
                optimizer1.minimize(g_loss,parameter_list=vars_G)             
                cycle_gan.clear_gradients()
    
    
                # optimize the d_A network
                #print(data_B.numpy())
                rec_B, fake_pool_rec_B = cycle_gan(data_B,fake_pool_B,False,True,False)
                #print("dy:rec_B=",rec_B.numpy()[0][0][0][:10])
		#print("dy:fake_pool_rec_B=",fake_pool_rec_B.numpy()[0][0][0][:10]) 
                d_loss_A = (fluid.layers.square(fake_pool_rec_B) +
                    fluid.layers.square(rec_B - 1)) / 2.0
                
                d_loss_A = fluid.layers.reduce_mean(d_loss_A) 
    
                #print("d_loss_A",d_loss_A.numpy())
                d_loss_A.backward()
                vars_da = []
                for param in cycle_gan.parameters():
                    if param.name[:47]=="cycle_gan/Cycle_Gan_0/build_gen_discriminator_0":
                        vars_da.append(param)
                optimizer2.minimize(d_loss_A,parameter_list=vars_da)                
                #cycle_gan.clear_gradients()
    
                # optimize the d_B network
    
                rec_A, fake_pool_rec_A = cycle_gan(data_A,fake_pool_A,False,False,True)
                d_loss_B = (fluid.layers.square(fake_pool_rec_A) +
                    fluid.layers.square(rec_A - 1)) / 2.0
                d_loss_B = fluid.layers.reduce_mean(d_loss_B)
    
                d_loss_B.backward()
                vars_db = []
                for param in cycle_gan.parameters():
                    if param.name[:47]=="cycle_gan/Cycle_Gan_0/build_gen_discriminator_1":
                        vars_db.append(param)
                optimizer3.minimize(d_loss_B,parameter_list=vars_db)
                cycle_gan.clear_gradients()

###            for p in discriminator.parameters():
###                dy_params[p.name] = p.numpy()
###            for p in generator.parameters():
###                dy_params[p.name] = p.numpy()
###
                d_loss_A = d_loss_A.numpy()
                d_loss_B = d_loss_B.numpy()
                #sta_g_loss = sta_g_loss.numpy()
                print ("dy: g_loss{},d_loss_A{},d_loss_B{}".format(g_loss_out,d_loss_A,d_loss_B))

        #self.assertEqual(dy_g_loss, static_g_loss)
        #self.assertEqual(dy_d_loss, static_d_loss)
        #for k, v in six.iteritems(dy_params):
        #    self.assertTrue(np.allclose(v, static_params[k]))


if __name__ == '__main__':
    unittest.main()
