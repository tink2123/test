from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import sys
import paddle
import argparse
import functools
import time
import numpy as np
from scipy.misc import imsave
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid import core
import data_reader
from utility import add_arguments, print_arguments, ImagePool
from trainer import *
from paddle.fluid.dygraph.base import to_variable
import six
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   1,          "Minibatch size.")
add_arg('epoch',             int,   2,        "The number of epoched to be trained.")
add_arg('output',            str,   "./output_0", "The directory the model and the test result to be saved to.")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('save_checkpoints',  bool,  True,       "Whether to save checkpoints.")
add_arg('run_test',          bool,  True,       "Whether to run test.")
add_arg('use_gpu',           bool,  True,       "Whether to use GPU to train.")
add_arg('profile',           bool,  False,       "Whether to profile.")
add_arg('run_ce',            bool,  False,       "Whether to run for model ce.")
add_arg('changes',           str,   "None",    "The change this time takes.")
# yapf: enable

lambda_A = 10.0
lambda_B = 10.0
lambda_identity = 0.5


def optimizer_setting():
    lr=0.0002
    optimizer = fluid.optimizer.Adam(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=[
                12 * step_per_epoch, 32 * step_per_epoch,
                52 * step_per_epoch, 72 * step_per_epoch
            ],
            values=[
                lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
            ]),
        beta1=0.5)
    return optimizer

def train(args):
    dy_param_init_value = {}
    with fluid.dygraph.guard():

        max_images_num = data_reader.max_images_num()
        shuffle = True
        if args.run_ce:
            np.random.seed(10)
            fluid.default_startup_program().random_seed = 90
            max_images_num = 1
            shuffle = False
        data_shape = [-1] + data_reader.image_shape()
        print(data_shape)

    ###    g_A_trainer = G_A(input_A, input_B)
    ###    g_B_trainer = GBTrainer(input_A, input_B)
    ###    gen_trainer = GTrainer(input_A, input_B)
    ###    d_A_trainer = DATrainer(input_B, fake_pool_B)
    ###    d_B_trainer = DBTrainer(input_A, fake_pool_A)

        # prepare environment
        #place = fluid.CPUPlace()
        #if args.use_gpu:
        #    place = fluid.CUDAPlace(0)
        #exe = fluid.Executor(place)2
        #exe.run(fluid.default_startup_program())
        A_pool = ImagePool()
        B_pool = ImagePool()

        A_reader = paddle.batch(
            data_reader.a_reader(shuffle=shuffle), args.batch_size)()
        B_reader = paddle.batch(
            data_reader.b_reader(shuffle=shuffle), args.batch_size)()

        def checkpoints(epoch):
            out_path = args.output + "/checkpoints/" + str(epoch)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fluid.io.save_persistables(
                exe, out_path + "/gen", main_program=gen_trainer.program)
    ###        fluid.io.save_persistables(
    ###            exe, out_path + "/g_b", main_program=g_B_trainer.program)
            fluid.io.save_persistables(
                exe, out_path + "/d_a", main_program=d_A_trainer.program)
            fluid.io.save_persistables(
                exe, out_path + "/d_b", main_program=d_B_trainer.program)
            print("saved checkpoint to {}".format(out_path))
            sys.stdout.flush()

        G = Cycle_Gan("g",istrain=True,is_G=True,is_DA=False,is_DB=False)
        D_A = Cycle_Gan("d_a",istrain=True,is_G=False,is_DA=True,is_DB=False)
        D_B = Cycle_Gan("d_b",istrain=True,is_G=False,is_DA=False,is_DB=True)

        def load_dict(self, stat_dict, include_sublayers=True):
            self._loaddict_holder = stat_dict
            for name, item in self.__dict__.get('_parameters', None).items():
                if item.name in stat_dict:
                    var = item._ivar.value()
                    tensor = var.get_tensor()
                    tensor.set(stat_dict[item.name].numpy(),
                               framework._current_expected_place())

            if include_sublayers:
                for layer_name, layer_item in self._sub_layers.items():
                    if layer_item is not None:
                        layer_item.load_dict(stat_dict)

        losses = [[], []]
        t_time = 0

        optimizer1 = optimizer_setting()
        optimizer2 = optimizer_setting()
        optimizer3 = optimizer_setting()
        vars1 = []
##        for param in G.parameters():
##            if param.name[:44]=="g/Cycle_Gan_0/build_generator_resnet_9blocks":
##                print (param.name)
##                vars1.append(var.name)
##
        for epoch in range(args.epoch):
            batch_id = 0
            for i in range(max_images_num):
                #if epoch == 0 and batch_id ==1:
                #    for param in G.parameters():
                #        print(param.name,param.shape)
                data_A = next(A_reader)
                data_B = next(B_reader)
                #print(data_A[0])

                s_time = time.time()
                data_A = np.array([data_A[0].reshape(3,256,256)]).astype("float32")
                data_B = np.array([data_B[0].reshape(3,256,256)]).astype("float32")
                data_A = to_variable(data_A)
                data_B = to_variable(data_B)

                # optimize the g_A network
                fake_A,fake_B,cyc_A,cyc_B,diff_A,diff_B,fake_rec_A,fake_rec_B,idt_A,idt_B = G(data_A,data_B)

                #print(fake_B.numpy()) 
                # cycle loss
                cyc_A_loss = fluid.layers.reduce_mean(diff_A) * lambda_A
                cyc_B_loss = fluid.layers.reduce_mean(diff_B) * lambda_B
                cyc_loss = cyc_A_loss + cyc_B_loss
                print(cyc_loss)
                #for k,v in six.iteritems(fluid.framework._dygraph_tracer()._ops):
                #    print(v.type)  
          
                #print(fluid.framework._dygraph_tracer()._ops)
                #gan loss D_A(G_A(A))
                g_A_loss = fluid.layers.reduce_mean(fluid.layers.square(fake_rec_A-1))
                #gan loss D_B(G_B(B))
                g_B_loss = fluid.layers.reduce_mean(fluid.layers.square(fake_rec_B-1))

                g_loss = g_A_loss + g_B_loss

                # identity loss G_A
                idt_loss_A = fluid.layers.reduce_mean(fluid.layers.abs(
                    fluid.layers.elementwise_sub(
                    x = data_B, y = idt_A))) * lambda_B * lambda_identity

                # identity loss G_B

                idt_loss_B = fluid.layers.reduce_mean(fluid.layers.abs(
                    fluid.layers.elementwise_sub(
                    x = data_A, y = idt_B))) * lambda_A * lambda_identity

                idt_loss = fluid.layers.elementwise_add(idt_loss_A, idt_loss_B)

                g_loss = cyc_loss + g_loss + idt_loss

                g_loss_out = g_loss.numpy()

                g_loss.backward()
                #optimizer1.minimize(g_loss)
                vars_G = []
                for param in G.parameters():
                    if param.name[:44]=="g/Cycle_Gan_0/build_generator_resnet_9blocks": 
                        #print (param.name)
                        vars_G.append(param)
                optimizer1.minimize(g_loss,parameter_list=vars_G)                
                #fluid.dygraph.save_persistables(G.state_dict(),"./G")
                G.clear_gradients()

                #for param in G.parameters():
                #    dy_param_init_value[param.name] = param.numpy()

                #restore = fluid.dygraph.load_persistables("./G")
                #G.load_dict(restore)


                print("epoch id: %d, batch step: %d, g_loss: %f" % (epoch, batch_id, g_loss_out))



                fake_pool_B = B_pool.pool_image(fake_B).numpy()
                fake_pool_B = np.array([fake_pool_B[0].reshape(3,256,256)]).astype("float32")
                fake_pool_B = to_variable(fake_pool_B)

                fake_pool_A = A_pool.pool_image(fake_A).numpy()
                fake_pool_A = np.array([fake_pool_A[0].reshape(3,256,256)]).astype("float32")
                fake_pool_A = to_variable(fake_pool_B)

                # optimize the d_A network
                rec_B, fake_pool_rec_B = D_A(data_B,fake_pool_B)
                if batch_id == 0:
                    print("rec_B",rec_B.numpy())
                    print("fake_pool_rec_B",fake_pool_rec_B.numpy())
                d_loss_A = (fluid.layers.square(fake_pool_rec_B) +
                    fluid.layers.square(rec_B - 1)) / 2.0
                d_loss_A = fluid.layers.reduce_mean(d_loss_A)

                d_loss_A.backward()
                vars_da = []
                for param in D_A.parameters():
                    if param.name[:41]=="d_a/Cycle_Gan_0/build_gen_discriminator_0":
                        #print (param.name)
                        vars_da.append(param)
                optimizer2.minimize(d_loss_A,parameter_list=vars_da)
                #fluid.dygraph.save_persistables(D_A.state_dict(),
                #                                    "./G")
                D_A.clear_gradients()

                #for param in G.parameters():
                #    dy_param_init_value[param.name] = param.numpy()
                #restore = fluid.dygraph.load_persistables("./G")
                #D_A.load_dict(restore)
                #D_A.clear_gradients()

                # optimize the d_B network

                rec_A, fake_pool_rec_A = D_B(data_A,fake_pool_A)
                d_loss_B = (fluid.layers.square(fake_pool_rec_A) +
                    fluid.layers.square(rec_A - 1)) / 2.0
                d_loss_B = fluid.layers.reduce_mean(d_loss_B)

                d_loss_B.backward()
                vars_db = []
                for param in D_B.parameters():
                    if param.name[:41]=="d_b/Cycle_Gan_0/build_gen_discriminator_1":
                        #print (param.name)
                        vars_db.append(param)
                optimizer2.minimize(d_loss_B,parameter_list=vars_db)
                #fluid.dygraph.save_persistables(D_B.state_dict(),
                #                                    "./G")
                D_B.clear_gradients()

                #for param in D_B.parameters():
                #    dy_param_init_value[param.name] = param.numpy()                
                #restore = fluid.dygraph.load_persistables("./G")
                #D_B.load_dict(restore)
                batch_time = time.time() - s_time
                t_time += batch_time
                print(
                    "epoch{}; batch{}; d_A_loss: {}; g_A_loss: {}; g_A_cyc_loss: {}; g_A_idt_loss: {}; d_B_loss: {}; g_B_loss: {}; g_B_cyc_loss:  {}; g_B_idt_loss: {};Batch_time_cost: {:.2f}".format(epoch, batch_id,d_loss_A.numpy()[0], g_A_loss.numpy()[0],cyc_A_loss.numpy()[0], idt_loss_A.numpy()[0], d_loss_B.numpy()[0], g_B_loss.numpy()[0],cyc_B_loss.numpy()[0],idt_loss_B.numpy()[0], batch_time))
                with open('logging_{}.txt'.format(args.changes), 'a') as log_file:
                    now = time.strftime("%c")
                    log_file.write(
                    "time: {}; epoch{}; batch{}; d_A_loss: {}; g_A_loss: {}; \
                    g_A_cyc_loss: {}; g_A_idt_loss: {}; d_B_loss: {}; \
                    g_B_loss: {}; g_B_cyc_loss: {}; g_B_idt_loss: {}; \
                    Batch_time_cost: {:.2f}\n".format(now, epoch, \
                        batch_id, d_loss_A[0], g_A_loss[ 0], cyc_A_loss[0], \
                        idt_loss_A[0], d_loss_B[0], g_A_loss[0], \
                        cyc_B_loss[0], idt_loss_B[0], batch_time))
                losses[0].append(g_A_loss[0])
                losses[1].append(d_loss_A[0])
                sys.stdout.flush()
                batch_id += 1

            if args.run_test and not args.run_ce:
                test(epoch)
            if args.save_checkpoints and not args.run_ce:
                checkpoints(epoch)

if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    if args.profile:
        if args.use_gpu:
            with profiler.profiler('All', 'total'):
                train(args)
###            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
###                train(args)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                train(args)
    else:
        train(args)
