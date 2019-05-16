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
one = 1


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

        g_a = G_A("g_a")
        d_a = D_A("d_a")
        d_b = D_B("d_b")

        losses = [[], []]
        t_time = 0


        for epoch in range(args.epoch):
            batch_id = 0
            for i in range(max_images_num):
                data_A = next(A_reader)
                data_B = next(B_reader)
                print(data_A[0])

                s_time = time.time()
                data_A = np.array([data_A[0].reshape(3,256,256)]).astype("float32")
                data_B = np.array([data_B[0].reshape(3,256,256)]).astype("float32")
                data_A = to_variable(data_A)
                data_B = to_variable(data_B)
                # optimize the g_A network
                fake_A,fake_B,cyc_A,cyc_B,diff_A,\
                diff_B,fake_rec_A,fake_rec_B,idt_A,idt_B = g_a(data_A,data_B)

                # cycle loss
                cyc_A_loss = fluid.layers.reduce_mean(diff_A) * lambda_A
                cyc_B_loss = fluid.layers.reduce_mean(diff_B) * lambda_B
                cyc_loss = cyc_A_loss + cyc_B_loss

                #gan loss D_A(G_A(A))
                g_A_loss = fluid.layers.reduce_mean(fluid.layers.square(fake_rec_A-one))
                #gan loss D_B(G_B(B))
                g_B_loss = fluid.layers.reduce_mean(fluid.layers.square(fake_rec_B-one))

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

                #g_loss.backward()
                optimizer1 = optimizer_setting()
                optimizer1.minimize(g_loss)
                g_a.clear_gradients()

                print("epoch id: %d, batch step: %d, loss: %f" % (epoch, batch_id, g_loss_out))

    ###            g_A_loss, g_A_cyc_loss, g_A_idt_loss, g_B_loss, g_B_cyc_loss, g_B_idt_loss, fake_A_tmp, fake_B_tmp = exe.run(
    ###                gen_trainer_program,
    ###                fetch_list=[gen_trainer.G_A, gen_trainer.cyc_A_loss, gen_trainer.idt_loss_A, gen_trainer.G_B, gen_trainer.cyc_B_loss,
    ###                            gen_trainer.idt_loss_B, gen_trainer.fake_A, gen_trainer.fake_B],
    ###                feed={"input_A": tensor_A,
    ###                      "input_B": tensor_B})

                fake_pool_B = B_pool.pool_image(fake_B)
                print(fake_pool_B[0])
                fake_pool_B = np.array([fake_pool_B[0].reshape(3,256,256)]).astype("float32")
                fake_pool_B = to_variable(fake_pool_B)
                print(fake_pool_B)
                fake_pool_A = A_pool.pool_image(fake_A)

                # optimize the d_A network
                rec_A, fake_pool_rec_A = d_a(data_B,fake_pool_B)
                d_loss_A = (fluid.layers.square(fake_pool_rec_A) +
                    fluid.layers.square(rec_A - 1)) / 2.0
                d_loss_A = fluid.layers.reduce_mean(d_loss_A)

                optimizer2 = optimizer_setting()
                optimizer2.minimize(d_loss_A)
                d_a.clear_gradients()

                # optimize the d_B network

                rec_B, fake_pool_rec_B = d_b(data_A,fake_pool_A)
                d_loss_B = (fluid.layers.square(self.fake_pool_rec_B) +
                    fluid.layers.square(rec_B - 1)) / 2.0
                d_loss_B = fluid.layers.reduce_mean(d_loss_B)

                optimize3 = optimizer_setting()
                optimize3.minimize(d_loss_B)
                d_b.clear_gradients()

                batch_time = time.time() - s_time
                t_time += batch_time
                print(
                    "epoch{}; batch{}; d_A_loss: {}; g_A_loss: {}; g_A_cyc_loss: {};\
                     g_A_idt_loss: {}; d_B_loss: {}; g_B_loss: {}; g_B_cyc_loss: {}; \
                     g_B_idt_loss: {}; Batch_time_cost: {:.2f}".format(epoch, batch_id,\
                      d_A_loss[0], g_A_loss[0], g_A_cyc_loss[0], g_A_idt_loss[0],\
                       d_B_loss[0], g_B_loss[0], g_B_cyc_loss[0], g_B_idt_loss[0], batch_time))
                with open('logging_{}.txt'.format(args.changes), 'a') as log_file:
                    now = time.strftime("%c")
                    log_file.write(
                    "time: {}     epoch{}; batch{}; d_A_loss: {}; g_A_loss: {}; \
                    g_A_cyc_loss: {}; g_A_idt_loss: {}; d_B_loss: {}; \
                    g_B_loss: {}; g_B_cyc_loss: {}; g_B_idt_loss: {}; \
                    Batch_time_cost: {:.2f}\n".format(now, epoch, \
                        batch_id, d_A_loss[0], g_A_loss[ 0], g_A_cyc_loss[0], \
                        g_A_idt_loss[0], d_B_loss[0], g_B_loss[0], \
                        g_B_cyc_loss[0], g_B_idt_loss[0], batch_time))
                losses[0].append(g_A_loss[0])
                losses[1].append(d_A_loss[0])
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
