import argparse
import functools
import os
from PIL import Image
from paddle.fluid import core
import paddle.fluid as fluid
import paddle
import numpy as np
from scipy.misc import imsave
from model import *
import glob
from utility import add_arguments, print_arguments
from paddle.fluid.dygraph.base import to_variable
from trainer import *
import data_reader

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('inputA',             str,   None, "The images to be infered.")
add_arg('inputB',             str,   None, "The images to be infered.")
add_arg('output',            str,   "./infer_result", "The directory the infer result to be saved to.")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('input_style',        str,  "A",       "The style of the input, A or B")
add_arg('use_gpu',           bool,  True,       "Whether to use GPU to train.")
# yapf: enable


def infer(args):

    with fluid.dygraph.guard():
        A_test_reader = data_reader.a_test_reader()
        B_test_reader = data_reader.b_test_reader()
        cycle_gan = Cycle_Gan("cycle_gan")
        restore = fluid.dygraph.load_persistables("./G/0")
        cycle_gan.load_dict(restore)
        out_path = args.output + "/test"
        cycle_gan.eval()
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for data_A , data_B in zip(A_test_reader(), B_test_reader()): 
                A_name = data_A[1]
                B_name = data_B[1]
                print(A_name)
                print(B_name)
                tensor_A = np.array([data_A[0].reshape(3,256,256)]).astype("float32")
                tensor_B = np.array([data_B[0].reshape(3,256,256)]).astype("float32")
                data_A_tmp = to_variable(tensor_A)
                data_B_tmp = to_variable(tensor_B)
                #print("!!!!!!!!test()!!!!!!!!",data_B_tmp.numpy())
                fake_A_temp,fake_B_temp,cyc_A_temp,cyc_B_temp,g_A_loss,g_B_loss,idt_loss_A,idt_loss_B,cyc_A_loss,cyc_B_loss,g_loss = cycle_gan(data_A_tmp,data_B_tmp,True,False,False)

                fake_A_temp = np.squeeze(fake_A_temp.numpy()[0]).transpose([1, 2, 0])
                fake_B_temp = np.squeeze(fake_B_temp.numpy()[0]).transpose([1, 2, 0])
                input_A_temp = np.squeeze(data_A[0]).transpose([1, 2, 0])
                input_B_temp = np.squeeze(data_B[0]).transpose([1, 2, 0])
                imsave(out_path + "/fakeB_" + A_name, (
                    (fake_B_temp + 1) * 127.5).astype(np.uint8))
                imsave(out_path + "/fakeA_" + B_name, (
                    (fake_A_temp + 1) * 127.5).astype(np.uint8))
                imsave(out_path + "/inputA_" + A_name, (
                    (input_A_temp + 1) * 127.5).astype(np.uint8))
                imsave(out_path + "/inputB_" + B_name, (
                    (input_B_temp + 1) * 127.5).astype(np.uint8))

if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    infer(args)

