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
    data_shape = [-1, 3, 256, 256]
    inputA = fluid.layers.data(name='inputA', shape=data_shape, dtype='float32')
    inputB = fluid.layers.data(name='inputB', shape=data_shape, dtype='float32')
    model_name = 'gen'
###    if args.input_style == "A":
###        model_name = 'g_a'
###        fake = build_generator_resnet_9blocks(input, name="g_A")
###    elif args.input_style == "B":
###        model_name = 'g_b'
###        fake = build_generator_resnet_9blocks(input, name="g_B")
###    else:
###        raise "Input with style [%s] is not supported." % args.input_style
    fakeB = build_generator_resnet_9blocks(inputA, name='g_A')
    fakeA = build_generator_resnet_9blocks(inputB, name='g_B')
    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    for var in fluid.default_main_program().global_block().all_parameters():
        print(var.name)
    print(args.init_model + '/' + model_name)

###    fluid.io.load_vars(exe, args.init_model + "/" + model_name)
    fluid.io.load_persistables(exe, args.init_model + "/" + model_name)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for file in glob.glob(args.inputA):
        print "read %s" % file
        fileB = file.replace('testA', 'testB')#.replace('leftImg8bit', 'gtFine_color').replace('png','jpg')
        ###fileB = file.replace('A', 'B')
        print "read %s" % fileB
        image_nameA = os.path.basename(file)
        imageA = Image.open(file).convert('RGB')
        imageA = imageA.resize((256, 256), Image.BICUBIC)
        imageA = np.array(imageA).transpose([2, 0, 1]).astype('float32')
        imageA = imageA / 255.0
        imageA = (imageA - 0.5) / 0.5
        dataA = imageA[np.newaxis, :]
       ### image = np.array(image) / 127.5 - 1
       ### if len(image.shape) != 3:
       ###     continue
       ### data = image.transpose([2, 0, 1])[np.newaxis, :].astype("float32")
        image_nameB = os.path.basename(fileB)
        imageB = Image.open(fileB).convert('RGB')
        imageB = imageB.resize((256, 256), Image.BICUBIC)
        imageB = np.array(imageB).transpose([2, 0, 1]).astype('float32')
        imageB = imageB / 255.0
        imageB = (imageB - 0.5) / 0.5
        dataB = imageB[np.newaxis, :]

        tensorA = core.LoDTensor()
        tensorB = core.LoDTensor()
        tensorA.set(dataA, place)
        tensorB.set(dataB, place)

        fakeB_temp, fakeA_temp = exe.run(fetch_list=[fakeB.name, fakeA.name], feed={"inputA": tensorA, "inputB": tensorB})
        fakeB_temp = np.squeeze(fakeB_temp[0]).transpose([1, 2, 0])
        fakeA_temp = np.squeeze(fakeA_temp[0]).transpose([1, 2, 0])
        inputA_temp = np.squeeze(dataA).transpose([1, 2, 0])
        inputB_temp = np.squeeze(dataB).transpose([1, 2, 0])

        imsave(args.output + "/fakeB_" + image_nameA, (
            (fakeB_temp + 1) * 127.5).astype(np.uint8))

        imsave(args.output + "/fakeA_" + image_nameB, (
            (fakeA_temp + 1) * 127.5).astype(np.uint8))


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    infer(args)
