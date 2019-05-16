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
add_arg('input',             str,   None, "The images to be infered.")
add_arg('output',            str,   "./infer_result_single", "The directory the infer result to be saved to.")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('input_style',        str,  "A",       "The style of the input, A or B")
add_arg('use_gpu',           bool,  True,       "Whether to use GPU to train.")
# yapf: enable


def infer(args):
    data_shape = [-1, 3, 256, 256]
    input = fluid.layers.data(name='input', shape=data_shape, dtype='float32')
    model_name = 'gen'
    if args.input_style == "A":
        fake = build_generator_resnet_9blocks(input, name="g_A")
    elif args.input_style == "B":
        fake = build_generator_resnet_9blocks(input, name="g_B")
    else:
        raise "Input with style [%s] is not supported." % args.input_style
    
    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    for var in fluid.default_main_program().global_block().all_parameters():
        print(var.name)
    print(args.init_model + '/' + model_name)
    fluid.io.load_persistables(exe, args.init_model + "/" + model_name)
    print('load params done')

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for file in glob.glob(args.input):
        print "read %s" % file
        image_name = os.path.basename(file)
        image = Image.open(file).convert('RGB')
        image = image.resize((256, 256), Image.BICUBIC)
        image = np.array(image) / 127.5 - 1
###        if len(image.shape) != 3:
###            continue
        image = image[:, :, 0:3].astype("float32")
        data = image.transpose([2, 0, 1])[np.newaxis,:]
###        data = image.transpose([2, 0, 1])[np.newaxis, :].astype("float32")
###        image = Image.open(file).convert('RGB')
###        image = image.resize((256, 256), Image.BICUBIC)
###        image = np.array(image).transpose([2, 0, 1]).astype('float32')
###        image = image / 255.0
###        image = (image - 0.5) / 0.5
###        data = image[np.newaxis, :]
        tensor = core.LoDTensor()
        tensor.set(data, place)

        fake_temp = exe.run(fetch_list=[fake.name], feed={"input": tensor})
        fake_temp = np.squeeze(fake_temp[0]).transpose([1, 2, 0])
        input_temp = np.squeeze(data).transpose([1, 2, 0])

        imsave(args.output + "/fake_" + image_name, (
            (fake_temp + 1) * 127.5).astype(np.uint8))


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    infer(args)
