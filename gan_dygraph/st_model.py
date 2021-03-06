from st_layers import *
import paddle.fluid as fluid


def st_build_resnet_block(inputres, dim, name="resnet", use_bias=False):
    out_res = fluid.layers.pad2d(inputres, [1, 1, 1, 1], mode="reflect")
    out_res = conv2d(out_res, dim, 3, 1, 0.02, name=name + "_c0", use_bias=use_bias)

    # TODO dropout here
    ###if use_dropout:
    ###out_res = fluid.layers.dropout(out_res, dropout_prob = 0.5)

    out_res = fluid.layers.pad2d(out_res, [1, 1, 1, 1], mode="reflect")
    out_res = conv2d(
        out_res, dim, 3, 1, 0.02, name=name + "_c1", relu=False, use_bias=use_bias)
###    return fluid.layers.relu(out_res + inputres)
    return out_res + inputres


def st_build_generator_resnet_9blocks(inputgen, name="generator"):
    '''The shape of input should be equal to the shape of output.'''
    pad_input = fluid.layers.pad2d(inputgen, [3, 3, 3, 3], mode="reflect")
###    o_c1 = conv2d(pad_input, 32, 7, 1, 0.02, name=name + "_c1")
###    o_c2 = conv2d(o_c1, 64, 3, 2, 0.02, "SAME", name + "_c2")
###    o_c3 = conv2d(o_c2, 128, 3, 2, 0.02, "SAME", name + "_c3")
###    o_r1 = build_resnet_block(o_c3, 128, name + "_r1")
###    o_r2 = build_resnet_block(o_r1, 128, name + "_r2")
###    o_r3 = build_resnet_block(o_r2, 128, name + "_r3")
###    o_r4 = build_resnet_block(o_r3, 128, name + "_r4")
###    o_r5 = build_resnet_block(o_r4, 128, name + "_r5")
###    o_r6 = build_resnet_block(o_r5, 128, name + "_r6")
###    o_r7 = build_resnet_block(o_r6, 128, name + "_r7")
###    o_r8 = build_resnet_block(o_r7, 128, name + "_r8")
###    o_r9 = build_resnet_block(o_r8, 128, name + "_r9")
###    o_c4 = deconv2d(o_r9, [128, 128], 64, 3, 2, 0.02, "SAME", name + "_c4")
###    o_c5 = deconv2d(o_c4, [256, 256], 32, 3, 2, 0.02, "SAME", name + "_c5")
###    o_c6 = conv2d(o_c5, 3, 7, 1, 0.02, "SAME", name + "_c6", relu=False)
    o_c1 = conv2d(pad_input, 32, 7, 1, 0.02, name=name + "_c0")
    o_c2 = conv2d(o_c1, 64, 3, 2, 0.02, 1, name + "_c1")
    o_c3 = conv2d(o_c2, 128, 3, 2, 0.02, 1, name + "_c2")
    #print("test_st:9blocks_tmp")
    ##fluid.layers.Print(o_c3,print_tensor_name=True,summarize=10)
    o_r1 = st_build_resnet_block(o_c3, 128, name + "_r0")
    o_r2 = st_build_resnet_block(o_r1, 128, name + "_r1")
    o_r3 = st_build_resnet_block(o_r2, 128, name + "_r2")
    o_r4 = st_build_resnet_block(o_r3, 128, name + "_r3")
    o_r5 = st_build_resnet_block(o_r4, 128, name + "_r4")
    o_r6 = st_build_resnet_block(o_r5, 128, name + "_r5")
    o_r7 = st_build_resnet_block(o_r6, 128, name + "_r6")
    o_r8 = st_build_resnet_block(o_r7, 128, name + "_r7")
    o_r9 = st_build_resnet_block(o_r8, 128, name + "_r8")
    o_c4 = deconv2d(o_r9, 64, 3, 2, 0.02, [1,1], [0,1,0,1], name + "_d0")
    o_c5 = deconv2d(o_c4, 32, 3, 2, 0.02, [1,1], [0,1,0,1], name + "_d1")
    o_p2 = fluid.layers.pad2d(o_c5, [3, 3, 3, 3], mode="reflect")
    o_c6 = conv2d(o_p2, 3, 7, 1, 0.02, name = name + "_c3", norm=False, relu=False, use_bias=True)
    out_gen = fluid.layers.tanh(o_c6, name + "_t1")
    #fluid.layers.Print(out_gen,print_tensor_name=True,summarize=10)
    return out_gen


def st_build_gen_discriminator(inputdisc, name="discriminator"):
    o_c1 = conv2d(
        inputdisc,
        64,
        4,
        2,
        0.02,
        1,
        name + "_c0",
        norm=False,
        relufactor=0.2, use_bias=True)
    o_c2 = conv2d(o_c1, 128, 4, 2, 0.02, 1, name + "_c1", relufactor=0.2)
    o_c3 = conv2d(o_c2, 256, 4, 2, 0.02, 1, name + "_c2", relufactor=0.2)
    o_c4 = conv2d(o_c3, 512, 4, 1, 0.02, 1, name + "_c3", relufactor=0.2)
    o_c5 = conv2d(
        o_c4, 1, 4, 1, 0.02, 1, name + "_c4", norm=False, relu=False, use_bias=True)
    return o_c5
