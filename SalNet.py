#encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import random
import collections
import math
import time

from save_images import save_images
from EdgeConstraint import conv2d1
from LoadImages import load_examples
from ImageProcess import preprocess, deprocess, augment
#import cv2
#调用GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'  #单引号中为GPU序号，可以根据闲置GPU来更改
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config, ...)
#参数定义
parser = argparse.ArgumentParser()   #argparse是Python标准库中推荐使用的编写命令行程序的工具
parser.add_argument("--mode", default="", choices=["train", "test", "export"])  #add default  #required=True,
parser.add_argument("--output_dir", default="" ,help="where to put output files")#add default #required=True,
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default="", help="directory with checkpoint to resume training from or use for testing") #已训练模型位置

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)") #训练步数
parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")#迭代次数，1个epoch等于使用训练集中的全部样本训练一次
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps") #更新每个summary_freq步骤结果
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps") #显示每个progress_freq步骤的进度
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable") #保存模型的每一个save_freq步骤

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)") #输出图像长宽比(宽/高)
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch") #每个batch的图像数
parser.add_argument("--ngf", type=int, default=64, help="number of unet filters in first conv layer")  #卷积层的滤波器个数
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256") #在裁剪到256x256之前将图像缩放到这个大小
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")  #优化算法学习率
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam") #momentum term of adam
parser.add_argument("--l1_weight", type=float, default=100, help="weight on L1 term for model gradient")  #L1损失权重
parser.add_argument("--CrossEntropy_weight", type=float, default=100, help="weight on Sptaial term for model gradient") #交叉熵损失权重
parser.add_argument("--Conv_weight", type=float, default=1, help="weight on Conv term for model gradient") #卷积约束项损失权重

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"]) #定义输出图像类型，默认为png格式
a = parser.parse_args()

CROP_SIZE = 256 #图像目标尺寸

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch") #调用collections.namedtuple构造一个带字段名的元组
Model = collections.namedtuple("Model", "outputs, Unet_loss_Conv, Unet_loss_L1, Unet_loss_CrossEntropy, Unet_grads_and_vars, train")

def conv(batch_input, out_channels, stride):  #定义卷积层
    with tf.variable_scope("conv"):    #tf.variable_scope 函数作用是,在一个作用域scope内共享一些变量
        in_channels = batch_input.get_shape()[3] #通道数
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID") #调用tensorflow库中的tf.nn.conv2d函数实现卷积操作
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x) #tf.identity是返回一个一模一样新的tensor的op，这会增加一个新节点到gragh中
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

#批正则化
def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)              #计算均值和方差，以进行批正则化
        variance_epsilon = 1e-5 
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)#调用tensorflow库中的tf.nn.batch_normalization函数实现
        return normalized

#定义反卷积层
def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()] #卷积操作对象为四位张量，各维度分别是：batch数、图像尺寸、通道数
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02)) #卷积核
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")#调用tensorflow库中的tf.nn.conv2d_transpose函数实现反卷积操作
        return conv

#构建U-形网络-网络框架
def My_Unet(inputs, outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(inputs, a.ngf, stride=2)
        layers.append(output)

    layer_specs = [ #编码器部分，由一组步长为2的卷积层构成，a.ngf * x 为该层的卷积核数目
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)                #激活函数lrelu
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)                     #进行批正则化
            layers.append(output)

    layer_specs = [ #解码器部分，由一组反卷积层构成，a.ngf * x 为该层的卷积核数目
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]
    print('layer_specs=')
    print(layer_specs)
    num_encoder_layers = len(layers) #获取编码器的层数
    print(num_encoder_layers) #输出编码器的层数
    #建立编码器与解码器的对应层之间的跳跃连接（SKIP-CONNECTION）
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1      #跳跃连接（SKIP CONNCETION）数目
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)   #对符合条件的卷基层和其对应的反卷积层实现跳跃连接

            rectified = tf.nn.relu(input)
            print('skip_layer=')
            print(skip_layer)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)  #反卷积操作
            output = batchnorm(output) #批正则化

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)  #调用tf.nn.dropout实现dropout操作

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, model_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)     #连接两个矩阵
        rectified = tf.nn.relu(input)   #
        output = deconv(rectified, outputs_channels)   #调用deconv函数实现反卷积操作
        output = tf.tanh(output)  #激活函数tf.tanh
        layers.append(output)

    return layers[-1]

def create_model(inputs, targets):

    with tf.variable_scope("unet") as scope:
        out_channels = int(targets.get_shape()[-1])  #获取输出通道数
        outputs = My_Unet(inputs, out_channels) #定义输出为U-net模型的运行结果

    with tf.name_scope("unet_loss"):  #定义模型的损失函数

        Unet_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs)) #定义L1损失，输出和Groundtruth之间差值的绝对值
        #调用tf.nn.sigmoid_cross_entropy_with_logits函数实现输出图与ground-truth之间用交叉熵
        Unet_loss_CrossEntropy = tf.reduce_mean(tf.abs(tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None, labels=targets, logits=outputs, name=None)))
        #边缘约束项：output跟target做完我们定义的卷积后用L1
        Unet_loss_Conv = tf.reduce_mean(tf.abs(conv2d1(outputs) - conv2d1(targets))) #调研我们定义的EdgeConstraint中的卷积实现边缘约束项
        #模型的损失函数：由L1、交叉熵、以及卷积约束项三部分组成
        Unet_loss =Unet_loss_L1 * a.l1_weight +Unet_loss_CrossEntropy* a.CrossEntropy_weight +Unet_loss_Conv* a.Conv_weight


    with tf.name_scope("unet_train"):  #模型训练

        Unet_tvars = [var for var in tf.trainable_variables() if var.name.startswith("unet")]
        Unet_optim = tf.train.AdamOptimizer(a.lr, a.beta1)                               #优化算法为Adam算法
        Unet_grads_and_vars = Unet_optim.compute_gradients(Unet_loss, var_list=Unet_tvars) #基于梯度下降
        Unet_train = Unet_optim.apply_gradients(Unet_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)  #调用tensorflow中的tf.train.ExponentialMovingAverage来实现滑动平均模型，他使用指数衰减来计算变量的移动平均值。
    update_losses = ema.apply([Unet_loss_Conv, Unet_loss_L1, Unet_loss_CrossEntropy])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        Unet_loss_L1=ema.average(Unet_loss_L1),   #返回L1损失值
        Unet_loss_CrossEntropy=ema.average(Unet_loss_CrossEntropy),  #返回交叉熵损失值
        Unet_loss_Conv=ema.average(Unet_loss_Conv),  #返回边缘约束项
        Unet_grads_and_vars=Unet_grads_and_vars,
        outputs=outputs,                              #模型输出
        train=tf.group(update_losses, incr_global_step, Unet_train),
    )

def append_index(filesets, step=False):  #将模型的测试结果写入index.html，以便于查看测试结果
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

def main():      #模型的主函数
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)       #随机数生成种子
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:            #模型测试时，实现检测是否存在已经训练好的模型（checkpoint）
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = { "ngf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # export the model to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)

        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
        # convert grayscale to RGB
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)

        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("unet"):
            batch_output = deprocess(My_Unet(preprocess(batch_input), 3))

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]

        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)           #根据之前选择的输出图像类型来确定输出的图像格式
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:                   #生成tensorflow会话
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)    #tensorflow对训练变量checkpoint的保存
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))  #meta
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)   #运行创建模型

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:         #数据处理
        inputs = augment(examples.inputs, examples.targets)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"), #tf.map_fn()：从0维度的 elems 中解压的张量列表上的映射
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs) #输出一个包含图像的summary,这个图像是通过一个4维张量构建的，这个张量的四个维度如下所示：[batch_size,height, width, channels]

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets) #输出一个包含图像的summary,这个图像是通过一个4维张量构建的

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs) #输出一个包含图像的summary,这个图像是通过一个4维张量构建的

    tf.summary.scalar("unet_loss_L1", model.Unet_loss_L1)  #输出一个含有标量值的Summary protocol buffer
    tf.summary.scalar("unet_loss_CrossEntropy", model.Unet_loss_CrossEntropy)
    tf.summary.scalar("unet_loss_Conv", model.Unet_loss_Conv)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var) #添加一个直方图的summary,它可以用于可视化数据的分布情况

    for grad, var in model.Unet_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad) #添加一个直方图的summary,它可以用于可视化数据的分布情况

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":  #当mode参数为test时，进行模型测试
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)

            print("wrote index at", index_path)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["Unet_loss_L1"] = model.Unet_loss_L1
                    fetches["Unet_loss_CrossEntropy"] = model.Unet_loss_CrossEntropy
                    fetches["Unet_loss_Conv"] = model.Unet_loss_Conv

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("Unet_loss_L1", results["Unet_loss_L1"])
                    print("Unet_loss_CrossEntropy", results["Unet_loss_CrossEntropy"])
                    print("Unet_loss_Conv", results["Unet_loss_Conv"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break

main()
