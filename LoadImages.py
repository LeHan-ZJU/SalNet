import os
import glob
import math
import random
import argparse
import collections
import tensorflow as tf
from rgb_lab import rgb_to_lab
from ImageProcess import preprocess, preprocess_lab

parser = argparse.ArgumentParser()   #argparse是Python标准库中推荐使用的编写命令行程序的工具
parser.add_argument("--mode", default="", choices=["train", "test", "export"])   #选择训练模型
parser.add_argument("--input_dir", default="",help="path to folder containing images")  #输入图像路径
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
a = parser.parse_args()           #add default #add_argument:读入命令行参数，该调用有多个参数

CROP_SIZE = 256 #图像目标尺寸
Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch") #构造一个带字段名的元组

def load_examples():  #读取图像
    if a.input_dir is None or not os.path.exists(a.input_dir): #判断输入图像路径是否存在
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg")) #默认输入图像格式为jpg格式
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png")) #若输入路径中不存在紧迫感格式，则读取png格式
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files") #若输入图像路径中不存在jpg和png格式的图像，则报错

    def get_name(path):                  #获取名称
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))  #如果图像以数字命名，则对其进行排序
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:                    #判别并转换图像的颜色空间。调用的函数为ImageProcessing.py中定义的图像处理函数
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            a_images = preprocess(raw_input[:,:width//2,:])
            b_images = preprocess(raw_input[:,width//2:,:])

    inputs, targets = [b_images, a_images]    #因为输入图像是成对拼接的，所以分割输入和Ground-truth

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)  #同步图像操作的种子，以便我们对输入和输出图像执行相同的操作
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:                                   #判断并调整图像尺寸到256*256
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)             #转化输入图像尺寸

    with tf.name_scope("target_images"):
        target_images = transform(targets)            #转化Ground-truth尺寸

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(      #返回图像加载结果
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )