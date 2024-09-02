import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="" ,help="where to put output files")    #输出结果的路径

parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])    #输出图像的格式，默认为.png格式
a = parser.parse_args()

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")     #图像输出路径
    if not os.path.exists(image_dir):      #如果输出路径文件夹不存在，则创建一个
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:   #依次存储图像，每组图像包含输入（inputs）、输出（outputs）以及Ground-truth（targets）
            filename = name + "-" + kind + ".png"      #图像命名，图像格式为png格式
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets