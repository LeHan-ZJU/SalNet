import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="" ,help="where to put output files")   

parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])   
a = parser.parse_args()

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")   
    if not os.path.exists(image_dir):    
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:  
            filename = name + "-" + kind + ".png"     
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets
