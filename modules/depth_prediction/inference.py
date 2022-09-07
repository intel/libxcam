import numpy as np
import cv2
import time

import os
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

from openvino.inference_engine import IECore


image_path = "C:/Users/Avenger/Desktop/gsoc/monodepth2/assets/test_image.jpg"
model_xml = "C:/Users/Avenger/Desktop/gsoc/monodepth2/outputs/monodepth2.xml"
model_bin = "C:/Users/Avenger/Desktop/gsoc/monodepth2/outputs/monodepth2.bin"
output_directory = "C:/Users/Avenger/Desktop/gsoc/monodepth2/assets/output_disp"

paths = [image_path]

def test_simple():
    ie = IECore()

    # Read IR
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Read and pre-process input images
    n, c, feed_height, feed_width = net.input_info[input_blob].input_data.shape

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            exec_net = ie.load_network(network=net, device_name="CPU")
            start_time = time.time()
            outputs = exec_net.infer(inputs={input_blob: input_image})
            disp = torch.from_numpy(outputs['263'])
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            output_name = os.path.splitext(os.path.basename(image_path))[0]

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            load_time = time.time() - start_time
            print("interface time(ms) : %.3f" % (load_time * 1000))
    print('-> Done!')


if __name__ == '__main__':
    test_simple()