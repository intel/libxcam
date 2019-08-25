import os
from glob import glob

import cv2
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToTensor

from architecture import arch

settings = {
	"cuda"	: False,
	"input"	: "lr/",
	"output": "hr/",
	"image_types" : [".jpg", ".png"],
	"video_types" : [".mp4", ".avi"],
	"model"	: "srgan",		# srcnn, fsrcnn, espcn, edsr, srgan, esrgan, or prosr
	"scale" : 4				# 2 or 4
}


def prepare_image(lr_img):
	y, cb, cr = "", "", ""
	if settings['model'] in ['fsrcnn', 'srcnn', 'espcn', 'srgan']:
		lr_img = lr_img.convert('YCbCr')
		y, cb, cr = lr_img.split()
	else:
		y = lr_img.convert('RGB')
	
	dt = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
	if settings['cuda']:
		dt = dt.cuda()

	return {
		"img" : dt,
		"cb"  : cb,
		"cr"  : cr
	}


def tensor_as_img(out, base, cb, cr, save = False):
	out_img = out.cpu().data[0].numpy()
	out_img *= 255.0
	out_img = out_img.clip(0, 255)

	if settings['model'] in ['fsrcnn', 'srcnn', 'espcn', 'srgan'] :
		out_img_y = Image.fromarray(np.uint8(out_img[0]), mode='L')
		out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
		out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
		out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
	else:
		out_img = np.transpose(out_img[[0, 1, 2], :, :], (1, 2, 0))
		out_img = Image.fromarray(np.uint8(out_img), 'RGB')

	if save:
		out_img.save(settings['output']+'{0}_{1}_{2}.png'.format(base, settings['scale'], settings['model']))

	return out_img


def main():

	lr_images = []
	for img_type in settings["image_types"]:
		lr_images.extend(glob(settings["input"] + "*" + img_type)) 

	lr_videos = []
	for vid_type in settings["video_types"]:
		lr_videos.extend(glob(settings["input"] + "*" + vid_type)) 
	
	model = arch(settings['model'], settings['scale'], settings['cuda']).getModel()
	model.eval()

	with torch.no_grad():
		for lr_img in lr_images:
			base = os.path.splitext(os.path.basename(lr_img))[0]
			lr_img = Image.open(lr_img)
			dt = prepare_image(lr_img)
			out = model(dt['img'])
			tensor_as_img(out, base, dt['cb'], dt['cr'], save = True)
			print(base + " done")

	with torch.no_grad():
		for lr_vid in lr_videos:
			base = os.path.splitext(os.path.basename(lr_vid))[0]
			videoCapture = cv2.VideoCapture(lr_vid)
			fps = videoCapture.get(cv2.CAP_PROP_FPS)
			size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * settings['scale']),
				int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * settings['scale'])

			output_name = settings['output']+'{0}_{1}_{2}.avi'.format(base, settings['scale'], settings['model'])
			videoWriter = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, size)

			success, frame = videoCapture.read()
			while success:
				img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
				dt = prepare_image(img)

				out = model(dt['img'])
				out_img = tensor_as_img(out, base, dt['cb'], dt['cr'])

				out_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)
				torch.cuda.empty_cache()
				videoWriter.write(out_img)
				success, frame = videoCapture.read()

			print(base + " done")


if __name__ == '__main__':
    main()