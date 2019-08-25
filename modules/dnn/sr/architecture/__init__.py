from pathlib import Path
import torch

from . import srcnn 
from . import fsrcnn 
from . import edsr 
from . import espcn 
from . import srgan 
from . import esrgan 
from . import prosr 

class arch(object):
	def __init__(self, model = 'espcn', upscale_factor = 4, cuda = False):
		super(arch, self).__init__()
		self.model = model
		self.upscale_factor = upscale_factor
		self.cuda = cuda
		

	def getModel(self):
		tmp = ''
		if self.model == 'srcnn':
			tmp = srcnn.SRCNN(1, 64, self.upscale_factor)
		
		elif self.model == 'fsrcnn':
			tmp = fsrcnn.FSRCNN(1, self.upscale_factor)
		
		elif self.model == 'srgan':
			tmp = srgan.SRGAN(self.upscale_factor)
		
		elif self.model == 'edsr':
			tmp = edsr.EDSR(self.upscale_factor)
		
		elif self.model == 'espcn':
			tmp = espcn.ESPCN(self.upscale_factor)
		
		elif self.model == 'prosr':
			tmp = prosr.ProSR(residual_denseblock = True, num_init_features = 160, growth_rate = 40, level_config = [[8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8], [8]], max_num_feature = 312, ps_woReLU = False, level_compression = -1, bn_size = 4, res_factor = 0.2, max_scale = 8)

		elif self.model == 'esrgan':
			tmp = esrgan.ESRGAN(
				3, 3, 64, 23, gc=32, upscale=self.upscale_factor, norm_type=None, act_type='leakyrelu', \
                mode='CNA', res_scale=1, upsample_mode='upconv'
			)

		pth = Path.cwd() / 'architecture' / self.model / 'pretrained_models' / ('x' + str(self.upscale_factor) + '.pth')
		
		if self.cuda:
			tmp.load_state_dict(torch.load(pth, map_location = 'cuda'))
			tmp = tmp.cuda()
		else:
			tmp.load_state_dict(torch.load(pth, map_location = 'cpu'))
			tmp = tmp.cpu()

		return tmp