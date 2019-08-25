import torch
from architecture import arch

settings = {
	"model"                 : "esrgan", #srcnn, fsrcnn, espcn, edsr, srgan, esrgan, or prosr
	"scale"                 : 4             # 2 or 4
}

def main():
	model = arch(settings['model'], settings['scale'], False).getModel()
	model.eval()

	dummy_input = torch.randn(1, 3, 200, 200, device = 'cpu')
	output_names = ["the_output"] # will use this in DnnSuperResolution

	with torch.no_grad():
		torch.onnx.export(
			model, 
			dummy_input, 
			settings['model'] + '_x' + str(settings['scale']) + '.onnx',
			output_names = output_names, 
			verbose=True
		)

if __name__ == '__main__':
    main()
