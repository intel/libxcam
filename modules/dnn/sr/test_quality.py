import os
import math
import numpy as np
import cv2
import torch
from glob import glob
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor

from architecture import arch

settings = {
    "test_dataset_folder"   : "dataset/",
    "output_dataset_folder" : "dataset_SR/",   
    "to_test"               : ["Set5/", "Set14/", "B100/"],
    "subfolder_lr"          : "LR_bicubic/",
    "subfolder_hr"          : "HR/",
    "cuda"                  : True,
    "model"                 : "espcn", #srcnn, fsrcnn, espcn, edsr, srgan, esrgan, or prosr
    "scale"                 : 4             # 2 or 4
}

def main():

    for dataset in settings['to_test']:

        d_path = settings['test_dataset_folder'] + dataset + settings['subfolder_lr'] + 'X' + str(settings['scale'])
        model_op_path = settings['output_dataset_folder'] + dataset + settings['model'] +'x' + str(settings['scale'])

        if os.path.isdir(model_op_path):
            continue
        else:
            os.makedirs(model_op_path)

        with torch.no_grad():
            model = arch(settings['model'], settings['scale'], settings['cuda']).getModel()
            model.eval()

            for lr_img in glob(d_path + '/*'):
                base = os.path.splitext(os.path.basename(lr_img))[0]
                lr_img = Image.open(lr_img)
                dt = prepare_image(lr_img)
                out = model(dt['img'])
                path = model_op_path + '/' + base + '.png'
                tensor_as_img(out, base, dt['cb'], dt['cr'], path)
                print(base + " done")


    for dataset in settings['to_test']:

        print("testing " + dataset)

        d_path = settings['test_dataset_folder'] + dataset + settings['subfolder_hr'] 
        model_op_path = settings['output_dataset_folder'] + dataset + settings['model'] +'x' + str(settings['scale'])

        test_Y = False  # True: test Y channel only; False: test RGB channels

        PSNR_all = []
        SSIM_all = []
        img_list = sorted(glob(d_path + '/*'))

        for i, img_path in enumerate(img_list):
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            im_GT = cv2.imread(img_path) / 255.
            path = model_op_path + '/' + base_name + 'x' + str(settings['scale'])+ '.png'
            im_Gen = cv2.imread(path) / 255.

            if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
                im_GT_in = bgr2ycbcr(im_GT)
                im_Gen_in = bgr2ycbcr(im_Gen)
            else:
                im_GT_in = im_GT
                im_Gen_in = im_Gen

            ########### Hack start ###############

            cropped_Gen = im_Gen_in
            cropped_GT = im_GT_in[0 : im_Gen_in.shape[0], 0 : im_Gen_in.shape[1]]

            if not (cropped_Gen.shape[0] == cropped_GT.shape[0] and cropped_Gen.shape[1] == cropped_GT.shape[1]):
                print(cropped_GT.shape, cropped_Gen.shape)
                print(img_path, 'and', path, 'don\'t have same dimensions')
                continue 

            ############ Hack end ################

            # calculate PSNR and SSIM
            PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)

            SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
            print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
                i + 1, base_name, PSNR, SSIM))
            PSNR_all.append(PSNR)
            SSIM_all.append(SSIM)
        
        print(dataset +': Average PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
            sum(PSNR_all) / len(PSNR_all),
            sum(SSIM_all) / len(SSIM_all)))


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def prepare_image(lr_img):
    y, cb, cr = "", "", ""
    if settings['model'] in ['fsrcnn', 'srcnn', 'espcn']:
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


def tensor_as_img(out, base, cb, cr, path):
    out_img = out.cpu().data[0].numpy()
    out_img *= 255.0
    out_img = out_img.clip(0, 255)

    if settings['model'] in ['fsrcnn', 'srcnn', 'espcn'] :
        out_img_y = Image.fromarray(np.uint8(out_img[0]), mode='L')
        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    else:
        out_img = np.transpose(out_img[[0, 1, 2], :, :], (1, 2, 0))
        out_img = Image.fromarray(np.uint8(out_img), 'RGB')

    out_img.save(path)


if __name__ == '__main__':
    main()
