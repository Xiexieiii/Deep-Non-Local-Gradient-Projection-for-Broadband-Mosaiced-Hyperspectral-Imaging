from model_GNN9 import GPPNN
import torch
import os
import numpy
import scipy.io as scio
import cv2
import argparse
from tqdm.contrib import tzip
import tqdm
import utils
import h5py
import copy, json

def MSFA_filter(dt, msfa):
    dt_t = torch.from_numpy(dt).float().permute(2, 0, 1).unsqueeze(0)
    msfa_kernel = torch.zeros(msfa.shape[0]*msfa.shape[1], 1, msfa.shape[0], msfa.shape[1]).to(dt_t.device)
    for i in range(msfa.shape[0]):
        for j in range(msfa.shape[1]):
            msfa_kernel[i*msfa.shape[0]+j, 0, i, j] += 1
    mosaic_t = torch.nn.functional.conv2d(dt_t, msfa_kernel, stride=msfa_kernel.shape[-1], padding=0, groups=dt_t.shape[1])
    # mosaic_t = torch.nn.functional.pixel_shuffle(mosaic_t, 4)
    mosaic = mosaic_t[0].permute(1, 2, 0).numpy()
    return mosaic

def create_F_31_2_16():
    F = numpy.array(
        [
            [2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 2,2, 4, 6, 8, 9, 11, 13, 15, 16, 17, 21, 21, 20, 19, 19, 17, 15, 12, 11, 10, 9],
            [5, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 4, 9, 12, 15, 20, 24, 29, 25, 21, 17, 13, 9, 6, 3, 2, 1, 1, 1, 1],
            [0, 0, 0, 0, 2, 8, 12, 17, 21, 23, 24, 24, 23, 21, 18, 16, 13, 10, 7, 5, 3, 2, 2, 1, 1, 1, 0,0,0,0,0],
            [1, 2, 4, 8, 12, 15, 19, 22, 25, 21, 18, 16, 12, 9, 7, 4, 2, 1, 1, 1, 1, 1,0, 0, 0, 0, 0, 0,1,1,1],
            [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [15, 17, 19, 20, 18, 17, 16, 15, 13, 11, 10, 8, 6, 5, 3, 1, 1, 1, 1, 1, 0, 0,0, 0,0, 0,0, 0,0, 0,0],

            [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 7,9, 11, 13, 15, 16, 17, 19, 20, 21, 22, 25, 23, 20, 18, 15, 14, 13, 12],
            [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 13, 16, 17, 19, 22, 25, 29, 27, 24, 20, 18, 15, 13, 11, 10, 8, 6, 3, 2,2],
            [1, 1, 3, 6, 8, 11, 15, 19, 21, 23, 24, 24, 23, 21, 18, 16, 13, 11, 9, 7, 6, 4, 2, 2, 1, 1, 1,1,1,1,1],
            [8,10,11,13, 16, 17, 19, 22, 25,27, 29, 27, 24,22, 20, 18, 17, 15, 13,12, 11,10, 8, 6, 3, 2, 2,1,1,1,1],
            [21, 22, 22, 21, 20, 19, 18, 16, 14, 12, 11, 9, 7, 6, 5, 4, 3, 3, 4, 5,7, 5,4, 3, 3, 2, 2, 1, 1, 1, 1],
            [4, 5, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 21, 20, 18, 16, 14, 12, 10, 8, 6, 4, 3, 3, 4, 5, 6, 7, 8, 9,9],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            [1, 1, 2, 3, 4, 3, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [6, 6, 6, 6, 7, 8,9, 9, 10,10, 11, 11, 12, 12,12,11, 11, 11, 10, 10,9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5],


           




            # [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
            # [5, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 21, 18, 15, 11, 8, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 4, 8, 12, 15, 19, 22, 25, 21, 18, 16, 12, 9, 7, 4, 2, 1, 1, 1, 1, 1],
            # [5, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 4, 9, 12, 15, 20, 24, 29, 25, 21, 17, 13, 9, 6, 3, 2, 1, 1, 1, 1],
            # [5, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 21, 18, 15, 11, 8, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # [8, 9, 11, 13, 15, 16, 17, 19, 21, 23, 24, 24, 23, 21, 18, 16, 13, 10, 7, 5, 3, 2, 2, 2, 3, 4, 6, 7, 8, 9,10],
            # [4, 5, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 21, 20, 18, 16, 14, 12, 10, 8, 6, 4, 3, 3, 4, 5, 6, 7, 8, 9,9],
            # [10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 22, 21, 20, 19, 18, 16, 14, 12, 11, 9, 7, 6, 5, 4, 3, 3, 4, 5,7, 8],
            # [6, 5, 5, 6, 8, 9, 11, 12, 13, 14, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 8],
            # [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5, 6, 7, 7],
            # [9, 8, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8, 9, 9],
            # [7, 6, 5, 4, 3, 3, 4, 6, 8, 10, 11, 12, 11, 10, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            # [8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9]
        ])
    for band in range(16):
        div = numpy.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div
    return F


def create_F_31_2_1():
    F = numpy.array(
        [[2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    F=numpy.sum(F,axis=0,keepdims=True)
    for band in range(1):
        div = numpy.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div
    return F


def input_matrix_wpn(inH, inW, msfa_size):

    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    for i in range(0,msfa_size):
        h_offset_coord[i::msfa_size, :, 0] = (i+1)/msfa_size
        w_offset_coord[:, i::msfa_size, 0] = (i+1)/msfa_size

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2)
    return pos_mat

def main(args):
    dir_dataset = os.path.join("./", args.dataset)
    if not os.path.exists(dir_dataset):
        os.mkdir(dir_dataset)

    dir_idx = os.path.join(dir_dataset, str(args.idx))
    if not os.path.exists(dir_idx):
        os.mkdir(dir_idx)

    ALL_net = GPPNN()
    print("==> Load demosaicing_net checkpoint: {}".format(args.load_demosaic_model))
    ALL_net.load_state_dict(torch.load(args.load_demosaic_model, map_location="cpu"), strict=False)
    

    ALL_net = ALL_net.cuda()
    ALL_net.eval()

    data_path = os.path.join(args.data_path, args.dataset, "test")
    ids, mosaics, pans, gts = [], [], [], []

    if args.simulate == True:
        data_names = os.listdir(data_path)
        MSFA = numpy.array([[0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [8, 9, 10, 11],
                            [12, 13, 14, 15]])
        msfa_size = 4
        msfa_kernel = torch.zeros(msfa_size**2, 1, msfa_size, msfa_size).cuda()
        for i in range(msfa_size):
            for j in range(msfa_size):
                msfa_kernel[i*msfa_size+j, 0, i, j] += 1
        
        for data_name in tqdm.tqdm(data_names):
            data_dir = os.path.join(data_path, data_name)
            if args.dataset == "CAVE":
                hrms = scio.loadmat(data_dir)['b']
            elif args.dataset == "ICVL":
                hrms = h5py.File(data_dir)["rad"][:]
                hrms = numpy.rot90(hrms.transpose(2, 1, 0))
                hrms /= hrms.max((0, 1))

            elif args.dataset == "chikusei_mix":
                hrms = scio.loadmat(data_dir)["chikusei"]

            hrms = hrms[:hrms.shape[0]//(args.msfa_size)*(args.msfa_size),
                                        :hrms.shape[1]//(args.msfa_size)*(args.msfa_size),
                                        0:].astype(numpy.float32)
               
               #
            MSFA = numpy.array([[0, 1, 2, 3],
                                [4, 5, 6, 7],
                                [8, 9, 10, 11],
                                [12, 13, 14, 15]])
            if args.dataset != "chikusei_mix":
                spe_res = create_F_31_2_16()
                spe_res1 = create_F_31_2_1()
            else:
                with open("SRF128_2_16.json", "r", encoding="utf-8") as f:
                    spe_res = numpy.array(json.load(f), dtype=float)
                    spe_res = spe_res.transpose(-1, -2)
                with open("SRF128_2_1.json", "r", encoding="utf-8") as f:
                    spe_res1 = numpy.array(json.load(f), dtype=float)
                    spe_res1 = spe_res1.transpose(-1, -2)

            # 光谱响应   31-->16
            mix = numpy.tensordot(hrms, spe_res, axes=([2], [1]))
            mosaic = MSFA_filter(mix, MSFA)           
            pan = numpy.tensordot(hrms,spe_res1,axes=([2], [1]))

            mosaics.append(mosaic)
            pans.append(pan)
            gts.append(hrms)
            ids.append(data_name.split(".")[0])

    if args.simulate == False:
        mosaic_path = os.path.join(data_path, "mosaic")
        pan_path = os.path.join(data_path, "pan")
        mosaic_imgs = os.listdir(mosaic_path)
        mosaic_imgs.sort()
        pan_imgs = os.listdir(pan_path)
        pan_imgs.sort()
        assert len(mosaic_imgs) == len(pan_imgs), "Length mismatch between MS and PAN images!"

        for mosaic_name, pan_name in tqdm.tqdm(tzip(mosaic_imgs, pan_imgs)):
            with open(os.path.join(mosaic_path, mosaic_name), "rb") as f:
                mosaic_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                mosaic = numpy.zeros((255*4, 276*4, 1))
                for i in range(args.msfa_size):
                    for j in range(args.msfa_size):
                        mosaic[i::args.msfa_size, j::args.msfa_size, 0] = mosaic_raw[255*276*(i*args.msfa_size+j): 255*276*(i*args.msfa_size+j+1)].reshape(255, 276)
                for i in range(args.msfa_size):
                    for j in range(args.msfa_size):
                        mosaic[i::args.msfa_size, j::args.msfa_size, 0] = mosaic[i::args.msfa_size, j::args.msfa_size, 0] / mosaic[i::args.msfa_size, j::args.msfa_size, 0].max()
                mosaics.append(mosaic)
            with open(os.path.join(pan_path, pan_name), "rb") as f:
                pan_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                pan = pan_raw.reshape((255*8, 276*8, 1))
                pan = pan / pan.max((0, 1))
                pans.append(pan)
            ids.append(mosaic_name.split(".")[0])

    dir_mat = os.path.join(dir_idx, "result", "mat")
    if not os.path.exists(dir_mat):
        os.makedirs(dir_mat)
    print("Start to generate the real_world results!")

    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    for cnt, (idx, mosaic, pan) in enumerate(tqdm.tqdm(tzip(ids, mosaics, pans))):
        mosaic_tensor = torch.from_numpy(mosaic.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0).cuda()
        pan_tensor = torch.from_numpy(pan.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            hrms_tensor = ALL_net(mosaic_tensor, pan_tensor, msfa_kernel.cuda()).detach()[0].cpu()

        
        hrms = hrms_tensor.permute(1, 2, 0).numpy()

        if args.mosaic_save == True:
            # mosaic_numpy = numpy.zeros((mosaic.shape[0]//MSFA.shape[0], mosaic.shape[1]//MSFA.shape[1], MSFA.shape[0]*MSFA.shape[1]))
            # for i in range(MSFA.shape[0]):
            #     for j in range(MSFA.shape[1]):
            #         mosaic_numpy[:, :, i*MSFA.shape[1]+j] = mosaic[i::MSFA.shape[0], j::MSFA.shape[1], 0]
            if not os.path.exists(os.path.join(dir_mat, "mosaic")):
                os.mkdir(os.path.join(dir_mat, "mosaic"))
            scio.savemat(os.path.join(dir_mat, "mosaic", f"{idx}.mat"), {'mosaic': mosaic})
        if args.pan_save == True:
            if not os.path.exists(os.path.join(dir_mat, "pan")):
                os.mkdir(os.path.join(dir_mat, "pan"))
            scio.savemat(os.path.join(dir_mat, "pan", f"{idx}.mat"), {'pan': pan})
        if args.demosaic_save == True:
            if not os.path.exists(os.path.join(dir_mat, "demosaic")):
                os.mkdir(os.path.join(dir_mat, "demosaic"))
            scio.savemat(os.path.join(dir_mat, "demosaic", f"{idx}.mat"), {'demosaic': demosaic})
        if args.simulate == True and args.gt_save == True:
            gt = gts[cnt]
            if not os.path.exists(os.path.join(dir_mat, "gt")):
                os.mkdir(os.path.join(dir_mat, "gt"))
            scio.savemat(os.path.join(dir_mat, "gt", f"{idx}.mat"), {'gt': gt})
        if not os.path.exists(os.path.join(dir_mat, "fused")):
                os.mkdir(os.path.join(dir_mat, "fused"))
        scio.savemat(os.path.join(dir_mat, "fused", f"{idx}.mat"), {'fused': hrms})
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--idx', type=int, default=1, help='Index to identify models.')
    parser.add_argument('--simulate', action='store_true', default=False, help='Determine whether to do simulating experiment')
    parser.add_argument('--real_world', action='store_true', default=False, help='Determine whether to do real-world experiment')
    parser.add_argument('--mosaic_save', action='store_true', default=False, help='Determine whether to generate mosaic data.')
    parser.add_argument('--pan_save', action='store_true', default=False, help='Determine whether to generate pan data.')
    parser.add_argument('--demosaic_save', action='store_true', default=False, help='Determine whether to generate demosaic data.')
    parser.add_argument('--gt_save', action='store_true', default=False, help='Determine whether to generate gt data. Effective only when in simulated dataset.')
    parser.add_argument('--spatial_ratio', type=int, default=4, help='Ratio of spatial resolutions between MS and PAN')
    parser.add_argument('--dataset', type=str, default="CAVE", help='Type of satellite data.')
    parser.add_argument('--num_bands', type=int, default=16, help='Number of bands of a MS image.')
    parser.add_argument('--msfa_size', type=int, default=4, help='Size of MSFA')
    parser.add_argument('--data_path', type=str, default="../DataSet/", help='Path of the dataset.')
    parser.add_argument('--load_ps_model', type=str, default='', help='The pansharpening model to be loaded.')
    parser.add_argument('--load_demosaic_model', type=str, default='', help='The demosaicing model to be loaded.')

    args = parser.parse_args()
    main(args)
