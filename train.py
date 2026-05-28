import time
import os
import torch
from torch.utils.data import DataLoader
from GetDataSet_model import MakeSimulateDatasetforDemosaic
import torch.nn as nn
from model_GNN import DNGP
import copy
import argparse
import json
import cv2, math
import numpy
import utils
import random
import tqdm
import quality_index
from torch.optim.lr_scheduler import StepLR

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

def mosaic(img, MSFA):
    msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0], MSFA.shape[1]).to(img.device)
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            msfa_kernel[int(MSFA[i, j]), 0, i, j] += 1
    assert img.shape[1] == msfa_kernel.shape[0] * msfa_kernel.shape[1]
    img_mosaic = torch.nn.functional.conv2d(img, msfa_kernel, bias=None, stride=msfa_kernel.shape[2], groups=img.shape[1])
    img_mosaic = torch.nn.functional.pixel_shuffle(img_mosaic, upscale_factor=msfa_kernel.shape[2])
    return img_mosaic


def train(fusion_net: nn.Module, optimizer, train_dataloader, val_dataloader, args, scheduler):
    print('===>Begin Training!')
    start_epoch = 0
    if args.resume != "":
        start_epoch = int(args.resume) if "best" not in args.resume else int(args.resume.split("_")[-1])

    best_psnr, best_epoch = 0, 0
    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    mse_loss = nn.MSELoss()
    t = time.time()
    device = next(fusion_net.parameters()).device
    msfa_size = 4
    msfa_kernel = torch.zeros(msfa_size**2, 1, msfa_size, msfa_size).to(device)
    for i in range(msfa_size):
        for j in range(msfa_size):
            msfa_kernel[i*msfa_size+j, 0, i, j] += 1
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # train
        fusion_net.train()
        start_time = time.time()

        for cnt, data in enumerate(train_dataloader):
            # lrms_mosaic = data.cuda()
            mosaic, pan, gt = data[0].to(device), data[1].to(device), data[2].to(device)
            # mosaic = torch.nn.functional.pixel_shuffle(mosaic, upscale_factor=4)
                        
            optimizer.zero_grad()


            fused = fusion_net(mosaic, pan, msfa_kernel)
       

            loss = mse_loss(fused, gt)

           
            loss.backward()

            optimizer.step()
        if scheduler is not None:
            scheduler.step()
     
###############################################
        # # val
        # psnr_avg = 0.
        # fusion_net.eval()
        # for cnt, data in enumerate(val_dataloader):
        #     mosaic, pan, gt = data[0].to(device), data[1].to(device), data[2].to(device)
        #     # mosaic = torch.nn.functional.pixel_shuffle(mosaic, upscale_factor=4)
            
        #     with torch.no_grad():

        #         demosaic = fusion_net(mosaic, pan, msfa_kernel)
        #         demosaic = demosaic.detach()

        #     psnr_avg += quality_index.calc_psnr(gt, demosaic).item()
                
        # psnr_avg /= cnt+1
###################################################
        # val
        psnr_avg = 0.
        fusion_net.eval()
        with torch.no_grad():
            for cnt, data in enumerate(val_dataloader):
                try:
                    mosaic, pan, gt = data[0].to(device), data[1].to(device), data[2].to(device)
                    k_input = msfa_kernel.to(device) if hasattr(msfa_kernel, 'to') else msfa_kernel
                    demosaic = fusion_net(mosaic, pan, k_input)
                    demosaic = demosaic.detach()
                    psnr_avg += quality_index.calc_psnr(gt, demosaic).item()

                except RuntimeError as e:
                    if "out of memory" in str(e) or "CUDA" in str(e):
                        # 清理刚才失败留下的显存碎片
                        torch.cuda.empty_cache()
                        # 1. 将数据放在 CPU
                        mosaic, pan, gt = data[0].cpu(), data[1].cpu(), data[2].cpu()
                        # 2. 将模型本体移到 CPU
                        fusion_net = fusion_net.cpu()
                        # 3. 将 Kernel 移到 CPU (防止设备不匹配报错)
                        k_input = msfa_kernel.cpu() if hasattr(msfa_kernel, 'to') else msfa_kernel
                        # 4. 在 CPU 上推理
                        demosaic = fusion_net(mosaic, pan, k_input) #########
                        demosaic = demosaic.detach()
                        # 5. 计算指标
                        psnr_avg += quality_index.calc_psnr(gt, demosaic).item()
                        # 6. [关键步骤] 推理完成后，立即将模型移回 GPU
                        # 这样下一个 batch 依然会优先尝试用 GPU 跑，保证整体速度
                        fusion_net = fusion_net.to(device)
                    else:
                        # 如果是其他类型的错误（如维度不匹配），则直接报错，不隐藏问题
                        raise e
                
        psnr_avg /= cnt + 1
###################################################################################

        # save model with highest PSNR
        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            if best_epoch != 0:
                os.remove(os.path.join(args.dir_model, "best_{}.pth".format(best_epoch)))
            best_epoch = epoch
            torch.save(fusion_net.state_dict(), os.path.join(args.dir_model, "best_{}.pth".format(epoch)))

        if args.record is not False:
            record = []
            if os.path.exists(args.record):
                with open(args.record, "r") as f:
                    record = json.load(f)
            record.append({"epoch": epoch, "psnr": psnr_avg, "best_psnr": best_psnr, "best_epoch": best_epoch, "learning rate": optimizer.param_groups[0]["lr"]})
            with open(args.record, "w") as f:
                record = json.dump(record, f, indent=2)

        # save model at some frequency
        if epoch % args.save_freq == 0:
            torch.save(fusion_net.state_dict(), os.path.join(args.dir_model, f"{epoch}.pth"))
        
        # log
        print("Epoch: ", epoch,
            "PSNR: %.4f"%psnr_avg,
            "time: %.2f"%((time.time() - start_time) / 60), "min",
            "best_PSNR: %.4f"%best_psnr,
            "best_epoch: ", best_epoch,
            "learning rate: ",  optimizer.param_groups[0]["lr"]
            )
        
    print(f"Total time: {(time.time() - t) / 60} min")
    print("Best_epoch: {}, PSNR: {}".format(best_epoch, best_psnr))


def main(args):
    dir_idx = os.path.join("./", str(args.idx))
    if not os.path.exists(dir_idx):
        os.mkdir(dir_idx)
    args.cache_path = dir_idx

    dir_model = os.path.join(dir_idx, "model")
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    args.dir_model = dir_model

    if args.record is True:
        dir_record = os.path.join(dir_idx, "record")
        if not os.path.exists(dir_record):
            os.makedirs(dir_record)
        args.dir_record = dir_record
        args.record = os.path.join(dir_record, "record.json")
        if args.resume == "" and os.path.exists(args.record):
            os.remove(args.record)

    train_set = MakeSimulateDatasetforDemosaic(args, "train")
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    val_set = MakeSimulateDatasetforDemosaic(args, "test")
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    total_iterations = args.epochs * len(train_dataloader)
    print('total_iterations:{}'.format(total_iterations))

 
    demosaic_net = DNGP()

    if args.resume != "":
        backup_pth = os.path.join(dir_model, args.resume + ".pth")
        print("==> Load checkpoint: {}".format(backup_pth))
        demosaic_net.load_state_dict(torch.load(backup_pth), strict=False)
    else:
        print('==> Train from scratch')

    demosaic_net = demosaic_net.cuda()

    # r1 = torch.load("./exp/Pre_Train_R1/model/best_980.pth")["convs.weight"].cuda()
    # demosaic_net.r1 = r1
    # r2 = torch.load("./exp/Pre_Train_R2/model/best_796.pth")["convs.weight"].cuda()
    # demosaic_net.r2 = r2

    optimizer = torch.optim.Adam(demosaic_net.parameters(), args.lr)
    lr_scheduler = StepLR(optimizer, step_size=10000, gamma=0.2)

    


    train(demosaic_net, optimizer, train_dataloader, val_dataloader, args, lr_scheduler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--idx', type=int, default=1 , help='Index to identify models.')
    parser.add_argument('--dataset', type=str, default="CAVE", help='Dataset to be loaded.')
    parser.add_argument('--train_size', type=int, default=64,
                        help='Size of a MS image in a batch, usually 4x less than pan.')
    parser.add_argument('--msfa_size', type=int, default=4, help='Size of MSFA')
    parser.add_argument('--noise_level', nargs="+", type=float, default=[0.01, 0.05], help='Noise level when generating simulated data.')
    parser.add_argument('--spatial_ratio', type=int, default=2, help='Ratio of spatial resolutions between MS and PAN')
    parser.add_argument('--num_bands', type=int, default=16, help='Number of bands of a MS image.')
    parser.add_argument('--stride', type=int, default=32, help='Stride when crop an original image into patches.')
    # parser.add_argument('--stride', type=int, default=64, help='Stride when crop an original image into patches.') CAVE用的32
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training dataset.')
    parser.add_argument('--epochs', type=int, default=1000, help='Total epochs to train the model.')
    parser.add_argument('--iters_per_epoch', type=int, default=100, help='Iteration steps per epoch.') 
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Save the checkpoints of the model every [save_freq] epochs.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate to train the model.')
    parser.add_argument('--lr_decay', action="store_true", help='Determine if to decay the learning rate.')
    parser.add_argument('--device', type=str, default='0', help='Device to train the model.')
    parser.add_argument('--num_workers', type=int, default=1, help='Num_workers to train the model.')
    parser.add_argument('--resume', type=str, default='', help='Index of the model to be resumed, eg. 1000.')
    parser.add_argument('--data_path', type=str, default="../DATA", help='Path of the dataset.')
    parser.add_argument('--record', type=bool, default=True, help='Whether to record the PSNR of each epoch.')

    args = parser.parse_args()
    main(args)



  
