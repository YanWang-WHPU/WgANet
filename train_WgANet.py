import numpy as np
from glob import glob
import logging
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random, time
import itertools
import argparse
import csv
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from utils_Mamba import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.UNetFormer import UNetFormer
from model.RS3Mamba import RS3Mamba, load_pretrained_ckpt
from model.WgANet import WgANet, load_pretrained_ckpt
from model.UNetMamba import UNetMamba
from model.CMTFNet import CMTFNet
from model.TransUNet_model.vit_seg_modeling import VisionTransformer as ViT_seg
from model.TransUNet_model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.freqmamba_arch import FreqMamba
from model.MCAFTMNet import MCAFTM
from model.MIFNet import MIFNet

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

if MODEL == 'UNetformer':
    net = UNetFormer(num_classes=N_CLASSES).cuda()
elif MODEL == 'WgANet':
    net = WgANet(num_classes=N_CLASSES).cuda()
    net = load_pretrained_ckpt(net)
elif MODEL == 'RS3Mamba':
    net = RS3Mamba(num_classes=N_CLASSES).cuda()
    net = load_pretrained_ckpt(net)
elif MODEL == 'UNetMamba':
    net = UNetMamba(num_classes=N_CLASSES,pretrained=True).cuda()
elif MODEL == 'CMTFNet':  
    net = CMTFNet(num_classes=N_CLASSES).cuda()
elif MODEL == 'TransUNet':
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = N_CLASSES
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(256 / 16), int(256 / 16))
    net = ViT_seg(config_vit, img_size=256, num_classes=N_CLASSES).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))
elif MODEL == 'FreqMamba':  
    net = FreqMamba(out_channels=N_CLASSES).cuda()
elif MODEL == 'MCAFTM':  
    net = MCAFTM(num_classes=N_CLASSES, image_size=256).cuda()
elif MODEL == 'MIFNet':  
    net = MIFNet(num_classes=N_CLASSES).cuda()


params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch',       type=int,   default=50,   help='epoch number')
parser.add_argument('--save_path',           type=str, default='./results/SG/Vaihingen/WgAnet_222/',    help='the path to save models and logs')
opt = parser.parse_args()
save_path        = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

base_lr = 0.01

# Set up logging
logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};save_path:{}'.format(opt.epoch,base_lr,BATCH_SIZE,WINDOW_SIZE,save_path))

# Load the datasets
print("training : ", str(len(train_ids)) + ", testing : ", str(len(test_ids)) + ", Stride_Size : ", str(Stride_Size), ", BATCH_SIZE : ", str(BATCH_SIZE))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)


base_lr = 0.01
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15, 25, 35], gamma=0.1)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30], gamma=0.1)

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    if DATASET == 'Urban':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids) # 读取测试图像
        test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids) # 读取测试标签
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64') - 1) for id in test_ids)
    elif DATASET == 'Potsdam' :
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids) # 读取测试图像
        # test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids) # 读取测试的dsm图像
        test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids) # 读取测试标签
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    elif DATASET == 'Vaihingen':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids) # 读取测试图像
        test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids) # 读取测试标签
        # test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids) # 读取测试的dsm图像
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    # Switch the network to inference mode
    with torch.no_grad():
        for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                # Do the inference
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
            
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


loss_values = []
train_acc = []
MIoU_values = []
lr_values = [] 


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.80
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss = loss_calc(output, target, weights)

            loss_values.append(loss.item())

            loss.backward()
            optimizer.step()

            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            lr_values.append(current_lr)

            # losses[iter_] = loss.data
            # mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                accuracy_value = accuracy(pred, gt)
                train_acc.append(accuracy_value)
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLr: {:.6f}\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'], loss.data, accuracy(pred, gt)))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.2e}, Loss: {:.4f} , Accuracy: {}'.
                    format( e, epochs, batch_idx, len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'], loss.data, accuracy_value))
            iter_ += 1

            # del (data, target, loss)
            del data, target, loss, output
            torch.cuda.empty_cache()

            ## !!! You can increase the frequency of testing to find better models.
            if iter_ % 1000 == 0:
                # We validate with the largest possible stride for faster computing
                net.eval()
                if DATASET == 'Urban':
                    MIoU = test(net, test_ids, stride=Stride_Size)
                elif DATASET == 'Potsdam' :
                    MIoU = test(net, test_ids, all=False, stride=Stride_Size)  
                elif DATASET == 'Vaihingen':              
                    MIoU = test(net, test_ids, all=False, stride=Stride_Size) 

                MIoU_values.append(MIoU.item()) 
                net.train()
                if MIoU > MIoU_best:
                    torch.save(net.state_dict(), save_path+'{}_epoch{}_{}'.format(MODEL, e, MIoU))
                    MIoU_best = MIoU

        # logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( e, epochs, loss.data)) # 记录日志和损失。
        ## 绘制accuary曲线
        plt.figure(figsize=(10, 5))  # 可选：设置图表大小
        plt.plot(train_acc, label='Train Accuracy')  
        plt.title('Train_acc Curve')  
        plt.xlabel('Epoch')  
        plt.ylabel('acc')  
        plt.grid(True)  
        plt.savefig(os.path.join(save_path, 'acc.png'))
        # plt.savefig(os.path.join(save_path, f'acc_epoch{e}.png'))
        plt.show()   
        ## 绘制loss曲线
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, label='Loss_fusion')
        plt.title('Individual Losses Curve')  
        plt.xlabel('Steps')  
        plt.ylabel('Loss')  
        # plt.legend()
        plt.grid(True)  
        plt.savefig(os.path.join(save_path, 'individual_losses.png'))
        ## 绘制学习率曲线
        # plt.savefig(os.path.join(save_path, f'acc_epoch{e}.png'))
        plt.show()  
        plt.figure(figsize=(10, 5))
        plt.plot(lr_values, label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Iteration')
        plt.ylabel('LR')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'lr_schedule.png'))
        plt.close()
        ## 绘制新的 Loss 曲线，横坐标改为 Epoch
        plt.figure(figsize=(10, 5))

        total_steps = len(train_loader) * epochs  # 训练总步数
        steps = np.arange(1, len(loss_values) + 1)  # 现有 loss 对应的 step
        epochs_mapped = steps / total_steps * epochs  # 映射成 epoch 范围

        plt.plot(epochs_mapped, loss_values, label='Loss', color='y')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'New_loss_epoch.png'))
        plt.show() 
        ## 绘制miou曲线
        plt.figure(figsize=(10, 5))  # 可选：设置图表大小
        # plt.plot(MIoU_values, label='Train MIoU')  
        plt.plot(range(0, len(MIoU_values)*500, 500), MIoU_values, label='Train MIoU')
        plt.title('Train_MIoU Curve')  
        plt.xlabel('Epoch')  
        plt.ylabel('MIoU')  
        plt.grid(True)  
        plt.savefig(os.path.join(save_path, 'MIoU.png'))
        plt.show()  

        ## 按 Epoch 绘制 Loss 和 MIoU 曲线（共用 X 轴: Epoch）
        plt.figure(figsize=(10, 5))
        epochs_range = range(1, epochs+1)
        # 1. 计算每个 epoch 的平均 loss（从 step 转成 epoch）
        loss_per_epoch = [
            np.mean(loss_values[i*len(train_loader):(i+1)*len(train_loader)]) 
            for i in range(epochs)
        ]
        # 左轴：Loss
        ax1 = plt.gca()
        ax1.plot(epochs_range, loss_per_epoch, label='Loss', color='y')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='y')
        ax1.tick_params(axis='y', labelcolor='y')

        # 2. MIoU 直接按 epoch 对齐（如果不是每个 epoch 都记录，可以线性插值）
        miou_epochs = np.linspace(1, epochs, len(MIoU_values))
        ax2 = ax1.twinx() 
        ax2.plot(miou_epochs, MIoU_values, label='MIoU', color='b') 
        ax2.set_ylabel('MIoU', color='b') 
        ax2.tick_params(axis='y', labelcolor='b') 
        plt.title('Loss and MIoU Curve per Epoch') 
        plt.grid(True) 
        plt.savefig(os.path.join(save_path, 'loss_miou_epoch.png')) 
        plt.show()

        # 保存训练指标
        metrics_dir = os.path.join(save_path, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # 保存 loss_values 和 train_acc
        np.save(os.path.join(metrics_dir, "loss_values.npy"), np.array(loss_values))
        np.save(os.path.join(metrics_dir, "train_acc.npy"), np.array(train_acc))
        np.save(os.path.join(metrics_dir, "MIoU_values.npy"), np.array(MIoU_values))

        # 保存为 CSV
        csv_file = os.path.join(metrics_dir, "metrics.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step_or_epoch", "loss", "train_acc", "MIoU"])
            steps = range(1, len(loss_values) + 1)
            miou_steps = np.linspace(1, len(loss_values), len(MIoU_values))
            # 将 MIoU 对齐 step，如果长度不一样，先插值
            miou_interp = np.interp(steps, miou_steps, MIoU_values) if len(MIoU_values) > 0 else [0]*len(steps)
            for s, l, a, m in zip(steps, loss_values, train_acc + [0]*(len(loss_values)-len(train_acc)), miou_interp):
                writer.writerow([s, l, a, m])

        print("✅ 已保存 loss_values.npy, train_acc.npy, MIoU_values.npy, metrics.csv")



if MODE == 'Train':
    train(net, optimizer, 50, scheduler)
elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('./results/SG/Vaihingen/WgAnet_128x128/WgAnet_epoch42_0.8177682921740267'), strict=False) # seg
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            # plt.imshow(img) and plt.show()
            io.imsave('./results/SG/Vaihingen/WgAnet_128x128/WgAnet_epoch42_0.8177_'+MODEL+'_tile_{}.png'.format(id_), img)

    elif DATASET == 'Urban':
        net.load_state_dict(torch.load('./results/SG/Urban/TransUNet_2/TransUNet_epoch37_0.4928604187263022'), strict=False)  # seg
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            # plt.imshow(img) and plt.show()
            io.imsave('./results/SG/Urban/TransUNet_2/TransUNet_epoch37_0.4928_'+MODEL+'_tile_{}.png'.format(id_), img)

    elif DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./results/SG/Potsdam/MIFNet_24_253545_256/MIFNet_epoch30_0.85337827993709'), strict=False) # seg,加载模型 
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32) # 进行测试，并获取平均交并比、预测结果和真实标签 
        print("MIoU: ", MIoU) # 打印平均交并比 
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p) # 将预测结果转换为彩色图像 
            # plt.imshow(img) and plt.show()
            io.imsave('./results/SG/Potsdam/MIFNet_24_253545_256/MIFNet_epoch30_0.8533_'+MODEL+'_tile_{}.png'.format(id_), img) # 保存测试结果的图像
