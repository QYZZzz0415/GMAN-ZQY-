import os

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_BJ import log_string, plot_train_val_loss
from utils.utils_BJ import count_parameters, load_data

from model.model_ import GMAN
from model.train import train
from model.test import test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()  # 创建对象
parser.add_argument('--time_slot', type=int, default=10,
                    help='a time step is 10 mins')#时间步长
parser.add_argument('--num_his', type=int, default=6,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=6,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')#每个注意力头输出向量维度
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')#数据集划分比例 train占 70%
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')#val占 10%
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')#test占 20%
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')#输入模型的数据样本数量
parser.add_argument('--max_epoch', type=int, default=200,
                    help='epoch to run')#打乱数据样本 每一个epoch把所有数据样本跑一遍
parser.add_argument('--patience', type=int, default=10,
                    help='patience for early stop')#验证集损失没有下降 第十轮停止
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='decay epoch')#学习率衰减 第十一轮开始衰减
parser.add_argument('--traffic_file', default='./data/10min_data.h5',
                    help='traffic file')
parser.add_argument('--weather_file', default='./data/10 min after normolization.csv')
parser.add_argument('--SE_file', default='./data/SE(BJ276).txt',
                    help='spatial embedding file')
parser.add_argument('--model_file', default='./data/GMAN.pkl',
                    help='save the model to disk')
parser.add_argument('--log_file', default='./data/log',
                    help='log file')
args = parser.parse_args()  # 解析对象
log = open(args.log_file, 'w') # 清空原有内容，全部重新生成
log_string(log, str(args)[10: -1])
T = 24 * 60 // args.time_slot  # Number of time steps in one day
# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
 testY, SE, mean, std, wea_trainX, wea_trainY, val_weaX, val_weaY, test_weaX, test_weaY) = load_data(args)
#  把客流和天气数据集转化成历史和预测  把向量存入空间嵌入SE 时间步TE转化成历史和预测
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
log_string(log, 'data loaded!')
del trainX, trainTE, valX, valTE, testX, testTE, mean, std
# build model
log_string(log, 'compiling model...')
model = GMAN(SE, args, bn_decay=0.1).to(device)  # spatial embedding  当前批次因素影响小

loss_criterion = nn.MSELoss()# 均方误差损失函数

optimizer = optim.Adam(model.parameters(), args.learning_rate) #优化器
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=args.decay_epoch,
                                      gamma=0.9)#动态调整学习率
parameters = count_parameters(model)#计划可训练参数总数
log_string(log, 'trainable parameters: {:,}'.format(parameters))

if __name__ == '__main__': #作为主程序导入才执行
    start = time.time()
    loss_train, loss_val = train(model, args, log, loss_criterion, optimizer, scheduler)
    # plot_train_val_loss(loss_train, loss_val, './figure/train_val_loss.png')
    trainPred, valPred, testPred = test(args, log)
    end = time.time()
    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()

    trainPred_ = trainPred.numpy().reshape(-1, trainY.shape[-1])
    trainY_ = trainY.cpu().numpy().reshape(-1, trainY.shape[-1])
    valPred_ = valPred.cpu().numpy().reshape(-1, valY.shape[-1])
    valY_ = valY.cpu().numpy().reshape(-1, valY.shape[-1])
    testPred_ = testPred.cpu().numpy().reshape(-1, testY.shape[-1])
    testY_ = testY.cpu().numpy().reshape(-1, testY.shape[-1])

    # Save training, validation and testing datas to disk
    # l = [trainPred_, trainY_, valPred_, valY_, testPred_, testY_]
    # name = ['trainPred', 'trainY', 'valPred', 'valY', 'testPred', 'testY']
    # for i, data in enumerate(l):
    #     np.savetxt('./figure/' + name[i] + '.txt', data, fmt='%s')

    # Plot the test prediction vs target（optional)
    # plt.figure(figsize=(10, 280))
    # for k in range(325):
    #     plt.subplot(325, 1, k + 1)
    #     for j in range(len(testPred)):
    #         c, d = [], []
    #         for i in range(12):
    #             c.append(testPred[j, i, k])
    #             d.append(testY[j, i, k])
    #         plt.plot(range(1 + j, 12 + 1 + j), c, c='b')
    #         plt.plot(range(1 + j, 12 + 1 + j), d, c='r')
    # plt.title('Test prediction vs Target')
    # plt.savefig('./figure/test_results.png')



