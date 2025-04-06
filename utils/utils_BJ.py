import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
#from metrics1 import evaluate_performance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def RMSE(Y_real, Y_pre):  # 计算RMSE
#     from sklearn.metrics import mean_squared_error
#     return np.sqrt(mean_squared_error(Y_real, Y_pre))  # Y_real为实际值，Y_pre为预测值
#
#
# def MAPE(Y_real, Y_pre):  # 计算mape
#     from sklearn.metrics import mean_absolute_percentage_error
#     return mean_absolute_percentage_error(Y_real, Y_pre)  # Y_real为实际值，Y_pre为预测值

def masked_mape_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

# 1log string
def log_string(log, string):

    log.write(string + '\n')
    log.flush()
    print(string)

# def metric(pred, label):
#     # print(pred)
#     # print(pred.shape)
#     RMSE_Tr, MAE_Tr, masked_MAPE_Tr= evaluate_performance(pred, label)
#     return MAE_Tr, RMSE_Tr, masked_MAPE_Tr

# metric
def metric(pred, label):
    mask = torch.ne(label, 0)   # 判断两个数组的元素是否不相等，若对应位置上的元素不相等，则该位置上的元素是True，否则是False
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)
    rmse = mae ** 2
    # mape = mae / label
    mae = torch.mean(mae)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.mean(rmse))
    # mape = mape * mask
    # mape = torch.mean(mape)
    mape = masked_mape_torch(pred, label, 0)
    smape = 2.0 * torch.mean(torch.abs(pred - label) / (torch.abs(pred) + torch.abs(label)))
    return mae, rmse, mape



# num_his = 12   num_pred =3
def seq2instance(data, num_his, num_pred):            # 把数据切分成X（12）和y（3）
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y


def load_data(args):
    # Traffic
    df = pd.read_hdf(args.traffic_file)                  # './data/10min_data.h5'
    traffic = torch.from_numpy(df.values).to(device)     # [17296,44]提取客流  [1800,275]
    # weather
    wea_df = pd.read_csv(args.weather_file, header=None)
    wea_df = wea_df.T                                       # [1800,10]
    # wea_df.drop('timestamp', axis=1, inplace=True)       # Drop函数删除Dataframe指定行列
    wea_data = torch.from_numpy(wea_df.values).to(device)# 生成数据
    print(wea_data.shape)                                # [17296,5] [1800,10]

    # adj
    # adj = pd.read_csv(args.adj, header=None)
    # adj = torch.from_numpy(adj.values).to(device)        # 邻接矩阵[44,44]
    # adj = adj.float()


    num_step = df.shape[0]                               # 17296 1800

    train_steps = round(args.train_ratio * num_step)     # 12107
    test_steps = round(args.test_ratio * num_step)       # 3459
    val_steps = num_step - train_steps - test_steps      # 1730
    # 分割数据集
    train = traffic[: train_steps].to(device)            # 切分训练集、验证集、测试集
    val = traffic[train_steps: train_steps + val_steps].to(device)
    test = traffic[-test_steps:].to(device)
    wea_train = wea_data[:train_steps].to(device)        # [12107,5]
    wea_val = wea_data[train_steps: train_steps + val_steps].to(device)  # [1730,5]
    wea_test = wea_data[-test_steps:].to(device)         # [3459,5]
    # print(wea_train.shape)
    # print("train,val,test", train.is_cuda)
    # X, Y
    # 训练集
    trainX, trainY = seq2instance(train, args.num_his, args.num_pred)   # 把数据集再切分成历史客流X和未来客流Y
    wea_trainX, wea_trainY = seq2instance(wea_train, args.num_his, args.num_pred)   # wea_trainX[12093,12,11] , wea_trainY[12093,3,11]
    trainX = trainX.to(device)                      # [12093,12,44]
    trainY = trainY.to(device)                      # [12093,3,44]
    wea_trainX = wea_trainX.to(device)
    wea_trainY =  wea_trainY.to(device)
    # print("trainX", trainX.is_cuda)
    # 验证集
    valX, valY = seq2instance(val, args.num_his, args.num_pred)
    val_weaX, val_weaY = seq2instance(wea_val, args.num_his, args.num_pred)     # val_weaX[1716,12,11] , val_weaY[1716,3,11]
    valX = valX.to(device)                          # [1716,12,44]
    valY = valY.to(device)                          # [1716,3,44]
    val_weaX =  val_weaX.to(device)
    val_weaY = val_weaY.to(device)
    # 测试集
    testX, testY = seq2instance(test, args.num_his, args.num_pred)
    test_weaX, test_weaY = seq2instance(wea_test, args.num_his, args.num_pred)    # test_weaX[3445,12,11] , test_weaY[3445,3,11]
    testX = testX.to(device)                        # [3445,12,44]
    testY = testY.to(device)                        # [3445,3,44]
    test_weaX = test_weaX.to(device)
    test_weaY = test_weaY.to(device)

    # normalization
    mean, std = torch.mean(trainX).cpu(), torch.std(trainX).cpu()           # 计算均值方差
    #mean 均值 std 标准差
    trainX = ((trainX - mean) / std).to(device)                             # 数据集归一化
    valX = ((valX - mean) / std).to(device)
    testX = ((testX - mean) / std).to(device)

    # spatial embedding                                                     # 生成SE
    with open(args.SE_file, mode='r') as f:                                 # './data/SE(BJ276).txt 只读
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32).to(device)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    # print("SE",SE.is_cuda)
    # temporal embedding
    time = pd.DatetimeIndex(df.index)                                     # pd.DatetimeIndex()直接生成时间戳索引

    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))        #  dayofweek[17296,1]生成一周内今天是哪一天的列表

    # delta = datetime(2012, 6, 28, 0, 0, 0) - datetime(2012, 3, 1, 0, 0, 0)
    timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
                // 900.0  # delta.total_seconds()

    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))           # [17296,1] 生成一天中数据在那个时刻的列表

    time = torch.cat((dayofweek, timeofday), -1).to(device)               # [17296,2]把dayofweek, timeofday拼接
    # train/val/test
    train = time[: train_steps].to(device)                                # 划分比例
    val = time[train_steps: train_steps + val_steps].to(device)
    test = time[-test_steps:].to(device)
    # shape = (num_sample, num_his + num_pred, 2)
    trainTE = seq2instance(train, args.num_his, args.num_pred)        # 每一部分数据转成序列（15=12+3）
    trainTE = torch.cat(trainTE, 1).type(torch.int32).to(device)         # [12093,15,2]
    # print("trainTE",trainTE.is_cuda)
    valTE = seq2instance(val, args.num_his, args.num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32).to(device)             # [1716,15,2]

    testTE = seq2instance(test, args.num_his, args.num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32).to(device)           # [3445,15,2]

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std, wea_trainX, wea_trainY, val_weaX, val_weaY, test_weaX, test_weaY)


# dataset creation
class dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.len = data_x.shape[0]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len


# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# The following function can be replaced by 'loss = torch.nn.L1Loss()  loss_out = loss(pred, target)
def mae_loss(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.tensor(0.0), mask)
    loss = torch.abs(torch.sub(pred, label))
    loss *= mask
    loss = torch.where(torch.isnan(loss), torch.tensor(0.0), loss)
    loss = torch.mean(loss)
    return loss


# plot train_val_loss
def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)


# plot test results
def save_test_result(trainPred, trainY, valPred, valY, testPred, testY):
    with open('./figure/test_results.txt', 'w+') as f:
        for l in (trainPred, trainY, valPred, valY, testPred, testY):
            f.write(list(l))
