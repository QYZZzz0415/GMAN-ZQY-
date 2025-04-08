import torch
import time
import math
import numpy as np
from utils.utils_BJ import log_string, load_data, metric, device


def test(args, log):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, SE, mean, std, wea_trainX, wea_trainY, val_weaX, val_weaY, test_weaX, test_weaY) = load_data(args) ##
    num_train, _, num_vertex = trainX.shape
    num_val = valX.shape[0]
    num_test = testX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)
    test_num_batch = math.ceil(num_test / args.batch_size)

    model = torch.load(args.model_file,weights_only=False)

    # test model
    log_string(log, '**** testing model ****')
    log_string(log, 'loading model from %s' % args.model_file)
    model = torch.load(args.model_file,weights_only=False)
    log_string(log, 'model restored!')
    log_string(log, 'evaluating...')

    with torch.no_grad():

        trainPred = []
        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            if end_idx <=num_train:
                X = trainX[start_idx: end_idx]
                TE = trainTE[start_idx: end_idx]
                Input = wea_trainX[start_idx: end_idx].to(device)  ## 获取当前批次的测试天气数据作为 Input
            else:
                break
            pred_batch = model(X, TE, Input)
            trainPred.append(pred_batch.cpu().detach().clone())
            del X, TE, pred_batch ,Input  ##
        trainPred = torch.from_numpy(np.concatenate(trainPred, axis=0))
        std.to('cpu')
        mean.to('cpu')
        trainPred = trainPred * std + mean

        valPred = []
        for batch_idx in range(val_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
            if end_idx <= num_val:
                X = valX[start_idx: end_idx]
                TE = valTE[start_idx: end_idx]
                Input = val_weaX[start_idx: end_idx].to(device)## 获取当前批次的测试天气数据作为 Input
            else:
                break
            pred_batch = model(X, TE, Input)
            valPred.append(pred_batch.cpu().detach().clone())
            del X, TE, pred_batch ,Input
        valPred = torch.from_numpy(np.concatenate(valPred, axis=0))
        valPred = valPred * std + mean
        valY = valY[0:valPred.shape[0]]

        testPred = []
        start_test = time.time()
        for batch_idx in range(test_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
            if end_idx <= num_test:
                X = testX[start_idx: end_idx]
                TE = testTE[start_idx: end_idx]
                Input = test_weaX[start_idx: end_idx].to(device)  ## 获取当前批次的测试天气数据作为 Input
            else:
                break
            pred_batch = model(X, TE, Input) ##
            testPred.append(pred_batch.cpu().detach().clone())
            del X, TE, pred_batch ,Input
        testPred = torch.from_numpy(np.concatenate(testPred, axis=0))
        testPred = testPred* std + mean
        testY = testY[0:testPred.shape[0]]
    end_test = time.time()
    train_mae, train_rmse, train_mape = metric(trainPred, trainY.cpu())
    val_mae, val_rmse, val_mape = metric(valPred, valY.cpu())
    test_mae, test_rmse, test_mape = metric(testPred, testY.cpu())
    log_string(log, 'testing time: %.1fs' % (end_test - start_test))
    log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
               (train_mae, train_rmse, train_mape * 100))
    log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
               (val_mae, val_rmse, val_mape * 100))
    log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
               (test_mae, test_rmse, test_mape * 100))
    log_string(log, 'performance in each prediction step')
    MAE, RMSE, MAPE = [], [], []
    for step in range(args.num_pred):
        mae, rmse, mape = metric(testPred[:, step].cpu(), testY[:, step].cpu())
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                   (step + 1, mae, rmse, mape * 100))
    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    average_mape = np.mean(MAPE)
    log_string(
        log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
             (average_mae, average_rmse, average_mape * 100))
    return trainPred, valPred, testPred
