import time
import datetime


from utils.utils_BJ import log_string, load_data
from model.model_ import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, args, log, loss_criterion, optimizer, scheduler):

    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, SE, mean, std, wea_trainX, wea_trainY, val_weaX, val_weaY, test_weaX, test_weaY) = load_data(args)

    num_train, _, num_vertex = trainX.shape              # num_train=36458 , _=12 , num_vertex=325
    log_string(log, '**** training model ****')
    num_val = valX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)           # train_num_batch=36458/32=1139.3125(1140)
    val_num_batch = math.ceil(num_val / args.batch_size)               # val_num_batch=5189/32=162.15625(163)


    wait = 0
    val_loss_min = float('inf')
    best_model_wts = None
    train_total_loss = []
    val_total_loss = []

    # Train & validation
    for epoch in range(args.max_epoch):                                # epoch =10
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        # shuffle
        permutation = torch.randperm(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        wea_trainX = wea_trainX[permutation]
        # train
        start_train = time.time()
        model.train()
        train_loss = 0
        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            X = trainX[start_idx: end_idx]
            TE = trainTE[start_idx: end_idx]
            #from main import wea_trainX
            Input = wea_trainX[start_idx: end_idx]  ## 新增：获取当前批次的天气数据作为 Input
            label = trainY[start_idx: end_idx]
            optimizer.zero_grad()
            pred = model(X, TE, Input)
            pred = pred * std + mean
            loss_batch = loss_criterion(pred, label)
            train_loss += float(loss_batch) * (end_idx - start_idx)
            loss_batch.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (batch_idx+1) % 5 == 0:
                print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}')
            del X, TE, label, pred, loss_batch
        train_loss /= num_train
        train_total_loss.append(train_loss)
        end_train = time.time()

        # val loss
        start_val = time.time()
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx in range(val_num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
                X = valX[start_idx: end_idx]
                TE = valTE[start_idx: end_idx]
                from main import val_weaX
                Input = val_weaX[start_idx: end_idx]  # 获取当前批次的验证天气数据作为 Input
                label = valY[start_idx: end_idx]
                pred = model(X, TE, Input)
                pred = pred * std + mean
                loss_batch = loss_criterion(pred, label)
                val_loss += loss_batch * (end_idx - start_idx)
                del X, TE, label, pred, loss_batch
        val_loss /= num_val
        val_total_loss.append(val_loss)
        end_val = time.time()
        log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             args.max_epoch, end_train - start_train, end_val - start_val))
        log_string(
            log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
        if val_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {args.model_file}')
            wait = 0
            val_loss_min = val_loss
            best_model_wts = model.state_dict()
        else:
            wait += 1
        scheduler.step()

    model.load_state_dict(best_model_wts)
    torch.save(model, args.model_file)
    log_string(log, f'Training and validation are completed, and model has been stored as {args.model_file}')
    return train_total_loss, val_total_loss
