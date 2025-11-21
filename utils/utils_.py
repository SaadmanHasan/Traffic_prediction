import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# metric
def metric(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)
    rmse = mae ** 2
    mape = mae / label
    mae = torch.mean(mae)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.mean(rmse))
    mape = mape * mask
    mape = torch.mean(mape)
    return mae, rmse, mape


def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y


def load_data(args):

    transform = [
        14, 99, 100, 113, 122, 5, 6, 188, 198, 267, 79, 111,
        181, 266, 321, 256, 258, 276, 285, 288, 133, 135, 157, 211,
        244, 2, 3, 144, 312, 323, 209, 262, 298, 299, 311, 149,
        167, 201, 204, 222, 163, 271, 272, 315, 318, 49, 143, 171,
        213, 237, 8, 9, 10, 231, 300, 86, 95, 96, 152, 161,
        23, 24, 25, 26, 27, 234, 241, 243, 260, 269, 64, 69,
        80, 112, 154, 92, 109, 134, 159, 199, 32, 74, 178, 189,
        250, 125, 150, 166, 205, 242, 84, 107, 128, 187, 223, 52,
        55, 58, 70, 132, 206, 212, 220, 229, 235, 18, 28, 277,
        290, 301, 155, 207, 221, 225, 230, 47, 56, 60, 126, 138,
        20, 22, 251, 254, 279, 180, 186, 194, 196, 197, 224, 253,
        273, 297, 305, 151, 200, 307, 308, 313, 215, 216, 232, 239,
        248, 162, 233, 236, 245, 249, 53, 90, 176, 202, 226, 214,
        246, 247, 252, 286, 15, 140, 169, 177, 179, 195, 289, 292,
        316, 320, 29, 33, 37, 81, 131, 283, 291, 293, 304, 310,
        45, 145, 192, 264, 270, 105, 110, 127, 172, 182, 153, 168,
        278, 294, 303, 103, 118, 137, 147, 165, 63, 88, 89, 98,
        117, 39, 77, 116, 295, 302, 4, 7, 219, 228, 322, 0,
        183, 208, 218, 261, 17, 30, 34, 35, 238, 12, 13, 40,
        158, 210, 62, 78, 130, 164, 185, 36, 66, 85, 91, 101,
        1, 203, 306, 309, 317, 75, 102, 104, 106, 156, 46, 57,
        61, 68, 83, 31, 41, 44, 93, 136, 284, 287, 296, 314,
        319, 173, 255, 257, 274, 275, 82, 94, 123, 146, 174, 38,
        42, 43, 71, 160, 67, 129, 139, 170, 190, 51, 54, 114,
        121, 263, 76, 148, 227, 282, 324, 108, 119, 120, 124,
        217, 11, 16, 59, 73, 87, 19, 21, 115, 142, 191, 259,
        265, 268, 280, 281, 65, 97, 141, 184, 240, 48, 50, 72,
        175, 193
    ]

    # Traffic
    df = pd.read_hdf(args.traffic_file)
    traffic = torch.from_numpy(df.values)
    
    indices = torch.tensor(transform, dtype=torch.long)
    valid_indices = indices[indices < 325]
    traffic = traffic[:, valid_indices] 

    # train/val/test
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    # train_steps = 350
    # test_steps = 100
    # val_steps = 50

    train = traffic[: train_steps]
    val = traffic[train_steps: train_steps + val_steps]
    test = traffic[-test_steps:]
    # X, Y
    trainX, trainY = seq2instance(train, args.num_his, args.num_pred)
    valX, valY = seq2instance(val, args.num_his, args.num_pred)
    testX, testY = seq2instance(test, args.num_his, args.num_pred)
    # normalization
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std
    # The order of the nodes in X is the from 0 to 324 acc to the indices in SE(PeMS)

    # spatial embedding
    with open(args.SE_file, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
        SE = SE[valid_indices] 
    # SE changes the order of the nodes based on the first col in SE(PeMS)

    # temporal embedding
    time = pd.DatetimeIndex(df.index)
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
    timeofday = (time.hour * 3600 + time.minute * 60 + time.second) // 300
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    time = torch.cat((dayofweek, timeofday), -1)
    # train/val/test
    train = time[: train_steps]
    val = time[train_steps: train_steps + val_steps]
    test = time[-test_steps:]
    # shape = (num_sample, num_his + num_pred, 2)
    trainTE = seq2instance(train, args.num_his, args.num_pred)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val, args.num_his, args.num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test, args.num_his, args.num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32)

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)


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
def plot_train_val_loss(train_total_loss, val_total_loss, train_file_path, val_file_path, txt_file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.legend(loc='best')
    plt.title('Train loss')
    plt.savefig(train_file_path)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Validation loss')
    plt.savefig(val_file_path)

    # Step 2: Open a file in write mode
    with open(txt_file_path, 'w') as file:
        # Step 3: Write the first array to the file in one line
        file.write('Train Total Loss: ' + ', '.join(f'{num:.4f}' for num in train_total_loss)+ '\n')
        
        # Step 4: Write the second array to the file in one line
        file.write('Val Total Loss: ' + ', '.join(f'{num:.4f}' for num in val_total_loss) + '\n')

    # The file is automatically closed after the with block


# plot test results / NOT USED 
def save_test_result(trainPred, trainY, valPred, valY, testPred, testY):
    with open('./STE_results/test_results.txt', 'w+') as f:
        for l in (trainPred, trainY, valPred, valY, testPred, testY):
            f.write(list(l))
