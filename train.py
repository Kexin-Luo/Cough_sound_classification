import torch
import torchvision
import torchvision.models as mm
import torch.nn as nn
import numpy as np
import json
import time
import pingjia
import utils
import validate
import argparse
import models.resnet
import models.inception
import models.myCNN
import models.ResneXt
import models.Mel_CNN
import models.densenet
import models.SqueezeNet
import models.lstm
import time
import dataloaders.datasetnormal
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import sklearn.metrics as me
import  pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import f1_score, precision_score, recall_score
parser = argparse.ArgumentParser()
# parser.add_argument("--config_path", type=str)

model_path= "model/mel_lstm15.3_3.pkl"
# pkl_path=os.path.join('trained_model/inception_p.pkl')
test_dir="dataset/csc4/meta"
# test_path=os.path.join('dataset/csc4/meta/CSC4_val.csv')
# # test_path=os.path.join('dataset/esc/meta/esc50.csv')
# test_path_data=pd.read_csv(test_path)
# target=list(test_path_data.loc[:,'target'])
# target=np.array(target)
# print(target)


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):
            inputs = data[0].to(device)
            target = data[1].squeeze(1).to(device)
#             print(inputs.shape)
            outputs = model(inputs)

            loss = loss_fn(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg()

def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, split,
                       scheduler=None):
    best_acc = 0.0
    #     torch.backends.cudnn.enable =True
    #     torch.backends.cudnn.benchmark = True
    for epoch in range(params.epochs):
        since = time.time()
        avg_loss = train(model, device, train_loader, optimizer, loss_fn)

        time_elapsed = time.time() - since
        #         print(time_elapsed)
        print('Training complete in  {}s'.format(time_elapsed))
        acc_time = time.time()
        acc = validate.evaluate(model, device, val_loader)
        acc_time_end = time.time() - acc_time
        print('val complete in  {}s'.format(acc_time_end))
        #         print('val complete in {:.0f}m {:.0f}s'.format(acc_time_end // 60, acc_time_end % 60))
        print("Epoch {}/{} Loss:{} Valid Acc:{}".format(epoch, params.epochs, avg_loss, acc))

        is_best = (acc > best_acc)
        if is_best:
            best_acc = acc
        if scheduler:
            scheduler.step()

        utils.save_checkpoint({"epoch": epoch + 1,
                               "model": model.state_dict(),
                               "optimizer": optimizer.state_dict()}, is_best, split, "{}".format(params.checkpoint_dir))
        writer.add_scalar("data{}/trainingLoss{}".format(params.dataset_name, split), avg_loss, epoch)
        writer.add_scalar("data{}/acc{}".format(params.dataset_name, split), acc, epoch)
    writer.close()
    torch.save(model, model_path)

cnn_config_dir = "config/csc_CNNnet.json"
resnet_config_dir = "config/csc_resnet.json"
inception_config_dir = "config/csc_inception.json"
resnext_config = "config/csc_resneXt.json"
mel_cnn_config_dir = "config/csc_MEL_CNN.json"
densenet_dir="config/csc_densenet.json"
squeezenet_dir="config/squeezenet.json"

if __name__ == "__main__":
    args = parser.parse_args(args=[])
    params = utils.Params(densenet_dir)
#    params = utils.Params(config_dir)
    # print(params.num_workers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(1, params.num_folds + 1):
        if params.dataaug:
            pass
        #             train_loader = dataloaders.datasetaug.fetch_dataloader( "{}training128mel{}.pkl".format(params.data_dir，i), params.dataset_name, params.batch_size, params.num_workers, 'train')
        #             val_loader = dataloaders.datasetaug.fetch_dataloader("{}validation128mel{}.pkl".format(params.data_dir,i), params.dataset_name, params.batch_size, params.num_workers, 'validation')
        #             test_loader = dataloaders.datasetaug.fetch_dataloader("{}validation128mel{}.pkl".format(params.data_dir,i),
        #                                                                  params.dataset_name, params.batch_size,
        #                                                                  params.num_workers, 'validation')
        else:
            train_loader = dataloaders.datasetnormal.fetch_dataloader(
                "{}training128mel{}.pkl".format(params.data_dir, i), params.dataset_name, params.batch_size,
                params.num_workers)
            val_loader = dataloaders.datasetnormal.fetch_dataloader(
                "{}validation128mel{}.pkl".format(params.data_dir, i), params.dataset_name, params.batch_size,
                params.num_workers)
            test_loader = dataloaders.datasetnormal.fetch_dataloader(
                "{}validation128mel{}.pkl".format(params.data_dir, i), params.dataset_name, 80, params.num_workers, )
        writer = SummaryWriter(comment=params.dataset_name)
        if params.model == "CNN":
            model = models.myCNN.Cnn_v1().to(device)
        elif params.model == "resnet":
            model = models.resnet.eca_resnet50().to(device)
        elif params.model == "inception":
            model = models.inception.Inception(params.pretrained).to(device)
        elif params.model == "densenet":
#             model=models.densenet.DenseNet().to(device)
            model=models.lstm.RNN().to(device)
        elif params.model == "resneXt":
            model=models.ResneXt.resnext50_32x4d().to(device)
        elif params.model == "Mel_CNN":
            model=models.Mel_CNN.Mel_CNN().to(device) 
        elif params.model == "squeezenet":
            model=models.SqueezeNet.Shuffle_v1(4).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

        if params.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
        else:
            scheduler = None

        train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, i, scheduler)
        test_name = "CSC4_{}.csv".format(i)
        test_path = os.path.join(test_dir, test_name)
        test_path_data = pd.read_csv(test_path)
        x_test = []
        y_test = []
        pred = []
        predict = []
        for batch_idx, data in enumerate(test_loader):
            
            x_test = data[0].to(device)
            target = data[1].squeeze(1).to(device).cpu()
            y_test.append(target)
            model = torch.load(model_path)
            model.eval()
            test_time = time.time()
            if pred == []:
                pred = model(x_test)
                # print(pred.shape)

            else:
                pred += model(x_test)
                # print(pred.shape)

                # predict=predict.data.numpy()
                # print(predict)
            time_elapsed = time.time() - test_time
            #         print(time_elapsed)
            print('test complete in  {}s'.format(time_elapsed))
        pred = pred.cpu().detach().numpy()
        #         print(pred)
        #         min_max_scaler=preprocessing.MinMaxScaler()
        #         x_minmax = min_max_scaler.fit_transform(x)
        #         print(np.round(x_minmax,2))
        #         print(y_test)
        #         print(len(predict))
        #         predict=pred.cpu().detach().numpy()
        res = np.argmax(pred, axis=1)
        #         print(res[:5])
        df = pd.DataFrame({"img_path": test_path_data["filename"], "tags": res})
        df.to_csv("submit.csv", index=None, header=None)
        result = me.confusion_matrix(target, res)
        print("The confusion_matrix is :")
        print(result)
        print(accuracy_score(target,res))
        print(f1_score(target,res, average='macro'))
        print(recall_score(target,res, average='macro'))
        print(precision_score(target,res, average='macro'))
        se = pingjia.sen(target, res, 4)
        sp = pingjia.spe(target, res, 4)
        acc = pingjia.ACC(target, res, 4)
        pre = pingjia.pre(target, res, 4)
        F1 = pingjia.sen_all(target, res, 4, se, pre)
#         print(np.round(se, 5))
        print(np.round(sp, 5))
#         print(np.round(acc, 5))
#         print(np.round(pre, 5))
#         print(np.round(F1, 5))






