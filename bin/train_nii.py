# python3.9 train.py -c config.yaml

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import shutil
import torch
import yaml
# import random
import os
import sys
# sys.path.append(r"/home/alex/GitLabProjects/NN-pipeline/")
# sys.path.append(r".../")
sys.path.append(r"../")
# sys.path.append(r"./")

from super_pipeline.main import lib, losses, optimizers, schedulers, models
from super_pipeline.dataload import dataloader_nii as dtl
# from super_pipeline.visual import acc_loss_curves as alc
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:1"


def loadconfig(configfilename):
    folder2savexperiment = os.path.join(
        "../results/", configfilename.replace('.yaml', ''))
    if not os.path.exists(folder2savexperiment):
        print(
            f"Folder path for result is missing. Creating one: {folder2savexperiment}")
        os.mkdir(folder2savexperiment)
    else:
        print(
            f"Folder path {folder2savexperiment} for result exists. Training continues.")

    ymconfig = lib.open_config_file(
        os.path.join('../configs/', configfilename))
    ymdatafn = ymconfig["dataset_conf"]

    ymargs = lib.open_config_file(ymconfig["args"])
    ymargs["folder2savexperiment"] = folder2savexperiment

    ymconfig["class_names"] = lib.check_classes(ymdatafn)
    with open(os.path.join('../configs/', configfilename), 'w') as f:
        yaml.dump(ymconfig, f)

    shutil.copy(ymconfig["args"], os.path.join(
        folder2savexperiment, ymconfig["args"]))
    shutil.copy(os.path.join('../configs/', configfilename),
                os.path.join(folder2savexperiment, configfilename))

    seed = ymconfig["seed"]
    lib.rseed(seed)

    ymdataset = lib.open_config_file('../configs/data/'+ymdatafn)
    par = [ymdataset["data_path"], ymdataset["train_dir"], ymargs['batch_size']]
    train_dataloader = dtl.dataloader3D(*par).dataloader()
    par = [ymdataset["data_path"], ymdataset["val_dir"], ymargs['batch_size']]
    val_dataloader = dtl.dataloader3D(*par).dataloader()

    return ymargs, ymconfig, train_dataloader, val_dataloader

# train_model


def train_model(model, train_dataloader, val_dataloader, loss, optimizer, scheduler, lossdf, ymargs):
    epoch_max = len(lossdf)
    bals = lossdf["acces_val"] - lossdf["losses_val"]
    bal = bals.max()
    balmi = bals.argmax()
    acc_loss_min = bal
    loss_best = lossdf.loc[balmi, "losses_val"]
    loss_min = lossdf.loc[balmi, "losses_val"]
    acc_max = lossdf.loc[balmi, "acces_val"]
    # best_model_state = copy.deepcopy(model.state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in range(epoch_max, ymargs["num_epochs"]):
        if epoch >= (epoch_max+ymargs["step"]):
            #             print('Best Epoch {}, loss {:.4f}, acc: {:.4f}:'.format(epoch_max, acc_loss_min, acc_max), flush=True)
            break
        print('Start epoch {}:'.format(epoch), flush=True, end=" ")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
                print("train")
            else:
                dataloader = val_dataloader
                model.eval()   # Set model to evaluate mode
                print("eval")

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            for inputs, labels in dataloader:  # tqdm(
                #                 for x_item, y_item in zip(inputs, labels):
                #                     show_input(x_item, title=class_names[y_item])
                print("3 : ", inputs.shape)
                print("4 : ", labels.shape)
                inputs = inputs.float().to(device)
                labels = labels.reshape(
                    inputs.shape[0], inputs.shape[2], inputs.shape[3]).float().to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    print("preds : ", preds.shape)
                    preds = preds.reshape_as(labels)
                    loss_value = loss(preds, labels)
                    preds_class = torch.trunc(preds*2.9999999)/2

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item()
                running_acc += (torch.eq(preds_class,
                                labels)).float().mean().cpu()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc.item() / len(dataloader)
            print('epoch_loss:', epoch_loss)
            print('epoch_acc:', epoch_acc)

            if phase == 'train':
                scheduler.step()
                lossdf.loc[epoch, "losses_train"] = epoch_loss
                lossdf.loc[epoch, "acces_train"] = epoch_acc
            else:
                lossdf.loc[epoch, "losses_val"] = epoch_loss
                lossdf.loc[epoch, "acces_val"] = epoch_acc
                # if acc_loss_min>epoch_loss:
                # (epoch_acc-epoch_loss)>acc_loss_min loss_best>epoch_loss
                if (epoch_acc-epoch_loss) > acc_loss_min:
                    acc_loss_min = epoch_acc-epoch_loss
                    epoch_max = epoch
                    acc_max = epoch_acc
                    loss_best = epoch_loss
                    # best_model_state = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(
                        ymargs["folder2savexperiment"], ymargs["savnambestmod"]))
                    print('Best ass-loss Epoch,', end=' ')
                    # print('Best ass-loss Epoch {}, loss {:.4f},  acc: {:.4f}:'.format(
                    # epoch_max, loss_best, acc_max))
                # (epoch_acc-epoch_loss)>acc_loss_min loss_best>epoch_loss
                if epoch_loss < loss_min:
                    loss_min = epoch_loss
                    # loss_epoch_max = epoch
                    print('Best loss Epoch,', end=' ')
                    # print('   Best loss Epoch {}, loss {:.4f},  acc: {:.4f}:'.format(
                        # loss_epoch_max, loss_min, epoch_acc))
                    # model_max = model

        torch.save(model.state_dict(), os.path.join(
            ymargs["folder2savexperiment"], ymargs["savnam"]))
        lossdf.to_csv(os.path.join(
            ymargs["folder2savexperiment"], ymargs["losscsv"]), index=False)
        print('epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(
            epoch, epoch_loss, epoch_acc), flush=True)

    print('Best epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_max, loss_best, acc_max), flush=True)
    # model.load_state_dict(best_model_state)
    # return model, lossdf


def prep_model(configfilename):

    ymargs, ymconfig, train_dataloader, val_dataloader = loadconfig(
        configfilename)
    savnam = os.path.join(ymargs["folder2savexperiment"], ymargs["savnam"])
    losscsv = os.path.join(ymargs["folder2savexperiment"], ymargs["losscsv"])

    fmodel = getattr(models, ymconfig["model_conf"])
    model = fmodel(ymconfig["model_param"])

    for param in model.parameters():
        param.requires_grad = ymconfig["paramrequiresgrad"]

    floss = getattr(losses, ymconfig["loss_conf"])
    loss = floss(ymconfig["loss_param"])

    foptimizer = getattr(optimizers, ymconfig["optimizer_conf"])
    optimizer = foptimizer(model.parameters(), ymconfig["optimizer_param"])

    fscheduler = getattr(schedulers, ymconfig["scheduler_conf"])
    scheduler = fscheduler(optimizer, ymconfig["scheduler_param"])
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # print(f'Model # {dir_num}')
    # resnext50_32x4d efficientnet_b3 regnet_x_16gf regnet_x_1_6gf regnet_x_32gf regnet_x_800mf resnet18
    if os.path.exists(savnam):
        m_state_dict = torch.load(savnam)
        model.load_state_dict(m_state_dict)
        print(f'Model downloaded')
    if os.path.exists(losscsv):
        lossdf = pd.read_csv(losscsv)
        print(f'losslist downloaded')
    else:
        lossdf = pd.DataFrame(np.array([[1, 1, 0, 0]]),
                              columns=["losses_train", "losses_val", "acces_train", "acces_val"])
    train_model(model, train_dataloader, val_dataloader, loss, optimizer, scheduler,
                lossdf, ymargs)


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', default="config.yaml",
#                         help='Specify the config file in "config" folder for training. Default: "config.yaml".')
#     args = parser.parse_args()
#     configfilename = args.config
#     prep_model(configfilename)
#     alc.acc_loss_curves(configfilename)
