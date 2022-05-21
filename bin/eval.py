# sys.path.append(r"/home/alex/GitLabProjects/NN_pipeline/")
# invoking: python3 eval.py -c <configfilename> -b <"best" or "last">
import os
import sys
sys.path.append(r"../")


from sklearn.metrics import multilabel_confusion_matrix,  classification_report, confusion_matrix
import torch
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix
from super_pipeline.dataload import augmentation as aug
from super_pipeline.main import lib, models
# import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default="config.yaml",
                    help='Specify the config file in "config" folder for training. Default: "config.yaml".')
parser.add_argument('-b', '--bestepoch', default="best",
                    help="""Specify the best model: 
                    "best" - best difference between accuracy and loss,
                    "last" - model on last epoch. 
                    Default: "best".""")
args = parser.parse_args()
configfilename = args.config
besttype = args.bestepoch
folder2savexperiment = os.path.join(
    "../results/", configfilename.replace('.yaml', ''))

test_dir, ymargs, ymconfig, test_dataloader = lib.testloadconfig(
    configfilename)

fmodel = getattr(models, ymconfig["model_conf"])
model = fmodel(len(ymconfig["class_names"]), ymconfig["model_param"])

if besttype == "best":
    savnam = os.path.join(folder2savexperiment, ymargs["savnambestmod"])
elif besttype == "last":
    savnam = os.path.join(folder2savexperiment, ymargs["savnam"])

if os.path.exists(savnam):
    m_state_dict = torch.load(savnam)
    model.load_state_dict(m_state_dict)
    print(f'Model {savnam} downloaded')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()

test_predictions = []
test_img_paths = []
print("Evaluation starts ...")

for inputs, labels, paths in tqdm(test_dataloader):
    # print(inputs.shape, '\n', labels, '\n', paths)
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.set_grad_enabled(False):
        preds = model(inputs)
    test_predictions.append(
        torch.nn.functional.softmax(preds, dim=1).data.cpu())
    test_img_paths.extend(paths)

test_predictions = torch.vstack(test_predictions)
test_predictions0 = torch.argmax(test_predictions, dim=1).numpy()

submission_df = pd.DataFrame.from_dict(
    {'file_paths': test_img_paths, 'class_pred': test_predictions0})
submission_df['class_true'] = submission_df['file_paths'].str.replace(
    test_dir+'/', '')
submission_df['filenam'] = submission_df['class_true'].apply(
    lambda x: x.split('/')[1])
submission_df['class_true'] = submission_df['class_true'].apply(
    lambda x: x.split('/')[0])
submission_df['pred'] = submission_df['class_pred'].apply(
    lambda x: ymconfig["class_names"][x])
# print(submission_df)
submission_df.to_csv(os.path.join(folder2savexperiment,"result.csv"))

y_true = submission_df['class_true']
y_pred = submission_df['pred']

mcm = multilabel_confusion_matrix(
    y_true, y_pred, sample_weight=None, labels=None, samplewise=False)

metrics = pd.DataFrame(
    columns=['sensitivity', 'spesifity', 'precision', 'accuracy', 'f1', 'samples'])
for i, clnam in enumerate(ymconfig["class_names"]):
    # print(clnam)
    # print("Number of samples:", len(
    #     submission_df[submission_df['class_true'] == clnam]))
    TP = mcm[i][1][1]
    FP = mcm[i][0][1]
    FN = mcm[i][1][0]
    TN = mcm[i][0][0]
    sensitivity = TP / (FN + TP)
    spesifity = TN / (TN + FP)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * TP / (2 * TP + FP + FN)
    samples = len(submission_df[submission_df['class_true'] == clnam])
    metrics.loc[clnam, :] = np.array(
        [sensitivity, spesifity, precision, accuracy, f1, int(samples)])

print(metrics)
metrics.to_csv(os.path.join(folder2savexperiment, "metrics_values.csv"))
