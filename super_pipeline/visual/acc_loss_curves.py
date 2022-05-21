import sys
sys.path.append(r"/home/alex/GitLabProjects/NN_pipeline/")
# sys.path.append(r"..")

from super_pipeline.main import lib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
matplotlib.use("TkCairo", force=True)

def acc_loss_curves(configfilename):
    projdir = "/home/alex/GitLabProjects/NN_pipeline/"
    folder2savexperiment = os.path.join(projdir, "results", configfilename.replace('.yaml', '/'))
    if not os.path.exists(folder2savexperiment):
        print(f"Folder {folder2savexperiment} not exist")
        return

    ds = lib.open_config_file(os.path.join(projdir, 'configs', configfilename))
    ymargs = lib.open_config_file(
        os.path.join(folder2savexperiment,  ds["args"]))

    losscsv = os.path.join(folder2savexperiment, ymargs['losscsv'])
    
    if os.path.exists(losscsv):
        lossdf = pd.read_csv(losscsv)
        print(f'File {losscsv} downloaded')
        print(f'losses_val={min(lossdf["losses_val"])}')
        x = [i for i in range(len(lossdf))]
        line_lt, = plt.plot(x, lossdf["losses_train"], '--g', label='train loss')
        line_at, = plt.plot(x, lossdf["acces_train"], '-g', label='train accuracy')
        line_lv, = plt.plot(x, lossdf["losses_val"], '--r', label='val loss')
        line_av, = plt.plot(x, lossdf["acces_val"], '-r', label='val accuracy')
        plt.legend(handles=[line_lt, line_at, line_lv, line_av])
        plt.title("Trainig")
        plt.xlabel("Number of epochs")
        plt.ylabel("Accuracy, loss");
        plt.show()
    else:
        print(f"File {losscsv} not exist")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="config.yaml",
                        help='Specify the config file in "../results/config/" folder. Default: "config.yaml".')
    args = parser.parse_args()
    configfilename = args.config
    acc_loss_curves(configfilename)
