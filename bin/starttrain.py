# train arguments
# python3.9 starttrain.py -c config.yaml
# import sys
# sys.path.append(r"/home/alex/GitLabProjects/NN_pipeline/")
# from bin.train import prep_model
import argparse
import train, train_nii

parser = argparse.ArgumentParser()
parser.add_argument('-c','--config', default=None,
                    help='Specify the config file in "config" folder for training. Default: "config.yaml".')
args = parser.parse_args()
configfilename = args.config
if not configfilename:
    configfilename = input("Insert config file name:")
train.prep_model(configfilename)
# train_nii.prep_model(configfilename)
# alc.acc_loss_curves(configfilename)