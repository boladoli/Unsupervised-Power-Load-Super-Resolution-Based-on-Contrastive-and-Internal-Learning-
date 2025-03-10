import matplotlib.pyplot as plt
from model import BlindSR
from torch.utils.data import DataLoader
import torch
from dataset import SR_fix_H5
import numpy as np
from ult import calcRMSE, fce_metric, Average_, conventional_interp
from sklearn.metrics import mean_absolute_error
from model_conventional import SRPNSE, SuperResolutionNet, DASR1D_SingleInput
import time
import h5py

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'



def count_data_in_hdf5(hdf5_path):
    with h5py.File(hdf5_path, 'r') as h5f:
        train_count = len(list(h5f['train'].keys()))
        val_count = len(list(h5f['val'].keys()))
        test_count = len(list(h5f['test'].keys()))

        print(f"Training data count: {train_count}")
        print(f"Validation data count: {val_count}")
        print(f"Testing data count: {test_count}")



def reverse_normalization(normalized_data, mean=0.8214132785797119, std=1.155210256576538):
    """
    Reverse the normalization to recover the original data.

    Parameters:
    - normalized_data: array-like, the data after normalization
    - mean: float, the mean used during normalization
    - std: float, the standard deviation used during normalization

    Returns:
    - original_data: numpy array, the data reverted back to its original scale
    """
    normalized_data = np.array(normalized_data)
    return normalized_data * std + mean


def compute_metics_interpolation(Scale, Input_size, fix=0):


    dataset = SR_fix_H5(
        hdf5_path='./HDF5/output_data.h5',
        split='test',
        scale=Scale,
        input_size=Input_size,
        fix=fix
    )
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        drop_last=False
    )

    MAE = []
    RMSE = []
    FCE = []

    start_time = time.time()
    for y, z in test_loader:
        x = np.array(y.cpu().squeeze(0).squeeze(0))
        y = np.array(z.cpu().squeeze(0).squeeze(0))
        pred = reverse_normalization(conventional_interp(x, scale=Scale, method='nearest'))
        y = reverse_normalization(y)

        #pred = conventional_interp(x, scale=Scale, method='nearest')

        """plt.figure(figsize=(10, 6))
        plt.plot(pred, label='Prediction', color='blue')
        plt.plot(y, label='Actual', color='orange')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Prediction vs Actual')
        plt.legend()
        plt.grid(True)
        plt.show()"""

        MAE.append(mean_absolute_error(y, pred))
        RMSE.append(calcRMSE(y, pred))
        FCE.append(fce_metric(pred, y))

    total_testing_time = time.time() - start_time
    print("MAE ", Average_(MAE))
    print("RMSE ", Average_(RMSE))
    print("FCE ", Average_(FCE))
    print("MAE", MAE)
    print("RMSE ", RMSE)
    print("FCE ", FCE)
    print("Total Testing Time: {:.2f} sec".format(total_testing_time))


def compare_metrics_proposed(input_dir, n_groups, n_blocks, Scale, Input_size, fix=0):

    Scale = Scale
    Input_size = Input_size

    model = BlindSR(scale=Scale, n_groups=n_groups, n_blocks=n_blocks).cuda()
    checkpoint = torch.load(f'./models/scale_2/{input_dir}')

    model.load_state_dict(checkpoint['model_state_dict'])
    ######################################

    dataset = SR_fix_H5(
        hdf5_path='./HDF5/output_data.h5',
        split='test',
        scale=Scale,
        input_size=Input_size,
        fix=fix
    )
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        drop_last=False
    )
    #####################
    MAE = []
    RMSE = []
    FCE = []

    for y, z in test_loader:
        x = y.cuda()
        y = z.cuda()
        model.eval()

        with torch.no_grad():
            pred = model(x, x)
            pred = reverse_normalization(np.array(pred.cpu().squeeze(0).squeeze(0)))
            y = reverse_normalization(np.array(y[0].cpu().squeeze(0).squeeze(-1)))

            #pred = np.array(pred.cpu().squeeze(0).squeeze(0))
            #y = np.array(y[0].cpu().squeeze(0).squeeze(-1))

            """plt.figure(figsize=(10, 6))
            plt.plot(pred, label='Prediction', color='blue')
            plt.plot(y, label='Actual', color='orange')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Prediction vs Actual')
            plt.legend()
            plt.grid(True)
            plt.show()"""

            MAE.append(mean_absolute_error(y, pred))
            RMSE.append(calcRMSE(y, pred))
            FCE.append(fce_metric(pred, y))

    print("MAE ", Average_(MAE))
    print("RMSE ", Average_(RMSE))
    print("FCE ", Average_(FCE))

    print("MAE", MAE)
    print("RMSE ", RMSE)
    print("FCE ", FCE)


def compare_metrics(model_choice, input_dir, Scale, Input_size, fix):

    if model_choice == 'noencoder':
        print("Training SuperResolutionNet1D")
        model = DASR1D_SingleInput(scale=Scale, n_groups=5, n_blocks=5).cuda()
    elif model_choice == 'srpcnn':
        print("Training SRPCNN")
        model = SRPNSE(upscale_factor=Scale).cuda()
    elif model_choice == 'generator':
        print("Training Generator")
        model = SuperResolutionNet(upscale_factor=Scale).cuda()
    else:
        raise ValueError("Unknown model_choice. Please select 'srnet', 'srpcnn', or 'generator'.")

    checkpoint = torch.load(f'./models/scale_2/{input_dir}')
    ######################################################
    model.load_state_dict(checkpoint['model_state_dict'])
    ######################################
    dataset = SR_fix_H5(
        hdf5_path='./HDF5/output_data.h5',
        split='test',
        scale=Scale,
        input_size=Input_size,
        fix=fix
    )
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        drop_last=False
    )

    MAE = []
    RMSE = []
    FCE = []

    start_time = time.time()
    for y, z in test_loader:
        x = y.cuda()
        y = z.cuda()
        model.eval()

        with torch.no_grad():
           pred = model(x)


           pred = reverse_normalization(np.array(pred.cpu().squeeze(0).squeeze(0)))
           y = reverse_normalization(np.array(y[0].cpu().squeeze(0).squeeze(-1)))

           #pred = np.array(pred.cpu().squeeze(0).squeeze(0))
           #y = np.array(y[0].cpu().squeeze(0).squeeze(-1))

           """plt.figure(figsize=(10, 6))
           plt.plot(pred, label='Prediction', color='blue')
           plt.plot(y, label='Actual', color='orange')
           plt.xlabel('Index')
           plt.ylabel('Value')
           plt.title('Prediction vs Actual')
           plt.legend()
           plt.grid(True)
           plt.show()"""

           MAE.append(mean_absolute_error(y, pred))
           RMSE.append(calcRMSE(y, pred))
           FCE.append(fce_metric(pred, y))

    total_testing_time = time.time() - start_time
    print("MAE ", Average_(MAE))
    print("RMSE ", Average_(RMSE))
    print("FCE ", Average_(FCE))

    print("MAE", MAE)
    print("RMSE ", RMSE)
    print("FCE ", FCE)
    print("Total Testing Time: {:.2f} sec".format(total_testing_time))


if __name__ == "__main__":

    #compute_metics_interpolation(Scale=2, Input_size=144, fix=0.2)

    #compare_metrics_proposed(input_dir='Proposed_ADAM_trial1.pth', n_groups=5, n_blocks=5, Scale=4, Input_size=360, fix=0.3)
    #compare_metrics_proposed(input_dir='Proposed_ADAM_trial2.pth', n_groups=5, n_blocks=5, Scale=4, Input_size=360, fix=0.3)
    #compare_metrics_proposed(input_dir='Proposed_ADAM_trial3.pth', n_groups=5, n_blocks=5, Scale=4, Input_size=360, fix=0.3)
    #compare_metrics_proposed(input_dir='Proposed_ADAM_trial4.pth', n_groups=5, n_blocks=5, Scale=4, Input_size=360, fix=0.3)


    #compare_metrics_proposed(input_dir='Proposed_ADAM_144_2_trial1.pth', n_groups=5, n_blocks=5, Scale=2, Input_size=144, fix=0.2)
    #compare_metrics_proposed(input_dir='Proposed_ADAM_144_2_trial2.pth', n_groups=5, n_blocks=5, Scale=2, Input_size=144, fix=0.2)
    compare_metrics_proposed(input_dir='Proposed_ADAM_144_2_trial3.pth', n_groups=5, n_blocks=5, Scale=2, Input_size=144, fix=0.2)


    #compare_metrics_proposed(input_dir='ADAM_14.pth', n_groups=14, n_blocks=5, Scale=4, Input_size=360, fix=0)

    #compare_metrics(model_choice='noencoder', input_dir='noencoder_input_size_144_2_trial1.pth', Scale=2, Input_size=144, fix=0.2)
    #compare_metrics(model_choice='noencoder', input_dir='noencoder_input_size_144_2_trial2.pth', Scale=2, Input_size=144, fix=0.2)
    #compare_metrics(model_choice='noencoder', input_dir='noencoder_input_size_144_2_trial3.pth', Scale=2, Input_size=144, fix=0.2)

    #compare_metrics(model_choice='srpcnn', input_dir='srpcnn_trial1.pth', Scale=4, Input_size=360, fix=0.3)
    #compare_metrics(model_choice='srpcnn', input_dir='srpcnn_trial2.pth', Scale=4, Input_size=360, fix=0.3)
    #compare_metrics(model_choice='srpcnn', input_dir='srpcnn_trial3.pth', Scale=4, Input_size=360, fix=0.3)

    #compare_metrics(model_choice='noencoder', input_dir='noencoder_2_trial1.pth', Scale=2, Input_size=360, fix=0.3)
    #compare_metrics(model_choice='noencoder', input_dir='noencoder_2_trial2.pth', Scale=2, Input_size=360, fix=0.3)
    #compare_metrics(model_choice='noencoder', input_dir='noencoder_2_trial3.pth', Scale=2, Input_size=360, fix=0.3)


    #count_data_in_hdf5('./HDF5/output_data.h5')

    #Training data count: 7368
    #Validation data count:2456
    #Testing data count: 2457