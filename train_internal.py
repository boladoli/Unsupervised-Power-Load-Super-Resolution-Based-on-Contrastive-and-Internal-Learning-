from torch.utils.data import DataLoader
from dataset import SR
import torch
from torch import nn
from model import BlindSR


def train(Pre_load, optimizer_name='SGD', lr_rate=1e-3, n_groups=2, n_blocks=5, b_factor=1, name='random', Scale=4, input_size=360):

    print(f"Starting training with {optimizer_name.upper()} optimizer")
    print("b factor is ", b_factor)
    Pre_load = Pre_load
    Input_size =input_size
    Batch_size = 64
    n_epochs = 100
    std_max = 0.3
    print("std_max is ", std_max)

    # Create training dataset and DataLoader with pinned memory.
    train_dataset = SR(
        hdf5_path='./HDF5/output_data.h5', #'./HDF5/output_data.h5'
        split='train',
        scale=Scale,  # Example scale factor
        input_size=Input_size,
        std_min=0,
        std_max=std_max
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Batch_size,
        shuffle=True,
        num_workers=6,
        prefetch_factor=4,
        pin_memory=True,
        drop_last=True# Speeds up host-to-device transfer
    )

    # Create validation dataset and DataLoader with pinned memory.
    validation_dataset = SR(
        hdf5_path='./HDF5/output_data.h5',
        split='val',
        scale=Scale,  # Example scale factor
        input_size=Input_size,
        std_min=0,
        std_max=std_max
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=Batch_size,
        shuffle=False,
        num_workers=6,
        prefetch_factor=4,
        drop_last=False,
        pin_memory=True
    )

    model = BlindSR(scale=Scale, n_groups=n_groups, n_blocks=n_blocks).cuda()
    contrast_loss = torch.nn.CrossEntropyLoss()

    # 根据输入的 optimizer_name 选择优化器
    if optimizer_name.upper() == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay=0.0001)
    elif optimizer_name.upper() == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=0.0001)
    elif optimizer_name.upper() == 'RMSPROP':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay=0.0001)
    else:
        raise ValueError(
            f"Unsupported optimizer name: {optimizer_name}. Please choose from 'SGD', 'ADAM', or 'RMSPROP'.")

    mse_loss = nn.MSELoss()
    ######################
    if Pre_load == True:

        checkpoint = torch.load(f'./models/pre_training/model_{optimizer_name.upper()}.pth')
        model.E.queue = checkpoint['model_settings']['queue']
        model.E.encoder_q.load_state_dict(checkpoint['model_settings']['encoder_q_state_dict'])
        model.E.encoder_k.load_state_dict(checkpoint['model_settings']['encoder_k_state_dict'])

        orig_optimizer_state = checkpoint['optimizer_state_dict']
        filtered_state_dict = optimizer.state_dict()
        for key in filtered_state_dict['state'].keys():
            if key in orig_optimizer_state['state']:
                filtered_state_dict['state'][key] = orig_optimizer_state['state'][key]
        optimizer.load_state_dict(filtered_state_dict)
        print("preload__")

    ######################
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for i, (lr, hr, _, lr2, _, _) in enumerate(train_loader):
            lr = lr.cuda()
            hr = hr.cuda()
            lr2 = lr2.cuda()

            optimizer.zero_grad()

            reconstrcted, output, target = model(lr, lr2)
            loss_SR = mse_loss(reconstrcted, hr)
            loss_constrast = contrast_loss(output, target)

            loss = b_factor * loss_constrast + loss_SR
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_tloss = running_loss / len(train_loader)


        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for lr_imgs, hr_imgs, _, _, _ in validation_loader:
                lr_imgs = lr_imgs.cuda()
                hr_imgs = hr_imgs.cuda()

                sr_imgs = model(lr_imgs, lr_imgs)
                loss = mse_loss(sr_imgs, hr_imgs)
                val_loss += loss.item()
        avg_vloss = val_loss / len(validation_loader)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {avg_tloss:.6f}, Validation Loss: {avg_vloss:.6f}")


        if avg_vloss < best_val_loss:
            best_val_loss = avg_vloss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f'./models/scale_2/Proposed_{optimizer_name.upper()}_{input_size}_{Scale}_{name}.pth')


if __name__ == '__main__':

    #train(Pre_load=False, optimizer_name='ADAM', lr_rate=1e-3, n_groups=5, n_blocks=2, b_factor=0.01, name='trial1')
    train(Pre_load=False, optimizer_name='ADAM', lr_rate=1e-3, n_groups=5, n_blocks=5, b_factor=0.01, name='trial1')
    #train(Pre_load=False, optimizer_name='ADAM', lr_rate=1e-3, n_groups=5, n_blocks=8, b_factor=0.01, name='trial1')
    #train(Pre_load=False, optimizer_name='ADAM', lr_rate=1e-3, n_groups=5, n_blocks=5, b_factor=0.01, name='trial4')
    #train(Pre_load=False, optimizer_name='ADAM', lr_rate=1e-3, n_groups=5, n_blocks=5, b_factor=0.01, name='trial5')
    #train(Pre_load=False, optimizer_name='ADAM', lr_rate=1e-3, n_groups=5, n_blocks=5, b_factor=0.01, name='trial6')

