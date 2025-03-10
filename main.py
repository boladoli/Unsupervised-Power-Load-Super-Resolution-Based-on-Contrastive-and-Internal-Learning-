from model import Encoder
from moco import MoCo
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from dataset import SRDataset
from ult import accuracy


def train(optimizer_name='SGD'):
    print(f"Starting training with {optimizer_name.upper()} optimizer")
    moco = MoCo(base_encoder=Encoder).cuda()
    SCALE_FACTOR = 4
    BATCH_SIZE = 256
    INPUT_SIZE = 360
    lr = 1e-4
    max_epochs = 150

    # Create training dataset and DataLoader with pinned memory.
    train_dataset = SRDataset(
        hdf5_path='./HDF5/output_data.h5',
        split='train',
        scale=SCALE_FACTOR,  # Example scale factor
        input_size=INPUT_SIZE
    )
    train_dloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        prefetch_factor=4,
        pin_memory=True,
        drop_last=True# Speeds up host-to-device transfer
    )


    # Create training dataset and DataLoader with pinned memory.
    v_dataset = SRDataset(
        hdf5_path='./HDF5/output_data.h5',
        split='val',
        scale=SCALE_FACTOR,  # Example scale factor
        input_size=INPUT_SIZE
    )
    val_dloader = DataLoader(
        v_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        prefetch_factor=4,
        pin_memory=True,
        drop_last=True# Speeds up host-to-device transfer
    )


    contrast_loss = torch.nn.CrossEntropyLoss().cuda()

    # 根据输入的 optimizer_name 选择优化器
    if optimizer_name.upper() == 'SGD':
        optimizer = optim.SGD(moco.encoder_q.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    elif optimizer_name.upper() == 'ADAM':
        optimizer = optim.Adam(moco.encoder_q.parameters(), lr=lr, weight_decay=0.0001)
    elif optimizer_name.upper() == 'RMSPROP':
        optimizer = optim.RMSprop(moco.encoder_q.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    else:
        raise ValueError(
            f"Unsupported optimizer name: {optimizer_name}. Please choose from 'SGD', 'ADAM', or 'RMSPROP'.")

    for epoch in range(max_epochs):
        # 训练阶段
        moco.train()
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_batches = 0

        for x_query, x_key in train_dloader:
            x_query, x_key = x_query.cuda(), x_key.cuda()
            optimizer.zero_grad()
            _, logits, labels = moco(x_query, x_key)
            loss = contrast_loss(logits, labels)
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            total_loss += loss.item()
            total_acc1 += acc1.item()
            total_acc5 += acc5.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        avg_acc1 = total_acc1 / total_batches
        avg_acc5 = total_acc5 / total_batches

        print(f"Epoch {epoch + 1}: Training Loss = {avg_loss:.6f}, "
              f"Avg Acc@1 = {avg_acc1:.2f}%, Avg Acc@5 = {avg_acc5:.2f}%")


    # 保存训练好的模型
    torch.save({
        'model_state_dict': moco.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_settings': {
            'queue': moco.queue,
            'encoder_q_state_dict': moco.encoder_q.state_dict(),
            'encoder_k_state_dict': moco.encoder_k.state_dict(),
        }
    }, f'./models/moco_{optimizer_name.upper()}.pth')


if __name__ == "__main__":
    train(optimizer_name='SGD')
    train(optimizer_name='ADAM')
    #train(optimizer_name='RMSPROP')

