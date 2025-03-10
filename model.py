import torch
import torch.nn as nn
import torch.nn.functional as F
from moco import MoCo


class PixelShuffle1D(torch.nn.Module):

    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv1d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class DA_conv1D(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv1D, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size, bias=False)  # Adjusted for 1D
        )
        self.conv = default_conv(channels_in, channels_out, 1)  # Assuming default_conv1d is adapted for 1D
        self.ca = CA_layer1D(channels_in, channels_out, reduction)  # Adjust CA_layer for 1D

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        b, c, l = x[0].size()  # Adjusted for 1D
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size)
        out = self.relu(F.conv1d(x[0].view(1, -1, l), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        out = self.conv(out.view(b, -1, l))

        # branch 2
        out = out + self.ca(x)
        return out


class CA_layer1D(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer1D, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv1d(channels_in, channels_in//reduction, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(channels_in // reduction, channels_out, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #att = self.conv_du(x[1][:, None, :])  # Adjusted for 1D
        att = self.conv_du(x[1][:, :, None])
        return x[0] * att


class DAB1D(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB1D, self).__init__()

        self.da_conv1 = DA_conv1D(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv1D(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)  # Assuming conv is adapted for 1D
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]

        return out


class DAG1D(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG1D, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DAB1D(conv, n_feat, kernel_size, reduction) for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))  # Assuming conv is adapted for 1D

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


class DASR1D(nn.Module):
    def __init__(self, scale, conv=default_conv, n_groups=2, n_blocks=5):
        super(DASR1D, self).__init__()

        self.n_groups = n_groups
        n_blocks = n_blocks

        print("n_groups", self.n_groups)
        print("n_blocks", n_blocks)

        n_feats = 64
        kernel_size = 3
        reduction = 8
        scale = int(scale)  # The stride for your transposed convolution

        # -------------------------
        # Head Module
        # -------------------------
        modules_head = [conv(1, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # -------------------------
        # Compress (k_v)
        # -------------------------
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )
        # -------------------------
        # Body
        # -------------------------
        modules_body = [
            DAG1D(conv, n_feats, kernel_size, reduction, n_blocks)
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # -------------------------
        # Tail (Replacing PixelShuffle1D)
        # -------------------------
        # Single transposed convolution:
        self.tail = nn.ConvTranspose1d(
            in_channels=64,
            out_channels=1,
            kernel_size=7,
            stride=scale,
            padding=3,
            output_padding=scale - 1
        )

    def forward(self, x, k_v):
        # Compress the auxiliary vector k_v
        k_v = self.compress(k_v)
        # Pass x through the head
        x = self.head(x)
        # Go through each group (DAG1D block)
        res = x
        for i in range(self.n_groups):
            res = self.body[i]([res, k_v])
        # The last element in self.body is a conv layer
        res = self.body[-1](res)
        # Global residual connection
        res = res + x
        # Finally, transposed convolution for upsampling
        out = self.tail(res)
        return out


class Encoder(nn.Module):
    def __init__(self, num_classes=256):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1)
        out = self.mlp(fea)
        return fea, out


class BlindSR(nn.Module):

    def __init__(self, scale, n_groups, n_blocks):
        super(BlindSR, self).__init__()

        self.E = MoCo(base_encoder=Encoder)
        self.G = DASR1D(scale=scale, n_groups=n_groups, n_blocks=n_blocks)

    def forward(self, x, x2):
        if self.training:
            fea, logits, labels = self.E(x, x2)
            sr = self.G(x, fea)
            return sr, logits, labels
        else:
            fea = self.E(x, x)
            sr = self.G(x, fea)
            return sr


if __name__ == "__main__":
    model = Encoder()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
