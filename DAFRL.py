"""DAFRL training entry point.

The original implementation executed training during import and contained
machine-specific paths.  Keep the model definitions reusable and expose a
small command-line entry point instead.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from skimage.metrics import peak_signal_noise_ratio as psnr

def mpsnr(X,Y):
    s = 0
    for b in range(X.shape[-1]):
        s = s + psnr(X[:,:,b],Y[:,:,b],data_range=1)
    s = s/X.shape[-1]
    return s

#---------------------------------------------Unet-----------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting path (Encoder)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Expansive path (Decoder)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))

        # Decoder
        dec4 = self.upconv4(bottleneck)

        if dec4.shape != enc4.shape:
            diffY = enc4.size()[2] - dec4.size()[2]
            diffX = enc4.size()[3] - dec4.size()[3]

            dec4 = F.pad(dec4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        output = self.final_conv(dec1)

        # Return output and feature maps from each layer
        return output, {'enc1': enc1, 'enc2': enc2, 'enc3': enc3, 'enc4': enc4,
                        'bottleneck': bottleneck, 'dec4': dec4, 'dec3': dec3,
                        'dec2': dec2, 'dec1': dec1}

#-----------------------------------Unet-----------------------------------

#-----------------------------------Xnet(initialization)-----------------------------------
class Xnet(nn.Module):
    def __init__(self, R):
        super().__init__()
        self.x = nn.Parameter(R)

    def forward(self):
        return self.x

#-----------------------------------Xnet(initialization)-----------------------------------
def train(data_path, output_dir, num_epochs=20000, device=None):
    """Train DAFRL on a MATLAB file containing GT, Mask and Ref arrays."""
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mat = scipy.io.loadmat(data_path)
    required = {'GT', 'Mask', 'Ref'}
    missing = required.difference(mat)
    if missing:
        raise KeyError(f'Missing variables in {data_path}: {sorted(missing)}')
    GT = mat['GT']
    Mask = mat['Mask']
    Ref = mat['Ref']
    Mask_np = mat['Mask']
    dtype = torch.float32
    Z1 = torch.rand(GT.shape, dtype=dtype, device=device)
    Z2 = torch.rand(GT.shape, dtype=dtype, device=device)

    O = torch.from_numpy((1 - Mask) * GT).to(device=device, dtype=dtype)
    Mask = torch.from_numpy(Mask).to(device=device, dtype=dtype)
    Ref = torch.from_numpy(Ref).to(device=device, dtype=dtype)

    best_psnr = 0
    best_state = None
    for alpha1 in [1,10]:
        for alpha2 in [1]:
            for beta, gamma in [(1, 1), (10, 1)]:
                print(f' alpha1={alpha1}, alpha2={alpha2}, beta={beta}, gamma={gamma}')

                net = Xnet(O).to(device)
                Xchannel = GT.shape[2]*GT.shape[3]
                unet = UNet(in_channels=Xchannel, out_channels=Xchannel).to(device)
                params = [p for p in net.parameters()] + [p for p in unet.parameters()]
                Adam = torch.optim.Adam(params, lr=5e-3)  # 5e-3,5e-2
                mse = nn.MSELoss()

                for epoch in range(num_epochs):
                    X = net()
                    unetX, featuresX = unet(X.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3], 1).permute(3, 2, 0, 1))
                    fx = featuresX['enc1'].type(dtype)
                    unetX = (unetX.permute(2, 3, 1, 0).reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])).type(dtype)

                    unetR, featuresR = unet(Ref.data.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3], 1).permute(3, 2, 0,1))
                    fr = featuresR['enc1'].type(dtype)
                    unetR = (unetR.permute(2, 3, 1, 0).reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])).type(dtype)

                    unetZ1, featuresZ1 = unet(Z1.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3], 1).permute(3, 2, 0, 1))
                    unetZ1 = (unetZ1.permute(2, 3, 1, 0).reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])).type(dtype)
                    # fz1 = featuresZ1['enc1'].type(dtype)

                    unetZ2, featuresZ2 = unet(Z2.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3], 1).permute(3, 2, 0, 1))
                    unetZ2 = (unetZ2.permute(2, 3, 1, 0).reshape(X.shape[0], X.shape[1], X.shape[2],X.shape[3])).type(dtype)
                    # fz2 = featuresZ2['enc1'].type(dtype)

                    # loss = alpha1 * mse(unetZ1, X) + alpha2 * mse(unetZ2, Ref.data) + beta * mse(fx, fr)
                    loss = alpha1 * mse(unetZ1, X) + alpha2 * mse(unetZ2, Ref.data) + beta * mse(fx, fr) + gamma * mse((1 - Mask) * X, (1 - Mask) * Ref.data)
                    # loss = alpha1 * mse(unetZ1, X) + alpha2 * mse(unetZ2, Ref.data) + beta * mse(fz1, fz2) + gamma * mse((1 - Mask) * X, (1 - Mask) * Ref.data)
                    # loss = alpha1 * mse(unetX, X) + alpha2 * mse(unetR, Ref.data) + 0 * mse(fx,fr) + gamma * mse((1 - Mask) * X, (1 - Mask) * Ref.data)

                    Adam.zero_grad()
                    loss.backward()
                    Adam.step()

                    if epoch % 100 == 0:
                        X_np = X.detach().cpu().numpy()
                        X0 = X_np[:, :, :, 0] * Mask_np[:, :, :, 0] + (1 - Mask_np[:, :, :, 0]) * GT[:, :, :, 0]
                        X1 = X_np[:, :, :, 1] * Mask_np[:, :, :, 1] + (1 - Mask_np[:, :, :, 1]) * GT[:, :, :, 1]
                        X2 = X_np[:, :, :, 2] * Mask_np[:, :, :, 2] + (1 - Mask_np[:, :, :, 2]) * GT[:, :, :, 2]
                        m0 = mpsnr(X0, GT[:, :, :, 0])
                        m1 = mpsnr(X1, GT[:, :, :, 1])
                        m2 = mpsnr(X2, GT[:, :, :, 2])
                        if m0 + m1 + m2 > best_psnr * 3:
                            best_psnr = (m0 + m1 + m2) / 3
                            best_psnr0 = m0
                            best_psnr1 = m1
                            best_psnr2 = m2
                            best_alpha1 = alpha1
                            best_alpha2 = alpha2
                            best_beta = beta
                            best_gamma = gamma
                            best_state = X_np.copy()
                        print(
                            'Iter: %5d, Loss: %.8f, Node1: %.5f, Node2: %.5f, Node3: %.5f, Best1: %.5f, Best2: %.5f Best3: %.5f， '
                            'alpha1: %.5f, alpha2: %.5f, beta: %.5f， bestPSNR: %.5f '
                            %(epoch, loss.item(), m0, m1, m2, best_psnr0, best_psnr1, best_psnr2, alpha1, alpha2, beta, best_psnr))
                with open(output_dir / 'training.log', 'a', encoding='utf-8') as f:
                        bestPSNR = str(best_psnr)
                        f.write('France' + ' bestPSNR:' + bestPSNR[0:5] + ' node1:' + str(best_psnr0) + ' node2:' + str(best_psnr1)
                                + ' node3:' + str(best_psnr2) + ' alpha1:' + str(best_alpha1) + ' alpha2:' + str(best_alpha2)
                                + ' beta:' + str(best_beta) + ' gamma:' + str(best_gamma) + '\n')
    if best_state is None:
        raise RuntimeError('No checkpoint was produced; increase num_epochs or inspect the data.')
    scipy.io.savemat(output_dir / 'prediction.mat',
                     {'pred': best_state * Mask_np + (1 - Mask_np) * GT})


def main():
    parser = argparse.ArgumentParser(description='Train DAFRL on a MATLAB dataset.')
    parser.add_argument('--data', required=True, help='Path to a .mat file with GT, Mask and Ref.')
    parser.add_argument('--output-dir', default='results', help='Directory for logs and prediction.mat.')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--device', default=None, help='torch device, e.g. cpu or cuda:0')
    args = parser.parse_args()
    train(args.data, args.output_dir, args.epochs, args.device)


if __name__ == '__main__':
    main()

