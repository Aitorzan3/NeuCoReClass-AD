# Models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *
from datasets import *

from torch.utils.data import DataLoader
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class res_trans1d_block(torch.nn.Module):

    def __init__(self, channel,bias=False):
        super(res_trans1d_block, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(channel, channel, 3, 1, 1, bias=bias)
        self.in1 = nn.InstanceNorm1d(channel, affine=bias)
        self.conv2 = nn.Conv1d(channel, channel, 3, 1, 1, bias=bias)
        self.in2 = nn.InstanceNorm1d(channel, affine=bias)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1,bias = False):
        super(ConvLayer, self).__init__()
        padding = dilation * (kernel_size // 2)
        self.reflection_pad = nn.ReflectionPad1d(padding)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias)  # , padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv1d(out)
        return out


class SeqTransformNet(nn.Module):
    def __init__(self, x_dim,hdim,num_layers):
        super(SeqTransformNet, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.conv1 = ConvLayer(x_dim, hdim, 3, 1,bias=False)
        self.in1 = nn.InstanceNorm1d(hdim, affine=False)
        res_blocks = []
        for _ in range(num_layers-2):
            res_blocks.append(res_trans1d_block(hdim,False))
        self.res = nn.Sequential(*res_blocks)
        self.conv2 = ConvLayer(hdim, x_dim, 3, 1,bias=False)

    def forward(self, x):
        x = x.permute(0,2,1)
        out = self.relu(self.in1(self.conv1(x)))
        for block in self.res:
            out = block(out)
        out = self.conv2(out)
        out = out.permute(0,2,1)
        return out

class Transformations(nn.Module):
    def __init__(self, n_transforms, x_dim):
        super(Transformations, self).__init__()
        self.transform_list = nn.ModuleList(
            [SeqTransformNet(x_dim, x_dim, 5) for _ in range(1, n_transforms)])

    def forward(self, x):
        # Generate views applying the set of transformations
        augmented_views = x
        for transform in self.transform_list:
            augment = transform(x)
            augmented_views = torch.cat((augmented_views, augment), 0)
        return augmented_views

class res_block(nn.Module):

    def __init__(self, in_dim, out_dim, conv_param=None, downsample=None, batchnorm=False,bias=False):
        super(res_block, self).__init__()

        self.conv1 = nn.Conv1d(in_dim, in_dim, 1, 1, 0,bias=bias)
        if conv_param is not None:
            self.conv2 = nn.Conv1d(in_dim, in_dim, conv_param[0], conv_param[1], conv_param[2],bias=bias)
        else:
            self.conv2 = nn.Conv1d(in_dim, in_dim, 3, 1, 1,bias=bias)

        self.conv3 = nn.Conv1d(in_dim, out_dim, 1, 1, 0,bias=bias)
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(in_dim)
            self.bn2 = nn.BatchNorm1d(in_dim)
            self.bn3 = nn.BatchNorm1d(out_dim)
            if downsample:
                self.bn4 = nn.BatchNorm1d(out_dim)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.batchnorm = batchnorm

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batchnorm:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if self.batchnorm:
                residual = self.bn4(residual)

        out += residual
        out = self.relu(out)

        return out

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims=320, hidden_dims=64, depth=10):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        x = self.input_fc(x)  # B x T x Ch

        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x


class DilatedConvDecoder(nn.Module):
    """
    Este módulo intenta invertir el bloque convolucional dilatado.
    Se basa en la misma estructura que en el encoder, pero se invierte el orden de los bloques y se ajustan las dilataciones.
    """
    def __init__(self, out_channels, channels, kernel_size):
        super().__init__()
        reversed_channels = list(reversed(channels))
        self.net = nn.Sequential(*[
            ConvBlock(
                reversed_channels[i-1] if i > 0 else out_channels,
                reversed_channels[i],
                kernel_size=kernel_size,
                dilation=2**(len(reversed_channels)-1 - i),
                final=(i == len(reversed_channels)-1)
            )
            for i in range(len(reversed_channels))
        ])
        
    def forward(self, x):
        return self.net(x)

class TSDecoder(nn.Module):
    def __init__(self, output_dims, input_dims, hidden_dims=64, depth=10):
        """
        output_dims: dimensión del espacio latente del encoder (salida de TSEncoder)
        input_dims: dimensión original de la serie
        hidden_dims: la dimensión interna del decoder, que típicamente iguala al hidden_dims del encoder
        depth: número de bloques; se usará para definir la lista de canales a invertir.
        """
        super().__init__()
        channels = [hidden_dims] * depth + [output_dims]
        self.feature_restorer = DilatedConvDecoder(output_dims, channels, kernel_size=3)
        self.output_fc = nn.Linear(hidden_dims, input_dims)
        
    def forward(self, x):
        # x: (B, T, output_dims)
        x = x.transpose(1, 2)  # B x output_dims x T
        x = self.feature_restorer(x)  # B x hidden_dims x T
        x = x.transpose(1, 2)  # B x T x hidden_dims
        x = self.output_fc(x)   # B x T x input_dims
        return x



class TransformationModule(nn.Module):
    def __init__(self, input_dims, hidden_dims=64, latent_dims=320, depth=10, n_transforms = 12):
        """
        input_dims: dimensión de cada instante de la serie.
        hidden_dims: dimensión interna (proyección intermedia en el encoder).
        latent_dims: salida del TSEncoder, que será el espacio latente del autoencoder.
        """
        super().__init__()
        self.n_transforms = n_transforms
        self.transformations = Transformations(n_transforms, input_dims)
        self.encoder = TSEncoder(input_dims, output_dims=latent_dims, hidden_dims=hidden_dims, depth=depth)
        self.decoder = TSDecoder(output_dims=latent_dims, input_dims=input_dims, hidden_dims=hidden_dims, depth=depth)
        self.classifier = nn.Linear(latent_dims, n_transforms).to(device)
        self.awl = AutomaticWeightedLoss(3).to(device)

    def forward(self, x):
        # x: (B, T, input_dims)
        x = x.to(device)
        augmented = self.transformations(x)
        repre = self.encoder(augmented)  # (B, T, latent_dims)
        recon = self.decoder(repre)  # (B, T, input_dims)
        classif = self.classifier(repre.mean(dim=-2))
        return [recon, repre.mean(dim=-2), classif]



class Experiment(nn.Module):
    def __init__(self, input_dims, train_data, val_data, test_data, n_transforms = 12, temperature = 0.1, measure ='cosine', batch_size=32):
        super().__init__()
        self.n_transforms = n_transforms
        self._net = TransformationModule(input_dims, n_transforms = n_transforms).to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_loader = DataLoader(SelfDataset(train_data), batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(SelfDataset(val_data), batch_size=1, shuffle=False)
        self.test_loader = DataLoader(SelfDataset(test_data), batch_size=1, shuffle=False)
        self.train_positives = load_positives(self.train_loader, n_transforms)
        self.val_positives = load_positives(self.val_loader, n_transforms)
        self.test_positives = load_positives(self.test_loader, n_transforms)
        self.temperature = temperature
        self.measure = measure

    def forward(self, batch):
        batch = batch.to(device)
        out = self._net(batch)
        return out
        

    def training_step(self, batch, pos_indices):
        recon, repre, classif = self.forward(batch)
        classes = torch.repeat_interleave(torch.arange(self.n_transforms),batch.shape[0]).to(device)
        class_loss = torch.nn.functional.cross_entropy(classif, classes, reduction='mean')
        con_loss = contrastive_loss(repre, pos_indices, temperature=self.temperature, measure=self.measure, n_transforms = self.n_transforms)
        target = batch.repeat(self.n_transforms, 1, 1).to(device)
        recon_loss = nn.functional.mse_loss(recon, target, reduction='mean')
        return self._net.awl(recon_loss, con_loss, class_loss)
        
        
    def train_model(self, max_epochs=1000, verbose=False):
        optimizer = torch.optim.Adam(self._net.parameters(), lr=1e-3)
        train_loader = self.train_loader
        val_loader = self.val_loader
        loss_log = []
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
        for n_epoch_iters in range(max_epochs):
            losses = []
            for idx, batch in enumerate(train_loader):
                    
                optimizer.zero_grad()
                loss = self.training_step(batch, self.train_positives[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
                optimizer.step()
                losses.append(loss.item())
            current_loss = np.mean(losses)
            loss_log.append(current_loss)
            self._net.eval()
            val_loss = 0
            with torch.no_grad():
                val_loss = np.mean(self.compute_scores(val_loader))
            self._net.train()
            scheduler.step(val_loss)
            if verbose and n_epoch_iters%10==0:
                print(f"Epoch #{n_epoch_iters}: loss={current_loss}, val_loss={val_loss}")

            if scheduler.optimizer.param_groups[0]['lr'] <= 1e-6:
                print(f"Epoch #{n_epoch_iters} Early Stopping")
                break  
                
        return loss_log

    def compute_scores(self, val_loader=None):
        if val_loader is not None:
            data_loader = self.val_loader
            pos_indices = self.val_positives
        else:
            data_loader = self.test_loader
            pos_indices = self.test_positives
        scores = []
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                score = self.training_step(sample.to(device), pos_indices[idx])
                scores.append(score.item())
        return scores