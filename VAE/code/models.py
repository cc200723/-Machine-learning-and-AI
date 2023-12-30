import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        assert mask_type in {"A", "B"}
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, kH // 2 + 1 :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ConvnetBlock(nn.Module):
    def __init__(self, filters, *args, **kwargs):
        super(ConvnetBlock, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(filters, filters, (3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(filters, filters, (3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        #
        # Task 1: Implement forward function for residual convnet block.
        #
        # Save the original input for the residual connection
        identity = x

        # First conv layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Adding the residual (identity)
        out += identity

        # Applying activation function after adding the residual
        out = F.relu(out)

        return out


class GaussianVAEDecoder(nn.Module):
    def __init__(self, capacity=32, depth=51, autoregress=False, *args, **kwargs):
        super(GaussianVAEDecoder, self).__init__(*args, **kwargs)
        self.capacity = capacity

        self.embed = nn.Linear(49, capacity * 7 * 7, bias=False)

        self.resnet = nn.ModuleList()
        for i in range(depth):
            self.resnet.append(ConvnetBlock(capacity))

        self.image = nn.ConvTranspose2d(capacity, 1, 4, stride=4, bias=True)
        self.bias = nn.Parameter(torch.Tensor(28, 28))

        for name, parm in self.named_parameters():
            if name.endswith("weight"):
                nn.init.normal_(parm, 0, 0.01)
            if name.endswith("bias"):
                nn.init.constant_(parm, 0.0)

    def sample(self, z, sigma):
        return torch.normal(self(z), sigma).clamp(0, 1)

    def forward(self, s):
        zx = F.relu(self.embed(s.view(-1, 49)).view(-1, self.capacity, 7, 7))
        for layer in self.resnet:
            zx = layer(zx)
        return torch.sigmoid(self.image(zx) + self.bias[None, None, :, :])


class VAEEncoder(nn.Module):
    def __init__(self, capacity=32, depth=9, flows=0, *args, **kwargs):
        super(VAEEncoder, self).__init__(*args, **kwargs)
        self.capacity = capacity

        self.embed = nn.Conv2d(1, capacity, 7, padding=3, stride=4, bias=False)

        self.resnet = nn.ModuleList()
        for i in range(depth):
            self.resnet.append(ConvnetBlock(capacity))
        self.mu = nn.Conv2d(capacity, 1, 3, padding=1, bias=True)
        self.var = nn.Conv2d(capacity, 1, 3, padding=1, bias=True)

        self.flows = nn.ModuleList()

        for name, parm in self.named_parameters():
            if name.endswith("weight"):
                nn.init.normal_(parm, 0, 0.01)
            if name.endswith("bias"):
                nn.init.constant_(parm, 0.0)

    def forward(self, x, epsilon):
        h = F.relu(self.embed(x))
        for layer in self.resnet:
            h = layer(h)
        mu, logvar = self.mu(h), self.var(h)

        z = mu + torch.exp(0.5 * logvar) * epsilon
        logqzx = 0.5 * (logvar + torch.pow((z - mu), 2) / logvar.exp()).view(
            -1, 49
        ).sum(1)

        return z.view(-1, 49), logqzx, mu.view(-1, 49), logvar.view(-1, 49)

