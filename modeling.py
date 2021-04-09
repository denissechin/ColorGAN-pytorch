import torch
import torch.nn as nn
import torchvision.models
from spectral import SpectralNorm

def conv_relu_block(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
      )


class ResNetUNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet_model = torchvision.models.resnet18(pretrained=True)
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50_layers = list(resnet50.children())
        self.resnet_layers = list(resnet_model.children())

        self.conv_original_size0 = conv_relu_block(1, 64, 3, 1)
        self.conv_original_size1 = conv_relu_block(64, 64, 3, 1)



        self.layer0 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),) # first resnet layer adapted for 1 channel input
        self.layer1 = nn.Sequential(*self.resnet_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.resnet_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.resnet_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.resnet_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.global_features = nn.Sequential(*self.resnet50_layers[3:-2]) # conv to 2048 features
        self.layer4_1x1 = conv_relu_block(512+2048, 256+1024, 1, 0)
        self.layer3_1x1 = conv_relu_block(256, 256, 1, 0)
        self.layer0_1x1 = conv_relu_block(64, 64, 1, 0)
        self.layer1_1x1 = conv_relu_block(64, 64, 1, 0)
        self.layer2_1x1 = conv_relu_block(128, 128, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = conv_relu_block(1536 , 512, 3, 1)
        self.conv_up2 = conv_relu_block(128 + 512, 256, 3, 1)
        self.conv_up1 = conv_relu_block(64 + 256, 256, 3, 1)
        self.conv_up0 = conv_relu_block(64 + 256, 128, 3, 1)

        self.conv_original_size2 = conv_relu_block(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 3, 1)
        self.activation = nn.Tanh()
        
        self.frozen = False
        
    def freeze_parameters(self):
        for param in self.resnet_layers:
            param.requires_grad = False
    
        for param in self.resnet50_layers:
            param.requires_grad = False
        
        self.frozen = True

    def unfreeze_parameters(self):
        for param in self.resnet_layers:
            param.requires_grad = True
    
        for param in self.resnet50_layers:
            param.requires_grad = True
            
        self.frozen = False


    def forward(self, input_image):
        
        x_original1 = self.conv_original_size0(input_image)
        x_original = self.conv_original_size1(x_original1)

        layer0 = self.layer0(input_image)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)


        global_features = self.global_features(layer0)
        merged_features = torch.cat((global_features, layer4), dim=1)
        layer4 = self.layer4_1x1(merged_features)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x) # 512

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x) #256

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x) # 256

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        out = self.activation(out)

        return out
    
    
class Discriminator(nn.Module):
    def __init__(self, ndf, input_height):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) input_height
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) input_height//2
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) input_height//4
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) input_height//8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(ndf*2*(input_height//16)**2, 1, bias=True),
                               nn.Sigmoid())

    def forward(self, input):
        out = self.main(input)
        out = self.fc(out)
        return out
