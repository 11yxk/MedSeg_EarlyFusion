""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomAffine, RandomElasticTransform, Normalize
class BERTModel(nn.Module):

    def __init__(self, bert_type = 'microsoft/BiomedVLP-CXR-BERT-specialized', project_dim = 784):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        embed = self.project_head(embed)

        return {'feature':output['hidden_states'],'project':embed}

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels = 6, n_classes = 1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.text_encoder = BERTModel()

        self.norm = AugmentationSequential(

            # Normalize
            Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406, 0.5,0.5,0.5]),
            std=torch.tensor([0.229, 0.224, 0.225,0.25,0.25,0.25]),
            ),
            data_keys=["input", "mask"],  # 指定输入和掩码都进行变换
        )



        self.generator = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  # 112 -> 224
            nn.Sigmoid()  # 输出范围 0~1
        )



        print('======================================')
        print('github_UNET_aug_text generate')
        print('======================================')

    def forward(self, data):


        x, _text, gt, gt_generate = data
        text = self.text_encoder(_text['input_ids'],_text['attention_mask'])
        proj = text['project']
        batch, _ = text['project'].shape
        proj = proj.reshape(batch,1,28,28)
        resized_tensor = self.generator(proj)


        # if True:
        #
        #     import cv2
        #     import numpy as np
        #
        #     for b in range(gt.shape[0]):
        #         # 获取当前的 gt 和特征图
        #         gt_image = gt_generate[b].permute(1,2,0).cpu().numpy()
        #         array = resized_tensor[b].permute(1,2,0).cpu().numpy()
        #
        #         normalized_gt_image = cv2.normalize(gt_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #         normalized_gt_image = normalized_gt_image.astype(np.uint8)
        #
        #         # 显示 groundtruth 图像
        #         normalized_gt_image = cv2.cvtColor(normalized_gt_image, cv2.COLOR_BGR2RGB)
        #         cv2.imshow('GT', normalized_gt_image)
        #
        #         normalized_array = cv2.normalize(array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #         normalized_array = normalized_array.astype(np.uint8)
        #         # normalized_array = cv2.applyColorMap(normalized_array, cv2.COLORMAP_JET)
        #
        #         # 显示 groundtruth 图像
        #         # normalized_array = cv2.cvtColor(normalized_array, cv2.COLOR_BGR2RGB)
        #         normalized_array = cv2.cvtColor(normalized_array, cv2.COLOR_BGR2GRAY)
        #         cv2.imshow('pred', normalized_array)
        #
        #
        #         cv2.waitKey(0)

        x = torch.cat([x,resized_tensor],dim = 1)
        x, gt = self.norm(x, gt)

        gt = (gt > 0.5).int()



        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # if True:
        #
        #     import cv2
        #     import numpy as np
        #
        #     logits_ = torch.sigmoid(logits)
        #     logits_ = (logits_>0.5)*255
        #
        #     for b in range(gt.shape[0]):
        #         # 获取当前的 gt 和特征图
        #         gt_image = gt[b,0].cpu().numpy().astype('uint8')*255
        #         logits_img = logits_[b,0].cpu().numpy().astype('uint8')
        #
        #         cv2.imshow('GT', gt_image)
        #         cv2.imshow('pred', logits_img)
        #         cv2.waitKey(0)



        return torch.sigmoid(logits), gt, resized_tensor, gt_generate



if __name__ == "__main__":
    model = UNet()

    x = torch.randn(1, 3, 352, 352)
    y = model(x)
    print(y.shape)