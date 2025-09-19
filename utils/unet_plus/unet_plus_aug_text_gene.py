from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
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
# For nested 3 channels are required

class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


# Nested Unet

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_ch=6, out_ch=1):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

        self.text_encoder = BERTModel()

        self.augmentations = AugmentationSequential(
            # RandomHorizontalFlip(p=0.5),  # 水平翻转
            # RandomVerticalFlip(p=0.5),  # 垂直翻转
            RandomAffine(
                degrees=90,  # 仿射变换角度范围
                scale=(0.5, 1.5),  # 缩放比例范围
                translate=(0.125, 0.125),  # 平移范围
                p=0.5  # 应用概率
            ),
            # Normalize
            Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406, 0.5,0.5,0.5]),
            std=torch.tensor([0.229, 0.224, 0.225,0.25,0.25,0.25]),
            ),
            data_keys=["input", "mask"],  # 指定输入和掩码都进行变换
        )


        self.augmentations_test = AugmentationSequential(
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
        self.saved_path  = '/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/code_text2img/vis_results//unet_plus/unet_plus_aug_text_gene_test_kuang'


    def forward(self, data):

        # x, _text, gt, gt_generate = data
        x, _text, gt, gt_generate,name, real_text = data
        text = self.text_encoder(_text['input_ids'], _text['attention_mask'])
        proj = text['project']
        batch, _ = text['project'].shape
        proj = proj.reshape(batch, 1, 28, 28)
        resized_tensor = self.generator(proj)

        if 1:

            import cv2
            import numpy as np
            import os

            for b in range(gt.shape[0]):
                file_path = os.path.join(self.saved_path, real_text[b]+'_'+name[b])
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                # else:
                #     continue



                non_zero_pixels = torch.nonzero(gt[b][0])
                min_x = non_zero_pixels[:, 1].min().item()  # x 最小值
                max_x = non_zero_pixels[:, 1].max().item()  # x 最大值
                min_y = non_zero_pixels[:, 0].min().item()  # y 最小值
                max_y = non_zero_pixels[:, 0].max().item()  # y 最大值


                # ori X
                x_show = x[b].permute(1, 2, 0).cpu().numpy()
                normalized_x_show = cv2.normalize(x_show, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                normalized_x_show = normalized_x_show.astype(np.uint8)
                normalized_x_show = cv2.cvtColor(normalized_x_show, cv2.COLOR_BGR2RGB)
                cv2.rectangle(normalized_x_show, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

                # Generate GT
                gt_image = gt_generate[b].permute(1,2,0).cpu().numpy()
                normalized_gt_image = cv2.normalize(gt_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                normalized_gt_image = normalized_gt_image.astype(np.uint8)
                normalized_gt_image = cv2.cvtColor(normalized_gt_image, cv2.COLOR_BGR2RGB)
                cv2.rectangle(normalized_gt_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

                # Generate
                array = resized_tensor[b].permute(1,2,0).cpu().numpy()
                normalized_array = cv2.normalize(array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                normalized_array = normalized_array.astype(np.uint8)

                cv2.rectangle(normalized_array, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

                # GT
                gt_real = (gt[b].repeat(3,1,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                gt_real = cv2.UMat(gt_real)  # 转换为 UMat 格式
                cv2.rectangle(gt_real, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

                cv2.imshow('x', normalized_x_show)
                cv2.imshow('GT', gt_real)
                cv2.imshow('GT_gene', normalized_gt_image)
                cv2.imshow('pred', normalized_array)
                cv2.waitKey(0)



                # cv2.imwrite(os.path.join(file_path, 'ori.jpg'), normalized_x_show)
                # cv2.imwrite(os.path.join(file_path, 'GT.jpg'), gt_real)
                # cv2.imwrite(os.path.join(file_path, 'GT_gene.jpg'), normalized_gt_image)
                # cv2.imwrite(os.path.join(file_path, 'pred.jpg'), normalized_array)






        x = torch.cat([x, resized_tensor], dim=1)

        if self.training:
            x, gt = self.augmentations(x, gt)
        else:
            x, gt = self.augmentations_test(x, gt)

        gt = (gt > 0.5).int()



        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)

        return torch.sigmoid(output), gt, resized_tensor, gt_generate


if __name__ == "__main__":
    model = NestedUNet()

    x = torch.randn(1, 3, 224, 224)  # 输入一个 256x256 的单通道图像
    y,gt = model(x)
    print(y.shape)  # 输出应该也是 1x1x256x256

