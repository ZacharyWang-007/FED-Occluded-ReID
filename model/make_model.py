import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
import random
import torchvision
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, \
    deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import torch.nn.functional as F

from model.perceiver import trans_ffn1, trans_ffn2, trans_ffn3

def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def feature_search(memory, x, B, k=5, label=None, training=False):
    x = F.normalize(x, dim=1)
    memory = F.normalize(memory, dim=1)
    distmat = torch.matmul(x, memory.t())

    index = torch.topk(distmat, k=k, dim=1)[1]
    candidates = []
 
    for i in range(B):
        if training:
            if label[i] in index[i]:
                candidates.append(index[i, label[i] != index[i]].unsqueeze(dim=0))
            else:
                candidates.append(index[i][0:k-1].unsqueeze(dim=0))
        else:
            candidates.append(index[i][0:k-1].unsqueeze(dim=0))
   
    candidates = torch.cat(candidates, dim=0)
    latents = memory[candidates]
    return latents

def cosine_similarity(x):
    x_norm = F.normalize(x, dim=-1)
    dist = torch.matmul(x_norm, x_norm.permute(0, 1, 3, 2))
    dist = dist.mean(dim=-1, keepdim=True)

    return torch.sum(x * dist, dim=2)


# class Attention(nn.Module):
#     def __init__(self, dim=768, num_heads=[2, 4, 8, 16], qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., parts=4):
#         super().__init__()
#         self.parts = parts
#         self.num_heads = num_heads
#         head_dims = [dim//num_head for num_head in num_heads]
        
#         self.scale = [head_dim ** -0.5 for head_dim in head_dims]
        
#         self.q = nn.Linear(dim * self.parts, dim * self.parts, bias=qkv_bias)
#         self.kv = nn.Linear(dim * self.parts, dim * self.parts * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim * self.parts, dim * self.parts)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, memory, instances=4):
#         B, N, C = x.shape
       
        # q = self.q(x)
        # memory = memory.view(B, instances, -1)

        # kv = self.kv(memory)

        # recons_list = []
        # for i in range(len(self.num_heads)):
        #     temp_q = q.reshape(B, N, 1, self.num_heads[i], C // self.num_heads[i]).permute(2, 0, 3, 1, 4)
        #     temp_kv = kv.reshape(B, instances, 2, self.num_heads[i], C // self.num_heads[i]).permute(2, 0, 3, 1, 4)
        #     temp_k, temp_v = temp_kv[0], temp_kv[1]

            
        #     temp_attn = (temp_q @ temp_k.transpose(-2, -1)) * self.scale[i]
        #     temp_attn = temp_attn.softmax(dim=-1)
        #     temp_attn = self.attn_drop(temp_attn)

        #     temp_recons = (temp_attn @ temp_v).transpose(1, 2).reshape(B, N, C)

        #     temp_recons = self.proj(temp_recons)
        #     temp_recons = self.proj_drop(temp_recons)
        #     temp_recons = temp_recons.view(B, self.parts, -1)

        #     recons_list.append(temp_recons.unsqueeze(dim=2))

        # return torch.cat(recons_list, dim=2)

class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., parts=4):
        super().__init__()
        self.parts = parts
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim * self.parts, dim * self.parts, bias=qkv_bias)
        self.kv = nn.Linear(dim * self.parts, dim * self.parts * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * self.parts, dim * self.parts)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, memory, instances=4):
        B, N, C = x.shape
       
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        memory = memory.view(B, instances, -1)

        kv = self.kv(memory).reshape(B, instances, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        recons = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        recons = self.proj(recons)
        recons = self.proj_drop(recons)
        return recons


class Attention_mask(nn.Module):
    def __init__(self, dim=768, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return self.sigmoid(x)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'els':
            self.in_planes = 2048
            # self.base = ResNet(last_stride=last_stride,
            #                    block=Bottleneck,
            #                    layers=[3, 4, 6, 3])

            resnet50 = torchvision.models.resnet50(pretrained=True)
            resnet50.layer4[0].conv2.stride = [1, 1]
            resnet50.layer4[0].downsample[0].stride = [1, 1]
            self.base = nn.Sequential(*list(resnet50.children())[:-2])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        # if pretrain_choice == 'imagenet':
        #     self.base.load_param(model_path)
        #     print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

     

    def forward(self, x, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        self.backbone = 'els'
        self.instances, self.parts = 9, 4
        self.num_classes, self.ID_LOSS_TYPE = num_classes, cfg.MODEL.ID_LOSS_TYPE
        self.gap, self.gap_n = nn.AdaptiveAvgPool1d(1), nn.AdaptiveAvgPool1d(self.parts)

        # if self.backbone == 'res50':
        #     resnet50 = torchvision.models.resnet50(pretrained=True)
        #     resnet50.layer4[0].conv2.stride = [1, 1]
        #     resnet50.layer4[0].downsample[0].stride = [1, 1]
        #     self.base = nn.Sequential(*list(resnet50.children())[:-2])
        #     self.in_planes_ori = 2048
        #     self.in_planes = self.in_planes_ori // 4

        #     reduction_list = []
        #     for _ in range(self.parts):
        #         temp = nn.Sequential(
        #             nn.Linear(self.in_planes_ori, self.in_planes_ori//4),
        #             nn.BatchNorm1d(self.in_planes_ori//4),
        #             nn.LeakyReLU(0.1)
        #         )
        #         reduction_list.append(temp)
        #     self.reduction_list = nn.ModuleList(reduction_list)

        # else:
        #     self.base = build_model_swin()
        #     checkpoint = torch.load('/mnt/lustre/wangzhikang/Code/transformer_pretrain/swin_tiny_patch4_window7_224.pth', map_location='cpu')
        #     self.base.load_state_dict(checkpoint['model'], strict=True)
        #     self.in_planes = 768
        #     self.in_planes_ori = 768


        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        self.in_planes = 768
        self.in_planes_ori = 768




        self.cross_attention = Attention(parts=self.parts, dim=self.in_planes)

        mask_detection = []
        for i in range(self.parts):
            temp = nn.Sequential(
                nn.Linear(self.in_planes_ori, self.in_planes_ori//4),
                nn.LayerNorm(self.in_planes_ori//4),
                nn.Linear(self.in_planes_ori//4, 1),
                nn.Sigmoid()
            )
            mask_detection.append(temp)
        self.mask_detection = nn.ModuleList(mask_detection)

        # for token classifier
        self.classifier1 = nn.Linear(self.in_planes * self.parts, self.num_classes, bias=False)
        self.classifier1.apply(weights_init_classifier)

        self.classifier2 = nn.Linear(self.in_planes * self.parts, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)

        self.classifier3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier3.apply(weights_init_classifier)

        # this is bn list for stage 1
        bn_list = []
        for _ in range(self.parts):
            temp = nn.BatchNorm1d(self.in_planes)
            temp.bias.requires_grad_(False)
            temp.apply(weights_init_kaiming)
            bn_list.append(temp)
        self.bottleneck_local = nn.ModuleList(bn_list)

        # this is bn list for stage 2
        bn_list = []
        for _ in range(self.parts):
            temp = nn.BatchNorm1d(self.in_planes)
            temp.bias.requires_grad_(False)
            temp.apply(weights_init_kaiming)
            bn_list.append(temp)
        self.bottleneck_local_stage2 = nn.ModuleList(bn_list)

        # this is classifier_list
        fc_list = []
        for _ in range(self.parts):
            temp = nn.Linear(self.in_planes, self.num_classes, bias=False)
            temp.apply(weights_init_classifier)
            fc_list.append(temp)
        self.classifier_local = nn.ModuleList(fc_list)

        # for recons
        self.ffn2 = trans_ffn3(latent_dim=self.in_planes)
        self.ffn3 = trans_ffn3(latent_dim=self.in_planes)

    # def feature_extraction(self, x, B):
    #     x = self.base(x).view(B, self.in_planes_ori, -1)
    #     x = self.gap_n(x)
    #     return x

    def feature_extraction(self, x, B):
        x = self.base(x)
        x_global_feat, x_local_feat = x[0], x[1]
        x_local_feat = self.gap_n(x_local_feat.permute(0, 2, 1))
        return x_global_feat, x_local_feat

    def mask_generation(self, x1, x2, threshold=0.5, momentum=0.3):
        mask = torch.cosine_similarity(x1, x2)
        ones, zeros = torch.ones_like(mask), torch.zeros_like(mask) # + 0.3  # maybe not adopt zero, 
        min_ = torch.min(mask, dim=1)[0].mean()
        threshold = threshold * momentum + min_ * (1-momentum)
        return torch.where(mask>threshold, ones, zeros), threshold

    def mask_generation2(self, x, B, times=3):
        mask = torch.ones(B, self.parts).cuda()

        ones, zeros = torch.ones(B, 1).cuda(), torch.zeros(B, 1).cuda()
        mask_temp = torch.where(x>0, ones, zeros)

        for i in range(B):
            if torch.sum(mask_temp[i]) == 4:
                continue
            if torch.sum(mask_temp[i]) == 1:
                mask[i] -= mask_temp[i]
            else:
                output, index = torch.topk(x[i], k=2)
                output, index =  output.squeeze(), index.squeeze()
                if (output[0] / output[1]) > 3:
                    mask[i, index[0]] = 0
                else:
                    mask[i] -= mask_temp[i]
        return mask


    def mask_detection_function(self, x):
        mask = []
        
        for i in range(self.parts):
            mask.append(self.mask_detection[i](x[:, i, :]))

        return torch.cat(mask, dim=1)


    def forward(self, x, memory_only=False, labels=None, memory_features=None):
        B = x.size(0)
    
        if memory_only is True:
            x = self.feature_extraction(x, B)[1]
            feat_list = []
            for i in range(self.parts):
                if self.backbone=='res50':
                    feat_list.append(self.reduction_list[i](x[:, :, i]))
                else:
                    feat_list.append(self.bottleneck_local[i](x[:, :, i]))
            return torch.cat(feat_list, dim=-1)

        else:
            if self.training:
                mask = x[:, -1, :, :].view(B, 1, -1)
                mask = self.gap_n(mask).squeeze(dim=1)
                x = x[:, 0: 6, :, :]

                x = torch.chunk(x, 2, dim=1)
                x1, x2 = x[0], x[1]
                x1, x2 = self.feature_extraction(x1, B), self.feature_extraction(x2, B)

                x1_global, x1 = x1[0], x1[1]
                x2_global, x2 = x2[0], x2[1]

                cls_global_1 = self.classifier3(x1_global)
                cls_global_2 = self.classifier3(x2_global)

               
                # mask, self.threshold = self.mask_generation(x1.detach(), x2.detach(), threshold=self.threshold, momentum=self.momentum)
                mask = self.mask_generation2(mask, B=B)

                x1, x2 = x1.permute(0, 2, 1), x2.permute(0, 2, 1)
                x1_mask, x2_mask = self.mask_detection_function(x1).unsqueeze(dim=-1), self.mask_detection_function(x2).unsqueeze(dim=-1)

                # x1_mask, x2_mask = self.mask_detection(x1), self.mask_detection(x2) 
                x1, x2 = x1_mask * x1, x2_mask * x2

                x1_list, x2_list = [], []
                if self.backbone == 'res50':
                    for i in range(self.parts):
                        x1_list.append(self.reduction_list[i](x1[:, i, :]))
                        x2_list.append(self.reduction_list[i](x2[:, i, :]))
                else:
                    for i in range(self.parts):
                        x1_list.append(self.bottleneck_local[i](x1[:, i, :]))
                        x2_list.append(self.bottleneck_local[i](x2[:, i, :]))

                x1_bn, x2_bn = torch.cat(x1_list, dim=1), torch.cat(x2_list, dim=1)
                cls_1_0, cls_1_1 = self.classifier1(x1_bn), self.classifier1(x2_bn)

                # recons for x2
                latents1 = feature_search(memory_features, x1_bn, B=B, k=self.instances, label=labels, training=True)
                latents2 = feature_search(memory_features, x2_bn, B=B, k=self.instances, label=labels, training=True)

                x1_bn, x2_bn = x1_bn.view(B, self.parts, -1), x2_bn.view(B, self.parts, -1)
                # here, detach for x2
                
                x1_recons = self.cross_attention(x1_bn.view(B, -1).unsqueeze(dim=1), latents1, instances=self.instances-1).view(B, self.parts, -1)
                x2_recons = self.cross_attention(x2_bn.view(B, -1).unsqueeze(dim=1), latents2, instances=self.instances-1).view(B, self.parts, -1)

                # x1_recons = cosine_similarity(x1_recons)
                # x2_recons = cosine_similarity(x2_recons)

                x1_recons = self.ffn2(x1_recons)
                x2_recons = self.ffn2(x2_recons)

                # x1_recons = self.ffn3(x1_recons + x1_bn)
                # x2_recons = self.ffn3(x2_recons + x2_bn)
                x1_recons = self.ffn3(x1_recons * x1_mask + x1_bn)
                x2_recons = self.ffn3(x2_recons * x2_mask + x2_bn)

                x1_list = []
                x2_list = []

                cls_2_0, cls_2_1 = [], []

                for i in range(self.parts):
                    x1_list.append(self.bottleneck_local[i](x1_recons[:, i, :]))
                    x2_list.append(self.bottleneck_local[i](x2_recons[:, i, :]))

                    cls_2_0.append(self.classifier_local[i](x1_list[i]))
                    cls_2_1.append(self.classifier_local[i](x2_list[i]))

                x1_recons_bn = torch.cat(x1_list, dim=1)
                x2_recons_bn = torch.cat(x2_list, dim=1)

                cls_1 = self.classifier2(x1_recons_bn)
                cls_2 = self.classifier2(x2_recons_bn)

                return [cls_1_0, cls_1_1, cls_1, cls_2] + [cls_global_1, cls_global_2], x1_bn.view(B, -1), x2_bn.view(B, -1), x1_recons_bn, x2_recons_bn, x1_mask.squeeze(dim=-1), x2_mask.squeeze(dim=-1), mask

            else:
                x1 = self.feature_extraction(x, B=B)
                x1_global, x1 = x1[0], x1[1]

                x1 = x1.permute(0, 2, 1)
                x1_mask = self.mask_detection_function(x1).unsqueeze(dim=-1)
                x1 = x1_mask * x1

                x1_list = []
                if self.backbone == 'res50':
                    for i in range(self.parts):
                        x1_list.append(self.reduction_list[i](x1[:, i, :]))
                else:
                    for i in range(self.parts):
                        x1_list.append(self.bottleneck_local[i](x1[:, i, :]))

                x1_bn = torch.cat(x1_list, dim=1)
               
                latents1 = feature_search(memory_features, x1_bn, B=B, k=self.instances, label=labels)
                x1_bn = x1_bn.view(B, self.parts, -1)
                
                x1_recons = self.cross_attention(x1_bn.view(B, -1).unsqueeze(dim=1), latents1, instances=self.instances-1).view(B, self.parts, -1)
                x1_recons = self.ffn2(x1_recons)
                x1_recons = self.ffn3(x1_recons * x1_mask + x1_bn)
                x1_list = []

                for i in range(self.parts):
                    x1_list.append(self.bottleneck_local[i](x1_recons[:, i, :]))

                x1_recons_bn = torch.cat(x1_list, dim=1)
                return x1_bn.view(B, -1), x1_recons_bn
                

    def load_param(self, trained_path):

        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
        print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model


