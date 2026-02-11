#
# 增强损失函数模块 - 用于提升SSIM和PSNR指标
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np


class MultiScaleSSIMLoss(nn.Module):
    """多尺度SSIM损失 - 在不同尺度上计算SSIM，提升细节保留"""
    def __init__(self, window_size=11, num_scales=4, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.num_scales = num_scales
        self.size_average = size_average
        
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
                              for x in range(window_size)])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim_single_scale(self, img1, img2, window, window_size, channel):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        """
        Args:
            img1, img2: [C, H, W] or [B, C, H, W]
        """
        # 确保输入是4D
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        channel = img1.size(1)
        window = self.create_window(self.window_size, channel)
        
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        
        # 多尺度计算
        scales = []
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # 默认权重
        
        img1_scale = img1
        img2_scale = img2
        
        for scale in range(self.num_scales):
            if scale > 0:
                img1_scale = F.avg_pool2d(img1_scale, kernel_size=2)
                img2_scale = F.avg_pool2d(img2_scale, kernel_size=2)
            
            ssim_val = self.ssim_single_scale(img1_scale, img2_scale, window, self.window_size, channel)
            scales.append(ssim_val)
        
        # 加权平均
        loss = 0.0
        for i, ssim_val in enumerate(scales):
            weight = weights[i] if i < len(weights) else 1.0 / len(scales)
            loss += weight * (1.0 - ssim_val)
        
        return loss


class FrequencyDomainLoss(nn.Module):
    """频率域损失 - 在频域中计算损失，提升高频细节"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # 低频和高频损失的权重平衡
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: [C, H, W] or [B, C, H, W]
        """
        # 确保输入是4D
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        
        # FFT变换
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # 分离低频和高频
        # 低频：中心区域
        h, w = pred_fft.shape[-2:]
        center_h, center_w = h // 4, w // 4
        
        # 低频损失
        pred_low = pred_fft[..., :center_h, :center_w]
        target_low = target_fft[..., :center_h, :center_w]
        loss_low = F.mse_loss(torch.abs(pred_low), torch.abs(target_low))
        
        # 高频损失
        pred_high = pred_fft[..., center_h:, center_w:]
        target_high = target_fft[..., center_h:, center_w:]
        loss_high = F.mse_loss(torch.abs(pred_high), torch.abs(target_high))
        
        # 组合损失
        loss = self.alpha * loss_low + (1 - self.alpha) * loss_high
        
        return loss



class CombinedImageLoss(nn.Module):
    """组合图像损失 - 整合多种损失函数"""
    def __init__(self, ):
        super().__init__()

        weights = {
            'multiscale_ssim': 0.7,
            'frequency': 0.3,
        }
        self.weights = weights

        self.multiscale_ssim = MultiScaleSSIMLoss()

        self.frequency_loss = FrequencyDomainLoss()

    
    def forward(self, pred, target):
        total_loss = 0.0
        losses = {}

        loss_ms_ssim = self.multiscale_ssim(pred, target)
        total_loss += self.weights['multiscale_ssim'] * loss_ms_ssim
        losses['multiscale_ssim'] = loss_ms_ssim.item()

        loss_freq = self.frequency_loss(pred, target)
        total_loss += self.weights['frequency'] * loss_freq
        losses['frequency'] = loss_freq.item()
        
        return total_loss, losses

