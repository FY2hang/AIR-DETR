 ######################################## SPD-Conv start ########################################

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

######################################## SPD-Conv end ########################################

######################################## Omni-Kernel Network for Image Restoration [AAAI-24] start ########################################

class FGM(nn.Module):
    """恢复小目标优化的FGM - 包含高频增强"""
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

        # 恢复：高频保护参数（保护小目标细节）
        self.high_freq_boost = nn.Parameter(torch.ones(dim, 1, 1) * 0.5)

    def forward(self, x):
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        # ============ 恢复: 高频增强 ============
        # 计算频率坐标
        H, W = x2_fft.shape[-2:]
        freq_h = torch.fft.fftfreq(H, device=x.device).view(-1, 1)
        freq_w = torch.fft.fftfreq(W, device=x.device).view(1, -1)
        freq_magnitude = torch.sqrt(freq_h**2 + freq_w**2)
        
        # 高频增强mask（增强小目标相关的高频信息）
        high_freq_mask = (freq_magnitude > 0.3).float()  # 高频阈值
        high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)
        
        # 应用高频增强
        enhanced_fft = x2_fft * (1.0 + self.high_freq_boost * high_freq_mask)
        out = x1 * enhanced_fft
        # =====================================

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta

class OmniKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 79
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

        ### sca ###
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

       
        ### fca ###
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)

        # VisDrone边缘检测器：检测俯视车辆边缘
        self.visdrone_detector = nn.Sequential(
            nn.Conv2d(dim, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        # VisDrone基础权重（针对俯视角度优化）
        self.register_buffer('base_weights', torch.tensor([3.5, 3.5, 0.7, 0.1, 1.0]))
        # ==========================================

        # 自适应融合模块（替换原有的简单加权）
        # self.adaptive_fusion = AdaptiveFusionModule(dim)

    def forward(self, x):
        out = self.in_conv(x)

        ### fca ###
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
        x_fca = torch.abs(x_fca)

        ### fca ###
        ### sca ###
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca

        ### sca ###
        x_sca = self.fgm(x_sca)

        dw13_out = self.dw_13(out)
        dw31_out = self.dw_31(out)
        dw33_out = self.dw_33(out)
        dw11_out = self.dw_11(out)

        # ============ 只添加这几行 ============
        # 检测边缘强度（俯视车辆特征）
        edge_strength = self.visdrone_detector(out)
        # 权重归一化
        w = F.softmax(self.base_weights, dim=0)
        # =====================================

        # out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = (x + 
               w[0] * (1 + edge_strength) * self.dw_13(out) +     # 俯视车辆垂直边缘增强
               w[1] * (1 + edge_strength) * self.dw_31(out) +     # 俯视车辆水平边缘增强
               w[2] * (1 - 0.5 * edge_strength) * self.dw_33(out) + # 边缘区域减少大感受野
               w[3] * self.dw_11(out) + 
               w[4] * x_sca)
        # 自适应智能融合
        # out = self.adaptive_fusion(out, dw13_out, dw31_out, dw33_out, dw11_out, x_sca)

        out = self.act(out)
        return self.out_conv(out)
        
class CSPOmniKernel(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        ok_branch, identity = torch.split(self.cv1(x), [int(x.size(1) * self.e), int(x.size(1) * (1 - self.e))], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))

######################################## Omni-Kernel Network for Image Restoration [AAAI-24] end ########################################

