aimport torch
import torch.nn as nn
import torch.nn.functional as F

'''
class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
'''


class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dirate):
        super(REBNCONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1 * dirate,
                                    dilation=1 * dirate,  # 这里相当于：融合了空洞卷积的深度可分离卷积
                                    groups=in_ch)
        # self.bn1 = nn.BatchNorm2d(in_ch)
        # self.relu_s1 = nn.ReLU(inplace=True)

        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    # dilation=1*dirate,
                                    groups=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu_s2 = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.depth_conv(input)
        # out=self.bn1(out)
        # out = self.relu_s1(out)

        out = self.point_conv(out)
        out = self.bn2(out)
        out = self.relu_s2(out)
        return out



## upsample tensor 'src' to have the same spatial size with tensor 'tar'
# #把src上采样到和tar一样的尺寸
def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    #F.upsample和nn.Upsample区别是前者不能作为 nn.Sequential中的一层
    return src

from torch.nn.parameter import Parameter
'''eca'''
class se_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(se_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


'''se'''
# class se_layer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(se_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):
    # En_1内部结构
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1) 
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)  

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.se = se_layer(channel=out_ch)
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)  

        hx1 = self.rebnconv1(hxin)  

        hx = self.pool1(hx1) 
        hx2 = self.rebnconv2(hx)  

        hx = self.pool2(hx2)  
        hx3 = self.rebnconv3(hx) 

        hx = self.pool3(hx3)  
        hx4 = self.rebnconv4(hx)  

        hx = self.pool4(hx4)  
        hx5 = self.rebnconv5(hx) 

        hx = self.pool5(hx5)  
        hx6 = self.rebnconv6(hx)  

        hx7 = self.rebnconv7(hx6)  

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))  

        hx6dup = _upsample_like(hx6d, hx5)  
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))  

        hx5dup = _upsample_like(hx5d, hx4)  
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))  

        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))

        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))

        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        hx1d_eca = self.se(hx1d)
        return hx1d_eca + hxin   # 将最后的输出和最开始的输入融合



### RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.se= se_layer(channel=out_ch)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        hx1d_eca = self.se(hx1d)
        return hx1d_eca + hxin



### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.se =se_layer(channel=out_ch)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        hx1d_se = self.se(hx1d)
        return hx1d_se + hxin


### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.se= se_layer(channel=out_ch)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        hx1d_se = self.se(hx1d)
        return hx1d_se + hxin


### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.se = se_layer(channel=out_ch)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        hx1d_se = self.se(hx1d)
        return hx1d_se + hxin


##### U^2-Net_3P ####
class U2NET_3P(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET_3P, self).__init__()
        En_in_channel = [3, 32, 32, 64, 64]
        En_mid_channel =[16, 16, 32, 32, 64]
        En_out_channel = [32, 32, 64, 64, 128]
        # En_1
        self.stage1 = RSU7(in_ch, En_mid_channel[0], En_out_channel[0])
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # En_2
        self.stage2 = RSU6(En_in_channel[1], En_mid_channel[1], En_out_channel[1])
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(En_in_channel[2], En_mid_channel[2], En_out_channel[2])
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(En_in_channel[3], En_mid_channel[3], En_out_channel[3])
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(En_in_channel[4], En_mid_channel[4], En_out_channel[4])
        # self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        #
        # self.stage6 = RSU4F(En_in_channel[5],En_mid_channel[5], En_out_channel[5])

        # -------Decoder-----------


        De_out_channel=[128,64,64,32]
        # De_in_channel = [sum(En_out_channel), sum(En_out_channel[0:4]) + De_out_channel[0],
        #                  sum(En_out_channel[0:3]) + De_out_channel[1],
        #                  sum(En_out_channel[0:2]) + De_out_channel[2],
        #                  En_out_channel[0] + De_out_channel[3]]
                       #De_in_channel [496, 368, 176, 80, 32]
        De_in_channel=[320,256,128,96]
        De_mid_channel=[64,32,32,16]

        # --------------------stage 5--------------
        # En1->De5
        # 16倍下采样
        # self.en1_de5_mp = nn.MaxPool2d(16, 16, ceil_mode=True)


        # En2->De5
        # 8倍下采样
        # self.en2_de5_mp = nn.MaxPool2d(8, 8, ceil_mode=True)



        # En3->De5
        # 4倍下采样
        # self.en3_de5_mp = nn.MaxPool2d(4, 4, ceil_mode=True)



        # En4->De5
        # 2倍下采样
        # self.en4_de5_mp = nn.MaxPool2d(2, 2, ceil_mode=True)



        # En5->De5
        # self.en5_de5_conv = nn.Conv2d(in_channels=En_out_channel[4], out_channels=self.cat_channel, kernel_size=3,
        #                               padding=1)


        # En6—>De5
        # 上采样2倍
        # self.en6_de5_up = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.en6_de5_up =_upsample_like()


        # fusion  En1~6  ->De5
        # fusion的时候应该用RSU模块才对
        # self.conv_En1_6_to_De5 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        # self.bn_En1_6_to_De5 = nn.BatchNorm2d(self.UpChannels)
        # self.relu_En1_6_to_De5 = nn.ReLU(inplace=True)

       # self.stage5d = RSU4F(De_in_channel[0], De_mid_channel[0], De_out_channel[0])

        # --------------------stage 4--------------
        # En1->De4
        # 8倍下采样
        self.en1_de4_mp = nn.MaxPool2d(8, 8, ceil_mode=True)

        # En2->De4
        # 4倍下采样
        self.en2_de4_mp = nn.MaxPool2d(4, 4, ceil_mode=True)


        # En3->De4
        # 2倍下采样
        self.en3_de4_mp = nn.MaxPool2d(2, 2, ceil_mode=True)


        # En4->De4

        # De5->De4
        # 上采样2倍
        self.de5_de4_up = nn.Upsample(scale_factor=2, mode='bilinear')

        # En6—>De4
        # 上采样4倍
        # self.en6_de4_up = nn.Upsample(scale_factor=4, mode='bilinear')

         self.stage4d = RSU4(De_in_channel[0], De_mid_channel[0], De_out_channel[0])

        # --------------------stage 3--------------
        # En1->De3
        # 4倍下采样
        self.en1_de3_mp = nn.MaxPool2d(4, 4, ceil_mode=True)


        # En2->De3
        # 2倍下采样
        self.en2_de3_mp = nn.MaxPool2d(2, 2, ceil_mode=True)


        # En3->De3
        # self.en3_de3_conv = nn.Conv2d(in_channels=En_out_channel[2], out_channels=self.cat_channel, kernel_size=3,
        #                               padding=1)


        # De4->De3
        # 2倍上采样
        self.de4_de3_up = nn.Upsample(scale_factor=2, mode='bilinear')

        # De5->De3
        # 上采样4倍
        # self.de5_de3_up = nn.Upsample(scale_factor=4, mode='bilinear')

        # En6—>De3
        # 上采样8倍
        # self.en6_de3_up = nn.Upsample(scale_factor=8, mode='bilinear')

        self.stage3d = RSU5(De_in_channel[1], De_mid_channel[1], De_out_channel[1])

        # --------------------stage 2--------------
        # En1->De2
        # 2倍下采样
        self.en1_de2_mp = nn.MaxPool2d(2, 2, ceil_mode=True)


        # En2->De2
        # self.en2_de2_conv = nn.Conv2d(in_channels=En_out_channel[1], out_channels=self.cat_channel, kernel_size=3,
        #                               padding=1)

        # De3->De2
        # 2倍上采样
        self.de3_de2_up = nn.Upsample(scale_factor=2, mode='bilinear')

        # De4->De2
        # 4倍上采样
        # self.de4_de2_up = nn.Upsample(scale_factor=4, mode='bilinear')

        # De5->De2
        # 上采样8倍
        # self.de5_de2_up = nn.Upsample(scale_factor=8, mode='bilinear')

        self.stage2d = RSU6(De_in_channel[2], De_mid_channel[2], De_out_channel[2])

        # --------------------stage 1--------------
        # En1->De1

        # De2->De1
        # 2倍上采样
        self.de2_de1_up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.stage1d = RSU7(De_in_channel[3], De_mid_channel[3], De_out_channel[3])



        # side_output
        self.outconv1 = nn.Conv2d(De_out_channel[3], out_ch, 3, padding=1)# 64->1
        self.outconv2 = nn.Conv2d(De_out_channel[2], out_ch, 3, padding=1)# 64->1
        self.outconv3 = nn.Conv2d(De_out_channel[1], out_ch, 3, padding=1)# 128->1
        self.outconv4 = nn.Conv2d(De_out_channel[0], out_ch, 3, padding=1)# 256->1
        self.outconv5 = nn.Conv2d(En_out_channel[-1], out_ch, 3, padding=1)# 512->1
        # self.outconv6 = nn.Conv2d(En_out_channel[-1], out_ch, 3, padding=1)# 512->1

        #最后的输出
        self.outconv = nn.Conv2d(5*out_ch, out_ch, 3, padding=1)


    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)  # stage 1为第一个小U结构
        hx = self.pool12(hx1)  # 下采样 为进入第二个小U结构准备

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        '''
        hx = self.pool56(hx5)
        # print('hx5',hx5.shape)
        # stage 6
        hx6 = self.stage6(hx)  # stage6后面没有池化层
        '''
        # print('hx6',hx6.shape)
       # hx6up = _upsample_like(hx6, hx5)


        # -------------------- decoder --------------------
        '''
        # stage5
        en1_de5 = self.en1_de5_mp(hx1)
        # print('en1_de5.shape',en1_de5.shape)
        en2_de5 = self.en2_de5_mp(hx2)
        # print("en2_de5.shape",en1_de5.shape)
        en3_de5 = self.en3_de5_mp(hx3)
        # print("en3_de5.shape",en3_de5.shape)
        en4_de5 = self.en4_de5_mp(hx4)
        # print("en4_de5.shape",en4_de5.shape)
        en5_de5 = hx5
        # print("en5_de5.shape",en5_de5.shape)
        # en6_de5 = self.en6_de5_up(hx6,hx5)
        en6_de5=_upsample_like(hx6,hx5)#上采样时用定义好的  _upsample_like，不要用 nn.Upsample
        # print("en6_de5.shape",en6_de5.shape)

        fuse5 = self.stage5d(torch.cat((en1_de5,en2_de5,en3_de5,en4_de5,en5_de5,en6_de5),1))
        '''
        # stage4
        en1_de4 = self.en1_de4_mp(hx1)
        en2_de4 = self.en2_de4_mp(hx2)
        en3_de4 = self.en3_de4_mp(hx3)
        en4_de4 = hx4
        # de5_de4 = self.de5_de4_up(fuse5)
        de5_de4=_upsample_like(hx5,hx4)
        # en6_de4 = self.en6_de4_up(hx6)
        fuse4= self.stage4d(torch.cat((en1_de4,en2_de4,en3_de4,en4_de4,de5_de4),1))
        # stage3
        en1_de3 = self.en1_de3_mp(hx1)
        # print("en1_de3.shape",en1_de3.shape)
        en2_de3 = self.en2_de3_mp(hx2)
        # print("en2_de3.shape",en2_de3.shape)
        en3_de3 = hx3
        # print("en3_de3.shape",en3_de3.shape)
        # de4_de3 = self.de4_de3_up(fuse4)
        de4_de3=_upsample_like(fuse4,hx3)
        # print("de4_de3.shape",de4_de3.shape)
        # de5_de3 = self.de5_de3_up(fuse5)
        # en6_de3 = self.en6_de3_up(hx6)
        stage_3d=torch.cat((en1_de3,en2_de3,en3_de3,de4_de3),1)
        print(stage_3d.shape)
        fuse3 = self.stage3d(torch.cat((en1_de3,en2_de3,en3_de3,de4_de3),1))

        # stage2
        en1_de2 = self.en1_de2_mp(hx1)
        en2_de2 = hx2
        # de3_de2 = self.de3_de2_up(fuse3)
        de3_de2=_upsample_like(fuse3,hx2)
        # de4_de2 = self.de4_de2_up(fuse4)
        # de5_de2 = self.de5_de2_up(fuse5)
        # en6_de2 = self.en6_de2_up(hx6)
        fuse2 = self.stage2d(torch.cat((en1_de2,en2_de2,de3_de2),1))

         #stage1
        en1_de1 = hx1
        # de2_de1 = self.de2_de1_up(fuse2)
        de2_de1=_upsample_like(fuse2,hx1)
        # print("de2_de1.shape",de2_de1.shape)
        # de3_de1 = self.de3_de1_up(fuse3)
        # de4_de1 = self.de4_de1_up(fuse4)
        # de5_de1 = self.de5_de1_up(fuse5)
        # en6_de1 = self.en6_de1_up(hx6)
        fuse1 = self.stage1d(torch.cat((en1_de1,de2_de1),1))

        d1 = self.outconv1(fuse1) # 256

       # d6 = self.outconv6(hx6)
        #d6 = _upsample_like(d6,d1)

        d5 = self.outconv5(hx5)
        d5 = _upsample_like(d5,d1) # 16->256

        d4 = self.outconv4(fuse4)
        d4 = _upsample_like(d4,d1) # 32->256

        d3 = self.outconv3(fuse3)
        d3 = _upsample_like(d3,d1)  # 64->256

        d2 = self.outconv2(fuse2)
        d2 = _upsample_like(d2,d1)  # 128->256

        d0=self.outconv(torch.cat((d1,d2,d3,d4,d5),1))

        return F.sigmoid(d0),F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5)



# #可视化网络
# import hiddenlayer as hl
# mynet=U2NET(3,1)
# hl_graph=hl.build_graph(mynet,torch.zeros([1,3,227,227]))
# hl_graph.theme=hl.graph.THEMES['blue'].copy()
# #保存图片
# hl_graph.save('C:\\11.png',format='png')


# from torchviz import make_dot
# mynet=U2NET(3,1)
# x=torch.randn(1,3,512,512).requires_grad_(True)
# y=mynet(x)
# my=make_dot(y,params=dict(list(mynet.named_parameters())+[('x',x)]))
# my.format='png'
# my.directory='C:\\12'
# my.view()
#
# from torchstat import stat
# mynet=U2NET_3P(3,1)
# # 导入模型，输入一张输入图片的尺寸
# # stat(mynet, (3, 288 ,288))
# stat(mynet, (3, 288 ,288))
# # print(mynet)

