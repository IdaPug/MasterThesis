import torch 
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DinoAdapterLarger(nn.Module):
    
    def __init__(self, in_channels=384, out_channels=128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.adapter(x)
    
class DinoAdapterSmall(nn.Module):
    
    def __init__(self, in_channels=384, out_channels=128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.adapter(x)

class UnetDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels, out_size=(256, 256)):
        super().__init__()
        self.out_size = out_size

        self.upconv1 = nn.ConvTranspose2d(in_channels, base_channels, kernel_size=2, stride=2)
        self.conv1   = ConvBlock(base_channels, base_channels)

        self.upconv2 = nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=2, stride=2)
        self.conv2   = ConvBlock(base_channels // 2, base_channels // 2)

        self.upconv3 = nn.ConvTranspose2d(base_channels // 2, base_channels // 4, kernel_size=2, stride=2)
        self.conv3   = ConvBlock(base_channels // 4, base_channels // 4)

        self.upconv4 = nn.ConvTranspose2d(base_channels // 4, base_channels // 8, kernel_size=2, stride=2)
        self.conv4   = ConvBlock(base_channels // 8, base_channels // 8)

        self.final_conv = nn.Conv2d(base_channels // 8, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.conv1(x)

        x = self.upconv2(x)
        x = self.conv2(x)

        x = self.upconv3(x)
        x = self.conv3(x)

        x = self.upconv4(x)
        x = self.conv4(x)

        x = self.final_conv(x)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        return x

class VanillaUNet(nn.Module):
    def __init__(self, in_channels, num_classes, out_size=(256, 256)):
        super().__init__()
        self.out_size = out_size

        # Encoder
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.poo1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(in_channels=64, out_channels=128)
        self.poo2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(in_channels=128, out_channels=256)
        self.poo3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(in_channels=256, out_channels=512)
        self.poo4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = ConvBlock(in_channels=512, out_channels=1024)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1   = ConvBlock(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2   = ConvBlock(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3   = ConvBlock(256, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4   = ConvBlock(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        poo1 = self.poo1(enc1)

        enc2 = self.enc2(poo1)
        poo2 = self.poo2(enc2)

        enc3 = self.enc3(poo2)
        poo3 = self.poo3(enc3)

        enc4 = self.enc4(poo3)
        poo4 = self.poo4(enc4)

        # bottleneck
        bottleneck = self.bottleneck(poo4)

        # Decoder with skip connections
        up1 = self.upconv1(bottleneck)
        merge1 = torch.cat([up1, enc4], dim=1)
        dec1 = self.conv1(merge1)

        up2 = self.upconv2(dec1)
        merge2 = torch.cat([up2, enc3], dim=1)
        dec2 = self.conv2(merge2)

        up3 = self.upconv3(dec2)
        merge3 = torch.cat([up3, enc2], dim=1)
        dec3 = self.conv3(merge3)

        up4 = self.upconv4(dec3)
        merge4 = torch.cat([up4, enc1], dim=1)
        dec4 = self.conv4(merge4)

        x = self.final_conv(dec4)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        return x
    
    
class AdapterUnetDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels, out_size=(256, 256)):
        super().__init__()
        self.out_size = out_size

        # Adapter to reduce DINO feature channels before decoding
        #self.adapter = DinoAdapterLarger(in_channels=in_channels, out_channels=base_channels)
        self.adapter = DinoAdapterSmall(in_channels=in_channels, out_channels=base_channels)

        self.upconv1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.conv1   = ConvBlock(base_channels, base_channels)

        self.upconv2 = nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=2, stride=2)
        self.conv2   = ConvBlock(base_channels // 2, base_channels // 2)

        self.upconv3 = nn.ConvTranspose2d(base_channels // 2, base_channels // 4, kernel_size=2, stride=2)
        self.conv3   = ConvBlock(base_channels // 4, base_channels // 4)

        self.upconv4 = nn.ConvTranspose2d(base_channels // 4, base_channels // 8, kernel_size=2, stride=2)
        self.conv4   = ConvBlock(base_channels // 8, base_channels // 8)

        self.final_conv = nn.Conv2d(base_channels // 8, num_classes, kernel_size=1)

    def forward(self, x):
        # 🔹 DINO features -> Adapter -> Decoder
        x = self.adapter(x)

        x = self.upconv1(x)
        x = self.conv1(x)

        x = self.upconv2(x)
        x = self.conv2(x)

        x = self.upconv3(x)
        x = self.conv3(x)

        x = self.upconv4(x)
        x = self.conv4(x)

        x = self.final_conv(x)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        return x


class DINOv3Encoder(nn.Module):
    def __init__(self, dinov3_model, n_layers=4):
        super().__init__()
        self.dino = dinov3_model
        self.n_layers = n_layers

        for p in self.dino.parameters():
            p.requires_grad = False

    def forward(self, x):
        feats = self.dino.get_intermediate_layers(
            x,
            n=self.n_layers,
            reshape=True,
            norm=0,
            return_class_token=False
        )
        # feats[-1] is deepest, earlier are skip connections
        deep = feats[-1]
        return deep, feats[:-1]

class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DINOv3UNetDecoder(nn.Module):
    def __init__(self, embed_dim=384, num_skips=3, base_ch=256,
                 num_classes=2, out_size=(256, 256)):
        """
        embed_dim: ViT embedding dim (384 for ViT-S/16)
        num_skips: number of skip connections (n-1 from get_intermediate_layers)
        base_ch:   starting decoder channels
        num_classes: segmentation output channels
        out_size:  final spatial size (HxW)
        """
        super().__init__()
        self.out_size = out_size
        self.num_skips = num_skips

        proj_channels = [base_ch // (2 ** (i + 1)) for i in range(num_skips)][::-1]
        print(f"Projection channels for skips: {proj_channels}")

        self.skip_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, proj_ch, kernel_size=1)
            for proj_ch in proj_channels
        ])


        # progressive decoder path
        self.up1 = nn.ConvTranspose2d(embed_dim, base_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBlock2(base_ch + proj_channels[-1], base_ch)

        self.up2 = nn.ConvTranspose2d(base_ch, base_ch // 2, kernel_size=2, stride=2)
        self.conv2 = ConvBlock2(base_ch // 2 + proj_channels[-2], base_ch // 2)

        self.up3 = nn.ConvTranspose2d(base_ch // 2, base_ch // 4, kernel_size=2, stride=2)
        self.conv3 = ConvBlock2(base_ch // 4 + proj_channels[-3], base_ch // 4)

        # optional fourth upsample (if you want 4× scaling to full size)
        self.up4 = nn.ConvTranspose2d(base_ch // 4, base_ch // 8, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(base_ch // 8, base_ch // 8)

        self.final_conv = nn.Conv2d(base_ch // 8, num_classes, kernel_size=1)

    def forward(self, deep_feature, skip_features):
        """
        deep_feature: deepest transformer output (B,384,H/16,W/16)
        skip_features: list of previous features [f11, f10, f9] (same spatial size)
        """
        # project skips to smaller dim
        skip_features = [proj(s) for proj, s in zip(self.skip_projs, skip_features)]

        # print shapes of skip features
        # for i, skip in enumerate(skip_features):
        #     print(f"Skip feature {i} shape: {skip.shape}")

        # stage 1
        x = self.up1(deep_feature)
        skip1 = F.interpolate(skip_features[-1], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv1(x)

        # stage 2
        x = self.up2(x)
        skip2 = F.interpolate(skip_features[-2], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv2(x)

        # stage 3
        x = self.up3(x)
        skip3 = F.interpolate(skip_features[-3], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv3(x)

        # stage 4 (no skip, just refine)
        x = self.up4(x)
        x = self.conv4(x)

        x = self.final_conv(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x






class DINOv3UNetDecoderAdapter(nn.Module):
    def __init__(self, embed_dim=384, num_skips=3, base_ch=256,
                 num_classes=2, out_size=(256, 256)):
        """
        embed_dim: ViT embedding dim (384 for ViT-S/16)
        num_skips: number of skip connections (n-1 from get_intermediate_layers)
        base_ch:   starting decoder channels
        num_classes: segmentation output channels
        out_size:  final spatial size (HxW)
        """
        super().__init__()
        self.out_size = out_size
        self.num_skips = num_skips

        proj_channels = [base_ch // (2 ** (i + 1)) for i in range(num_skips)]
        print(f"Projection channels for skips: {proj_channels}")

        self.skip_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, proj_ch, kernel_size=1)
            for proj_ch in proj_channels
        ])

        # make adapter
        self.adapter = DinoAdapterSmall(in_channels=embed_dim, out_channels=base_ch)

        # progressive decoder path
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBlock2(base_ch + proj_channels[-1], base_ch)

        self.up2 = nn.ConvTranspose2d(base_ch, base_ch // 2, kernel_size=2, stride=2)
        self.conv2 = ConvBlock2(base_ch // 2 + proj_channels[-2], base_ch // 2)

        self.up3 = nn.ConvTranspose2d(base_ch // 2, base_ch // 4, kernel_size=2, stride=2)
        self.conv3 = ConvBlock2(base_ch // 4 + proj_channels[-3], base_ch // 4)

        # optional fourth upsample (if you want 4× scaling to full size)
        self.up4 = nn.ConvTranspose2d(base_ch // 4, base_ch // 8, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(base_ch // 8, base_ch // 8)

        self.final_conv = nn.Conv2d(base_ch // 8, num_classes, kernel_size=1)

    def forward(self, deep_feature, skip_features):
        """
        deep_feature: deepest transformer output (B,384,H/16,W/16)
        skip_features: list of previous features [f11, f10, f9] (same spatial size)
        """
        # project skips to smaller dim
        skip_features = [proj(s) for proj, s in zip(self.skip_projs, skip_features)]

        # print shapes of skip features
        for i, skip in enumerate(skip_features):
            print(f"Skip feature {i} shape: {skip.shape}")

        # pass deep feature through adapter
        deep_feature = self.adapter(deep_feature)

        # stage 1
        x = self.up1(deep_feature)
        skip1 = F.interpolate(skip_features[-1], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv1(x)

        # stage 2
        x = self.up2(x)
        skip2 = F.interpolate(skip_features[-2], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv2(x)

        # stage 3
        x = self.up3(x)
        skip3 = F.interpolate(skip_features[-3], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv3(x)

        # stage 4 (no skip, just refine)
        x = self.up4(x)
        x = self.conv4(x)

        x = self.final_conv(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x



