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
        # project skips to smaller dim. 
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



class DINOv3UNetDecoderAlternative(nn.Module):
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
        self.conv1 = ConvBlock2(base_ch , base_ch)

        self.up2 = nn.ConvTranspose2d(base_ch, base_ch // 2, kernel_size=2, stride=2)
        self.conv2 = ConvBlock2(base_ch // 2 + proj_channels[-1], base_ch // 2)

        self.up3 = nn.ConvTranspose2d(base_ch // 2, base_ch // 4, kernel_size=2, stride=2)
        self.conv3 = ConvBlock2(base_ch // 4 + proj_channels[-2], base_ch // 4)

        # optional fourth upsample (if you want 4× scaling to full size)
        self.up4 = nn.ConvTranspose2d(base_ch // 4, base_ch // 8, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(base_ch // 8 + proj_channels[-3], base_ch // 8)

        self.final_conv = nn.Conv2d(base_ch // 8, num_classes, kernel_size=1)

    def forward(self, deep_feature, skip_features):
        """
        deep_feature: deepest transformer output (B,384,H/16,W/16)
        skip_features: list of previous features [f11, f10, f9] (same spatial size)
        """
        # project skips to smaller dim. 
        skip_features = [proj(s) for proj, s in zip(self.skip_projs, skip_features)]

        # print shapes of skip features
        # for i, skip in enumerate(skip_features):
        #     print(f"Skip feature {i} shape: {skip.shape}")

        # stage 1
        x = self.up1(deep_feature)
        x = self.conv1(x)

       
        # stage 2
        x = self.up2(x)
        skip1 = F.interpolate(skip_features[-1], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv2(x)

        
        # stage 3
        x = self.up3(x)
        skip2 = F.interpolate(skip_features[-2], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3(x)

        
        x = self.up4(x)
        skip3 = F.interpolate(skip_features[-3], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv4(x)

        x = self.final_conv(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x



class DINOv3UNetDecoderAlternative2(nn.Module):
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
        self.up1 = nn.ConvTranspose2d(embed_dim, base_ch//2, kernel_size=2, stride=2)
        self.conv1 = ConvBlock2(base_ch//2 + proj_channels[-1], base_ch//2)

        self.up2 = nn.ConvTranspose2d(base_ch//2, base_ch // 4, kernel_size=2, stride=2)
        self.conv2 = ConvBlock2(base_ch // 4 + proj_channels[-2], base_ch // 4)

        self.up3 = nn.ConvTranspose2d(base_ch // 4, base_ch // 8, kernel_size=2, stride=2)
        self.conv3 = ConvBlock2(base_ch // 8 + proj_channels[-3], base_ch // 8)

        # optional fourth upsample (if you want 4× scaling to full size)
        self.up4 = nn.ConvTranspose2d(base_ch // 8, base_ch // 16, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(base_ch // 16, base_ch // 16)

        self.final_conv = nn.Conv2d(base_ch // 16, num_classes, kernel_size=1)

    def forward(self, deep_feature, skip_features):
        """
        deep_feature: deepest transformer output (B,384,H/16,W/16)
        skip_features: list of previous features [f11, f10, f9] (same spatial size)
        """
        # project skips to smaller dim. 
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


class DINOv3UNetDecoderAlternative3(nn.Module):
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

        proj_channels = [base_ch // (2 ** ( i)) for i in range(num_skips)][::-1]
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
        # project skips to smaller dim. 
        skip_features = [proj(s) for proj, s in zip(self.skip_projs, skip_features)]

        # print shapes of skip features
        # for i, skip in enumerate(skip_features):
        #     print(f"Skip feature {i} shape: {skip.shape}")
        print(deep_feature.shape)
        # stage 1
        x = self.up1(deep_feature)
        skip1 = F.interpolate(skip_features[-1], size=x.shape[2:], mode="bilinear", align_corners=False)
        print(f"After up1: {x.shape}, skip1: {skip1.shape}")


        x = torch.cat([x, skip1], dim=1)
        x = self.conv1(x)

        print(f"After conv1: {x.shape}")

        # stage 2
        x = self.up2(x)
        skip2 = F.interpolate(skip_features[-2], size=x.shape[2:], mode="bilinear", align_corners=False)
        print(f"After up2: {x.shape}, skip2: {skip2.shape}")

        x = torch.cat([x, skip2], dim=1)
        x = self.conv2(x)
        print(f"After conv2: {x.shape}")

        # stage 3
        x = self.up3(x)
        skip3 = F.interpolate(skip_features[-3], size=x.shape[2:], mode="bilinear", align_corners=False)
        print(f"After up3: {x.shape}, skip3: {skip3.shape}")

        x = torch.cat([x, skip3], dim=1)
        x = self.conv3(x)
        print(f"After conv3: {x.shape}")

        # stage 4 (no skip, just refine)
        x = self.up4(x)
        x = self.conv4(x)
        print(f"After conv4: {x.shape}")

        x = self.final_conv(x)
        print(f"After final_conv: {x.shape}")
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x



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
        # project skips to smaller dim. 
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


class CrossSliceAttention(nn.Module):
    def __init__(self, C, K, num_heads=4,  return_attn=False):
        super().__init__()
        assert C % num_heads == 0
        self.C = C
        self.K = K
        self.num_heads = num_heads
        self.head_dim = C // num_heads

        self.qkv = nn.Linear(C, C * 3)
        self.out_proj = nn.Linear(C, C)

        self.ln = nn.LayerNorm(C)

        # Positional embedding per slice
        self.pos_emb = nn.Parameter(torch.randn(1, K, C))

        self.return_attn = return_attn

    def forward(self, x, mask=None):
        B, K, C, H, W = x.shape

        # reshape to (BHW, K, C)
        x = x.permute(0, 3, 4, 1, 2).reshape(B*H*W, K, C)

        # add slice positional embedding
        x = x + self.pos_emb[:, :K, :]

        # LN
        x = self.ln(x)

        # (BHW, K, 3C) → q,k,v
        qkv = self.qkv(x).reshape(B*H*W, K, 3, C).permute(2,0,1,3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # split heads
        q = q.reshape(-1, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(-1, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(-1, K, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # mask
        if mask is not None:
            mask = mask[:, None, None, :].expand(B, self.num_heads, 1, K)
            mask = mask.repeat_interleave(H*W, 0)
            scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # (BHW, heads, K, head_dim)
        out = out.transpose(1, 2).reshape(B*H*W, K, C)
        out = self.out_proj(out)

        out = out.reshape(B, H, W, K, C).permute(0, 3, 4, 1, 2)

        if self.return_attn:
            return out, attn

        return out


class SliceAttentionPool(nn.Module):
    """
    Computes attention weights over the K slices and returns 
    a weighted sum across slices.
    
    Input:  x (B, K, C, H, W)
    Output: y (B, C, H, W)
    """
    def __init__(self, C, K):
        super().__init__()
        
        # 1x1 conv to produce a scalar score per slice
        # Input: (B*K, C, H, W) → Output: (B*K, 1, H, W)
        self.score_layer = nn.Conv2d(C, 1, kernel_size=1)

    def forward(self, x):
        B, K, C, H, W = x.shape

        # Reshape slices into batch dimension
        xf = x.reshape(B * K, C, H, W)

        # Compute per-slice scores (B*K, 1, H, W)
        scores = self.score_layer(xf)
        scores = scores.view(B, K, 1, H, W)

        # Softmax across slice dimension K
        attn = F.softmax(scores, dim=1)  # (B, K, 1, H, W)

        # Weighted sum across slices
        out = (attn * x).sum(dim=1)      # (B, C, H, W)
        return out




class DINOv3UNetDecoderWithAttention(nn.Module):
    def __init__(self, embed_dim=384, num_skips=3, base_ch=256,
                 num_classes=2, out_size=(256, 256), num_heads=4, K=5, apply_attention_on_skips=False, pooling_method='avg'):
        super().__init__()
        self.out_size = out_size
        self.num_skips = num_skips
        self.K = K
        self.apply_attention_on_skips = apply_attention_on_skips
        self.pooling_method = pooling_method

        # print pooling method
        print(f"Using pooling method: {pooling_method}")

        if pooling_method == 'attn':
            self.slice_pool = SliceAttentionPool(C=embed_dim, K=K)
            self.skip_slice_pool = nn.ModuleList([
            SliceAttentionPool(C=embed_dim, K=K) for _ in range(num_skips)
            ])

        proj_channels = [base_ch // (2 ** (i + 1)) for i in range(num_skips)][::-1]
        print(f"Projection channels for skips: {proj_channels}")

        self.skip_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, proj_ch, kernel_size=1)
            for proj_ch in proj_channels
        ])

        self.cross_slice_attn = CrossSliceAttention(C=embed_dim, K=K, num_heads=num_heads)

        if apply_attention_on_skips:
            self.skipatten = nn.ModuleList([
                CrossSliceAttention(C=embed_dim, K=K, num_heads=num_heads)
                for _ in range(num_skips)
            ])
        else:
            self.skipatten = None

        # Progressive decoder path
        self.up1 = nn.ConvTranspose2d(embed_dim, base_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBlock2(base_ch + proj_channels[-1], base_ch)

        self.up2 = nn.ConvTranspose2d(base_ch, base_ch // 2, kernel_size=2, stride=2)
        self.conv2 = ConvBlock2(base_ch // 2 + proj_channels[-2], base_ch // 2)

        self.up3 = nn.ConvTranspose2d(base_ch // 2, base_ch // 4, kernel_size=2, stride=2)
        self.conv3 = ConvBlock2(base_ch // 4 + proj_channels[-3], base_ch // 4)

        self.up4 = nn.ConvTranspose2d(base_ch // 4, base_ch // 8, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(base_ch // 8, base_ch // 8)

        self.final_conv = nn.Conv2d(base_ch // 8, num_classes, kernel_size=1)

    def forward(self, deep_feature, skip_features):
        B, K, C, H, W = deep_feature.shape
        assert K == self.K, f"Expected K={self.K} slices, got K={K}"

        # Cross-slice attention on deep feature
        x = self.cross_slice_attn(deep_feature)


        # Optionally cross-slice attention on skips
        if self.apply_attention_on_skips and self.skipatten is not None:
            skip_features = [atten(s) for atten, s in zip(self.skipatten, skip_features)]

        # Take center slice after attention
        # center = K // 2
        # x = x[:, center, :, :, :]
        # skip_features = [s[:, center, :, :, :] for s in skip_features]
        if self.pooling_method == 'avg':
            x = x.mean(dim=1)
            if self.apply_attention_on_skips:
                skip_features = [s.mean(dim=1) for s in skip_features]
        elif self.pooling_method == 'attn':
            x = self.slice_pool(x)
            if self.apply_attention_on_skips:
                skip_features = [self.skip_slice_pool[i](s) for i, s in enumerate(skip_features)]
        elif self.pooling_method == 'none':
            center = K // 2
            x = x[:, center, :, :, :]
            if self.apply_attention_on_skips:
                skip_features = [s[:, center, :, :, :] for s in skip_features]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # Project skips
        skip_features = [proj(s) for proj, s in zip(self.skip_projs, skip_features)]

        # ------------------ Decoder ------------------
        x = self.up1(x)
        skip1 = F.interpolate(skip_features[-1], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        skip2 = F.interpolate(skip_features[-2], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        skip3 = F.interpolate(skip_features[-3], size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = self.conv4(x)

        x = self.final_conv(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F



class DINOv3UNetEncodeDecoder(nn.Module):
    def __init__(self, in_channels, embed_dim, dinov3_model, n_layers=4, num_classes=2, out_size=(256, 256)):
        super().__init__()

        self.out_size = out_size
        self.embed_dim = embed_dim

        # ------------------------------------------
        # DINO Encoder
        # ------------------------------------------
        self.encoder = DINOv3Encoder(dinov3_model=dinov3_model, n_layers=n_layers)

        # DINO produces: B, C_dino, Hd, Wd
        # We project it down to match the UNet bottleneck input channels
        # CNN bottleneck input = embed_dim // 2
        self.dino_proj = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1)
 

        # ------------------------------------------
        # CNN Encoder (UNet-style)
        # ------------------------------------------
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim // 16)
        self.poo1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(embed_dim // 16, embed_dim // 8)
        self.poo2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(embed_dim // 8, embed_dim // 4)
        self.poo3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(embed_dim // 4, embed_dim // 2)
        self.poo4 = nn.MaxPool2d(2)

        # Bottleneck of the CNN
        # Input channels: embed_dim // 2  (from CNN)
        # After concatenation with projected DINO → embed_dim total
        self.bottleneck = ConvBlock(embed_dim, embed_dim)

        # ------------------------------------------
        # UNet Decoder
        # ------------------------------------------
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
        self.conv1   = ConvBlock(embed_dim, embed_dim // 2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2)
        self.conv2   = ConvBlock(embed_dim // 2, embed_dim // 4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
        self.conv3   = ConvBlock(embed_dim // 4, embed_dim // 8)

        self.upconv4 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2)
        self.conv4   = ConvBlock(embed_dim // 8, embed_dim // 16)

        self.final_conv = nn.Conv2d(embed_dim // 16, num_classes, 1)

    def forward(self, x):

        # -----------------------------
        # CNN Encoder
        # -----------------------------
        enc1 = self.enc1(x)
        poo1 = self.poo1(enc1)

        enc2 = self.enc2(poo1)
        poo2 = self.poo2(enc2)

        enc3 = self.enc3(poo2)
        poo3 = self.poo3(enc3)

        enc4 = self.enc4(poo3)
        poo4 = self.poo4(enc4)  # shape: B, embed_dim//2, H, W

        # -----------------------------
        # DINO deep features
        # -----------------------------
        dino_deep, _ = self.encoder(x)
        # dino_deep: B, embed_dim, Hd, Wd
        

        # Match spatial size with poo4
        _, _, H, W = poo4.shape
        dino_resized = F.interpolate(dino_deep, size=(H, W), mode='bilinear', align_corners=False)

        # Project DINO channels: embed_dim → embed_dim // 2
        #dino_proj = self.dino_proj(dino_resized)

        # -----------------------------
        # Fusion at Bottleneck
        # -----------------------------
        # poo4 has embed_dim//2 channels
        # dino_proj has embed_dim//2 channels
        # concatenated → embed_dim channels
        #bottleneck_input = torch.cat([poo4, dino_proj], dim=1)
        bottleneck_input = torch.cat([poo4, dino_resized], dim=1)

        bneck = self.bottleneck(bottleneck_input)

        # -----------------------------
        # Decoder
        # -----------------------------
        up1 = self.upconv1(bneck)
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

        # Final segmentation output
        x = self.final_conv(dec4)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)

        return x



class DINOv3UNetEncodeDecoderWithSkips(nn.Module):
    def __init__(self, in_channels, embed_dim, dinov3_model, n_layers=4, num_classes=2, out_size=(256, 256)):
        super().__init__()

        self.out_size = out_size
        self.embed_dim = embed_dim

        # ------------------------------------------
        # DINO Encoder
        # ------------------------------------------
        self.encoder = DINOv3Encoder(dinov3_model=dinov3_model, n_layers=n_layers)

        self.dproj1 = nn.Conv2d(embed_dim, embed_dim // 16, 1)
        self.dproj2 = nn.Conv2d(embed_dim, embed_dim // 8, 1)
        self.dproj3 = nn.Conv2d(embed_dim, embed_dim // 4, 1)
        self.dproj4 = nn.Conv2d(embed_dim, embed_dim // 2, 1)
        
        # ------------------------------------------
        # CNN Encoder (UNet-style)
        # ------------------------------------------
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim // 16)
        self.poo1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(embed_dim // 16, embed_dim // 8)
        self.poo2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(embed_dim // 8, embed_dim // 4)
        self.poo3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(embed_dim // 4, embed_dim // 2)
        self.poo4 = nn.MaxPool2d(2)

        # Bottleneck of the CNN
        # Input channels: embed_dim // 2  (from CNN)
        # After concatenation with projected DINO → embed_dim total
        self.bottleneck = ConvBlock(embed_dim // 2, embed_dim)

        # ------------------------------------------
        # UNet Decoder
        # ------------------------------------------
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
        self.conv1   = ConvBlock(embed_dim // 2 + embed_dim, embed_dim // 2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2)
        self.conv2   = ConvBlock(embed_dim // 4 + embed_dim // 2, embed_dim // 4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
        self.conv3   = ConvBlock(embed_dim // 8 + embed_dim // 4, embed_dim // 8)

        self.upconv4 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2)
        self.conv4   = ConvBlock(embed_dim // 16 + embed_dim // 8, embed_dim // 16)

        self.final_conv = nn.Conv2d(embed_dim // 16, num_classes, 1)

    def forward(self, x):

        # -----------------------------
        # CNN Encoder
        # -----------------------------
        enc1 = self.enc1(x)
        poo1 = self.poo1(enc1)

        enc2 = self.enc2(poo1)
        poo2 = self.poo2(enc2)

        enc3 = self.enc3(poo2)
        poo3 = self.poo3(enc3)

        enc4 = self.enc4(poo3)
        poo4 = self.poo4(enc4)  # shape: B, embed_dim//2, H, W

        # -----------------------------
        # DINO deep features
        # -----------------------------
        dino_deep, skips = self.encoder(x)
        # dino_deep: B, embed_dim, Hd, Wd
        d1 = skips[0]  # shallowest
        d2 = skips[1]
        d3 = skips[2]
        d4 = dino_deep  # deepest

        d1_res = F.interpolate(d1, size=enc1.shape[2:], mode="bilinear", align_corners=False)
        d1_res = self.dproj1(d1_res)

        # Level 2 skip (UNet e2 at H/2)
        d2_res = F.interpolate(d2, size=enc2.shape[2:], mode="bilinear", align_corners=False)
        d2_res = self.dproj2(d2_res)

        # Level 3 skip (UNet e3 at H/4)
        d3_res = F.interpolate(d3, size=enc3.shape[2:], mode="bilinear", align_corners=False)
        d3_res = self.dproj3(d3_res)

        # Level 4 skip (UNet e4 at H/8)
        d4_res = F.interpolate(d4, size=enc4.shape[2:], mode="bilinear", align_corners=False)
        d4_res = self.dproj4(d4_res)

        # print shapes of dino skips
        print(f"DINO skip shapes after projection: {d1_res.shape}, {d2_res.shape}, {d3_res.shape}, {d4_res.shape}")
    

        s1 = torch.cat([enc1, d1_res], dim=1)
        s2 = torch.cat([enc2, d2_res], dim=1)
        s3 = torch.cat([enc3, d3_res], dim=1)
        s4 = torch.cat([enc4, d4_res], dim=1)
        # print shapes of concatenated skips
        #print(f"Concatenated skip shapes: {s1.shape}, {s2.shape}, {s3.shape}, {s4.shape}")


        bneck = self.bottleneck(poo4)

        # -----------------------------
        # Decoder
        # -----------------------------
        up1 = self.upconv1(bneck)
        #print(f"Up1 shape: {up1.shape}, S4 shape: {s4.shape}")
        merge1 = torch.cat([up1, s4], dim=1)
        #print(f"Merge1 shape: {merge1.shape}")

        dec1 = self.conv1(merge1)

        up2 = self.upconv2(dec1)
        merge2 = torch.cat([up2, s3], dim=1)
        dec2 = self.conv2(merge2)

        up3 = self.upconv3(dec2)
        merge3 = torch.cat([up3, s2], dim=1)
        dec3 = self.conv3(merge3)

        up4 = self.upconv4(dec3)
        merge4 = torch.cat([up4, s1], dim=1)
        dec4 = self.conv4(merge4)

        # Final segmentation output
        x = self.final_conv(dec4)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)

        return x



class DINOv3UNetEncodeDecoderV2(nn.Module):
    def __init__(self, in_channels, dinov3_model,dino_dim, embed_dim=1024, n_layers=4, num_classes=2, out_size=(256, 256)):
        super().__init__()

        self.out_size = out_size
        self.embed_dim = embed_dim
        self.dino_dim = dino_dim

        # ------------------------------------------
        # DINO Encoder
        # ------------------------------------------
        self.encoder = DINOv3Encoder(dinov3_model=dinov3_model, n_layers=n_layers)

        # DINO produces: B, C_dino, Hd, Wd
        # We project it down to match the UNet bottleneck input channels
        # CNN bottleneck input = embed_dim // 2
        #self.dino_proj = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1)

        bneckdino_proj_channels = embed_dim + dino_dim
        self.bneckdino_proj = nn.Conv2d(bneckdino_proj_channels, embed_dim , kernel_size=1)

        # define projection from embed_dim to embed_dim - dino_dim
        self.unetdownproj = nn.Conv2d(embed_dim, embed_dim - dino_dim, kernel_size=1)

        # ------------------------------------------
        # CNN Encoder (UNet-style)
        # ------------------------------------------
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim // 16)
        self.poo1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(embed_dim // 16, embed_dim // 8)
        self.poo2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(embed_dim // 8, embed_dim // 4)
        self.poo3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(embed_dim // 4, embed_dim // 2)
        self.poo4 = nn.MaxPool2d(2)

        # Bottleneck of the CNN
        # Input channels: embed_dim // 2  (from CNN)
        # After concatenation with projected DINO → embed_dim total
        self.bottleneck = ConvBlock(embed_dim // 2, embed_dim)

        # ------------------------------------------
        # UNet Decoder
        # ------------------------------------------
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
        self.conv1   = ConvBlock(embed_dim, embed_dim // 2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2)
        self.conv2   = ConvBlock(embed_dim // 2, embed_dim // 4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
        self.conv3   = ConvBlock(embed_dim // 4, embed_dim // 8)

        self.upconv4 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2)
        self.conv4   = ConvBlock(embed_dim // 8, embed_dim // 16)

        self.final_conv = nn.Conv2d(embed_dim // 16, num_classes, 1)

    def forward(self, x):

        # -----------------------------
        # CNN Encoder
        # -----------------------------
        enc1 = self.enc1(x)
        poo1 = self.poo1(enc1)

        enc2 = self.enc2(poo1)
        poo2 = self.poo2(enc2)

        enc3 = self.enc3(poo2)
        poo3 = self.poo3(enc3)

        enc4 = self.enc4(poo3)
        poo4 = self.poo4(enc4)  # shape: B, embed_dim//2, H, W

        # -----------------------------
        # DINO deep features
        # -----------------------------
        dino_deep, _ = self.encoder(x)
        # dino_deep: B, embed_dim, Hd, Wd


        bneck = self.bottleneck(poo4)
        
        # Match spatial size with poo4
        _, _, H, W = bneck.shape
        dino_resized = F.interpolate(dino_deep, size=(H, W), mode='bilinear', align_corners=False)

        # normalize everything
        #dino_resized = F.normalize(dino_resized, p=2, dim=1)
        #bneck = F.normalize(bneck, p=2, dim=1)
        bneck = self.unetdownproj(bneck)


        fused = torch.cat([bneck, dino_resized], dim=1)

        #bneck = self.bneckdino_proj(fused)

        # -----------------------------
        # Decoder
        # -----------------------------
        up1 = self.upconv1(fused)
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

        # Final segmentation output
        x = self.final_conv(dec4)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)

        return x


class DINOv3UNetDecoderWithUpsampling(nn.Module):
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

        self.up4 = nn.ConvTranspose2d(base_ch // 4, base_ch // 8, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(base_ch // 8, base_ch // 8)

        self.final_conv = nn.Conv2d(base_ch // 8, num_classes, kernel_size=1)

    def forward(self, img,deep_feature, upsampFeatures):
        """
        deep_feature: deepest transformer output (B,384,H/16,W/16)
        skip_features: list of previous features [f11, f10, f9] (same spatial size)
        """
        skip_features = []
        # unpack the upsampled features
        for i in range(1,self.num_skips+1):
            skip_features.append(upsampFeatures[i])

        # project skips to smaller dim
        skip_features = [proj(s) for proj, s in zip(self.skip_projs, skip_features)]
        # print shapes of projected skip features
        # for i, s in enumerate(skip_features):
        #     print(f"Projected skip feature {i} shape: {s.shape}")

        # stage 1
        x = self.up1(deep_feature)
        # print shape of x and skip_features[-1]
        #print(f"Decoder stage 1 - x shape: {x.shape}, skip shape: {skip_features[-1].shape}")
        #skip1 = F.interpolate(skip_features[-1], size=x.shape[2:], mode="bilinear", align_corners=False)
        #skip1 = self.upsampler(img, skip_features[-1], output_size=self.skip_sizes[-1])
        x = torch.cat([x, skip_features[-1]], dim=1)
        x = self.conv1(x)

        # stage 2
        x = self.up2(x)
        #skip2 = self.upsampler(img, skip_features[-2], output_size=self.skip_sizes[-2])
        x = torch.cat([x, skip_features[-2]], dim=1)
        x = self.conv2(x)

        # stage 3
        x = self.up3(x)
        x = torch.cat([x, skip_features[-3]], dim=1)
        x = self.conv3(x)

        # stage 4 (no skip, just refine)
        x = self.up4(x)
        x = self.conv4(x)

        x = self.final_conv(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x








# class DINOv3UNetEncodeDecoderV2SkipsAnyUp(nn.Module):
#     def __init__(self, upsampler, in_channels,dinov3_model,dino_dim, embed_dim=1024 , n_layers=4, num_classes=2, out_size=(256, 256)):
#         super().__init__()

#         self.out_size = out_size
#         self.embed_dim = embed_dim
#         self.dino_dim = dino_dim
#         self.upsampler = upsampler

#         # ------------------------------------------
#         # DINO Encoder
#         # ------------------------------------------
#         self.encoder = DINOv3Encoder(dinov3_model=dinov3_model, n_layers=n_layers)

#         self.dproj1 = nn.Conv2d(dino_dim+embed_dim // 16, embed_dim // 16, 1)
#         self.dproj2 = nn.Conv2d(dino_dim+embed_dim // 8, embed_dim // 8, 1)
#         self.dproj3 = nn.Conv2d(dino_dim+embed_dim // 4, embed_dim // 4, 1)
#         self.dproj4 = nn.Conv2d(dino_dim+embed_dim // 2, embed_dim // 2, 1)

#         # get spacial sizes for each skip connection based on out_size
#         self.skip_sizes = []
#         H, W = out_size
#         self.skip_sizes.append((H, W))
#         for i in range(len(n_layers)-1):
#             H = H // 2
#             W = W // 2
#             self.skip_sizes.append((H, W))
#             print(f"Skip connection {i} spatial size: {self.skip_sizes[i]}")
        
#         # ------------------------------------------
#         # CNN Encoder (UNet-style)
#         # ------------------------------------------
#         self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim // 16)
#         self.poo1 = nn.MaxPool2d(2)

#         self.enc2 = ConvBlock(embed_dim // 16, embed_dim // 8)
#         self.poo2 = nn.MaxPool2d(2)

#         self.enc3 = ConvBlock(embed_dim // 8, embed_dim // 4)
#         self.poo3 = nn.MaxPool2d(2)

#         self.enc4 = ConvBlock(embed_dim // 4, embed_dim // 2)
#         self.poo4 = nn.MaxPool2d(2)

#         # Bottleneck of the CNN
#         # Input channels: embed_dim // 2  (from CNN)
#         # After concatenation with projected DINO → embed_dim total
#         self.bottleneck = ConvBlock(embed_dim // 2, embed_dim)

#         # ------------------------------------------
#         # UNet Decoder
#         # ------------------------------------------
#         self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
#         self.conv1   = ConvBlock(embed_dim  , embed_dim // 2)

#         self.upconv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2)
#         self.conv2   = ConvBlock(embed_dim // 2, embed_dim // 4)

#         self.upconv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
#         self.conv3   = ConvBlock(embed_dim // 4, embed_dim // 8)

#         self.upconv4 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2)
#         self.conv4   = ConvBlock(embed_dim // 8, embed_dim // 16)

#         self.final_conv = nn.Conv2d(embed_dim // 16, num_classes, 1)

#     def forward(self, x):

#         # -----------------------------
#         # CNN Encoder
#         # -----------------------------
#         enc1 = self.enc1(x)
#         poo1 = self.poo1(enc1)

#         enc2 = self.enc2(poo1)
#         poo2 = self.poo2(enc2)

#         enc3 = self.enc3(poo2)
#         poo3 = self.poo3(enc3)

#         enc4 = self.enc4(poo3)
#         poo4 = self.poo4(enc4)  # shape: B, embed_dim//2, H, W

#         # -----------------------------
#         # DINO deep features
#         # -----------------------------
#         dino_deep, skips = self.encoder(x)
#         # dino_deep: B, embed_dim, Hd, Wd
#         d1 = skips[0]  # shallowest
#         d2 = skips[1]
#         d3 = skips[2]
#         d4 = dino_deep  # deepest

        
#         #d1_res = self.dproj1(d1)
#         d1_res = self.upsampler(x, d1, output_size=self.skip_sizes[0])

#         # Level 2 skip (UNet e2 at H/2)
#         #d2_res = self.dproj2(d2)
#         d2_res = self.upsampler(x, d2, output_size=self.skip_sizes[1])

#         # Level 3 skip (UNet e3 at H/4)
#         #d3_res = self.dproj3(d3)
#         d3_res = self.upsampler(x, d3, output_size=self.skip_sizes[2])

#         # Level 4 skip (UNet e4 at H/8)
#         #d4_res = self.dproj4(d4)
#         d4_res = self.upsampler(x, d4, output_size=self.skip_sizes[3])

#         # print shapes of dino skips
#         #print(f"DINO skip shapes after projection: {d1_res.shape}, {d2_res.shape}, {d3_res.shape}, {d4_res.shape}")
#         #print(f"UNet encoder skip shapes: {enc1.shape}, {enc2.shape}, {enc3.shape}, {enc4.shape}")

#         s1 = torch.cat([enc1, d1_res], dim=1)
#         s2 = torch.cat([enc2, d2_res], dim=1)
#         s3 = torch.cat([enc3, d3_res], dim=1)
#         s4 = torch.cat([enc4, d4_res], dim=1)
#         # print shapes of concatenated skips
#         #print(f"Concatenated skip shapes: {s1.shape}, {s2.shape}, {s3.shape}, {s4.shape}")

#         # proj to get the correct channel size for skips
#         s1 = self.dproj1(s1)
#         s2 = self.dproj2(s2)
#         s3 = self.dproj3(s3)
#         s4 = self.dproj4(s4)

#         bneck = self.bottleneck(poo4)

#         # -----------------------------
#         # Decoder
#         # -----------------------------
#         up1 = self.upconv1(bneck)
#         #print(f"Up1 shape: {up1.shape}, S4 shape: {s4.shape}")
#         merge1 = torch.cat([up1, s4], dim=1)
#         #print(f"Merge1 shape: {merge1.shape}")

#         dec1 = self.conv1(merge1)

#         up2 = self.upconv2(dec1)
#         merge2 = torch.cat([up2, s3], dim=1)
#         dec2 = self.conv2(merge2)

#         up3 = self.upconv3(dec2)
#         merge3 = torch.cat([up3, s2], dim=1)
#         dec3 = self.conv3(merge3)

#         up4 = self.upconv4(dec3)
#         merge4 = torch.cat([up4, s1], dim=1)
#         dec4 = self.conv4(merge4)

#         # Final segmentation output
#         x = self.final_conv(dec4)
#         x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)

#         return x



class DINOv3UNetEncodeDecoderV2SkipsAnyUp(nn.Module):
    def __init__(self, in_channels,dino_dim, embed_dim=1024 , num_classes=2, out_size=(256, 256)):
        super().__init__()

        self.out_size = out_size
        self.embed_dim = embed_dim
        self.dino_dim = dino_dim
  

        # ------------------------------------------
        # DINO Encoder
        # ------------------------------------------

        self.dproj1 = nn.Conv2d(dino_dim+embed_dim // 16, embed_dim // 16, 1)
        self.dproj2 = nn.Conv2d(dino_dim+embed_dim // 8, embed_dim // 8, 1)
        self.dproj3 = nn.Conv2d(dino_dim+embed_dim // 4, embed_dim // 4, 1)
        self.dproj4 = nn.Conv2d(dino_dim+embed_dim // 2, embed_dim // 2, 1)

 
        # ------------------------------------------
        # CNN Encoder (UNet-style)
        # ------------------------------------------
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim // 16)
        self.poo1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(embed_dim // 16, embed_dim // 8)
        self.poo2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(embed_dim // 8, embed_dim // 4)
        self.poo3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(embed_dim // 4, embed_dim // 2)
        self.poo4 = nn.MaxPool2d(2)

        # Bottleneck of the CNN
        # Input channels: embed_dim // 2  (from CNN)
        # After concatenation with projected DINO → embed_dim total
        self.bottleneck = ConvBlock(embed_dim // 2, embed_dim)

        # ------------------------------------------
        # UNet Decoder
        # ------------------------------------------
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
        self.conv1   = ConvBlock(embed_dim  , embed_dim // 2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2)
        self.conv2   = ConvBlock(embed_dim // 2, embed_dim // 4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
        self.conv3   = ConvBlock(embed_dim // 4, embed_dim // 8)

        self.upconv4 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2)
        self.conv4   = ConvBlock(embed_dim // 8, embed_dim // 16)

        self.final_conv = nn.Conv2d(embed_dim // 16, num_classes, 1)

    def forward(self, x,features):

        # -----------------------------
        # CNN Encoder
        # -----------------------------
        enc1 = self.enc1(x)
        poo1 = self.poo1(enc1)

        enc2 = self.enc2(poo1)
        poo2 = self.poo2(enc2)

        enc3 = self.enc3(poo2)
        poo3 = self.poo3(enc3)

        enc4 = self.enc4(poo3)
        poo4 = self.poo4(enc4)  # shape: B, embed_dim//2, H, W

        # -----------------------------
        # DINO deep features
        # -----------------------------
        
        # dino_deep: B, embed_dim, Hd, Wd
        d1 = features[0]  # shallowest
        d2 = features[1]
        d3 = features[2]
        d4 = features[3]



        # print shapes of dino skips
        #print(f"DINO skip shapes after projection: {d1_res.shape}, {d2_res.shape}, {d3_res.shape}, {d4_res.shape}")
        #print(f"UNet encoder skip shapes: {enc1.shape}, {enc2.shape}, {enc3.shape}, {enc4.shape}")
        
        s1 = torch.cat([enc1, d1], dim=1)
        s2 = torch.cat([enc2, d2], dim=1)
        s3 = torch.cat([enc3, d3], dim=1)
        s4 = torch.cat([enc4, d4], dim=1)
        # print shapes of concatenated skips
        #print(f"Concatenated skip shapes: {s1.shape}, {s2.shape}, {s3.shape}, {s4.shape}")

        # proj to get the correct channel size for skips
        s1 = self.dproj1(s1)
        s2 = self.dproj2(s2)
        s3 = self.dproj3(s3)
        s4 = self.dproj4(s4)

        bneck = self.bottleneck(poo4)

        # -----------------------------
        # Decoder
        # -----------------------------
        up1 = self.upconv1(bneck)
        #print(f"Up1 shape: {up1.shape}, S4 shape: {s4.shape}")
        merge1 = torch.cat([up1, s4], dim=1)
        #print(f"Merge1 shape: {merge1.shape}")

        dec1 = self.conv1(merge1)

        up2 = self.upconv2(dec1)
        merge2 = torch.cat([up2, s3], dim=1)
        dec2 = self.conv2(merge2)

        up3 = self.upconv3(dec2)
        merge3 = torch.cat([up3, s2], dim=1)
        dec3 = self.conv3(merge3)

        up4 = self.upconv4(dec3)
        merge4 = torch.cat([up4, s1], dim=1)
        dec4 = self.conv4(merge4)

        # Final segmentation output
        x = self.final_conv(dec4)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)

        return x






class DINOv3UNetEncodeDecoderSkipsAnyUp_final(nn.Module):
    def __init__(self, in_channels,dino_dim, embed_dim=1024 , num_classes=2, out_size=(256, 256)):
        super().__init__()

        self.out_size = out_size
        self.embed_dim = embed_dim
        self.dino_dim = dino_dim
  

        # ------------------------------------------
        # DINO Encoder
        # ------------------------------------------

        self.dproj1 = nn.Conv2d(dino_dim+embed_dim // 16, embed_dim // 16, 1)
        self.dproj2 = nn.Conv2d(dino_dim+embed_dim // 8, embed_dim // 8, 1)
        self.dproj3 = nn.Conv2d(dino_dim+embed_dim // 4, embed_dim // 4, 1)
        self.dproj4 = nn.Conv2d(dino_dim+embed_dim // 2, embed_dim // 2, 1)

 
        # ------------------------------------------
        # CNN Encoder (UNet-style)
        # ------------------------------------------
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim // 16)
        self.poo1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(embed_dim // 16, embed_dim // 8)
        self.poo2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(embed_dim // 8, embed_dim // 4)
        self.poo3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(embed_dim // 4, embed_dim // 2)
        self.poo4 = nn.MaxPool2d(2)

        # Bottleneck of the CNN
        # Input channels: embed_dim // 2  (from CNN)
        # After concatenation with projected DINO → embed_dim total
        self.bottleneck = ConvBlock(embed_dim, embed_dim)

        # ------------------------------------------
        # UNet Decoder
        # ------------------------------------------
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
        self.conv1   = ConvBlock(embed_dim  , embed_dim // 2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2)
        self.conv2   = ConvBlock(embed_dim // 2, embed_dim // 4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
        self.conv3   = ConvBlock(embed_dim // 4, embed_dim // 8)

        self.upconv4 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2)
        self.conv4   = ConvBlock(embed_dim // 8, embed_dim // 16)

        self.final_conv = nn.Conv2d(embed_dim // 16, num_classes, 1)

    def forward(self, x,features):

        # -----------------------------
        # CNN Encoder
        # -----------------------------
        enc1 = self.enc1(x)
        poo1 = self.poo1(enc1)

        enc2 = self.enc2(poo1)
        poo2 = self.poo2(enc2)

        enc3 = self.enc3(poo2)
        poo3 = self.poo3(enc3)

        enc4 = self.enc4(poo3)
        poo4 = self.poo4(enc4)  # shape: B, embed_dim//2, H, W

        # -----------------------------
        # DINO deep features
        # -----------------------------
        
        # dino_deep: B, embed_dim, Hd, Wd
        d1 = features[0]  # shallowest
        d2 = features[1]
        d3 = features[2]
        d4 = features[3]



        # print shapes of dino skips
        #print(f"DINO skip shapes after projection: {d1_res.shape}, {d2_res.shape}, {d3_res.shape}, {d4_res.shape}")
        #print(f"UNet encoder skip shapes: {enc1.shape}, {enc2.shape}, {enc3.shape}, {enc4.shape}")
        
        s1 = torch.cat([enc1, d1], dim=1)
        s2 = torch.cat([enc2, d2], dim=1)
        s3 = torch.cat([enc3, d3], dim=1)
        s4 = torch.cat([enc4, d4], dim=1)
        # print shapes of concatenated skips
        #print(f"Concatenated skip shapes: {s1.shape}, {s2.shape}, {s3.shape}, {s4.shape}")

        # proj to get the correct channel size for skips
        s1 = self.dproj1(s1)
        s2 = self.dproj2(s2)
        s3 = self.dproj3(s3)
        s4 = self.dproj4(s4)

        bneck = self.bottleneck(poo4)

        # -----------------------------
        # Decoder
        # -----------------------------
        up1 = self.upconv1(bneck)
        #print(f"Up1 shape: {up1.shape}, S4 shape: {s4.shape}")
        merge1 = torch.cat([up1, s4], dim=1)
        #print(f"Merge1 shape: {merge1.shape}")

        dec1 = self.conv1(merge1)

        up2 = self.upconv2(dec1)
        merge2 = torch.cat([up2, s3], dim=1)
        dec2 = self.conv2(merge2)

        up3 = self.upconv3(dec2)
        merge3 = torch.cat([up3, s2], dim=1)
        dec3 = self.conv3(merge3)

        up4 = self.upconv4(dec3)
        merge4 = torch.cat([up4, s1], dim=1)
        dec4 = self.conv4(merge4)

        # Final segmentation output
        x = self.final_conv(dec4)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)

        return x


class DINOv3UNetEncodeDecoderV1SkipsUpsampling(nn.Module):
    def __init__(self, in_channels, dino_dim,embed_dim, dinov3_model, n_layers=4, num_classes=2, out_size=(256, 256)):
        super().__init__()

        self.out_size = out_size
        self.embed_dim = embed_dim
        self.dino_dim = dino_dim

        # ------------------------------------------
        # DINO Encoder
        # ------------------------------------------
        self.encoder = DINOv3Encoder(dinov3_model=dinov3_model, n_layers=n_layers)

        self.dinoskip1_proj = nn.Conv2d(dino_dim, embed_dim // 16, kernel_size=1)
        self.dinoskip2_proj = nn.Conv2d(dino_dim, embed_dim // 8, kernel_size=1)
        self.dinoskip3_proj = nn.Conv2d(dino_dim, embed_dim // 4, kernel_size=1)
        self.dinoskip4_proj = nn.Conv2d(dino_dim, embed_dim // 2, kernel_size=1)

        # ------------------------------------------
        # CNN Encoder (UNet-style)
        # ------------------------------------------
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim // 16)
        self.poo1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(embed_dim // 16, embed_dim // 8)
        self.poo2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(embed_dim // 8, embed_dim // 4)
        self.poo3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(embed_dim // 4, embed_dim // 2)
        self.poo4 = nn.MaxPool2d(2)

        # Bottleneck of the CNN
        # Input channels: embed_dim // 2  (from CNN)
        # After concatenation with projected DINO → embed_dim total
        self.bottleneck = ConvBlock(embed_dim, embed_dim)

        # ------------------------------------------
        # UNet Decoder
        # ------------------------------------------
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
        self.conv1   = ConvBlock(embed_dim, embed_dim // 2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2)
        self.conv2   = ConvBlock(embed_dim // 2, embed_dim // 4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
        self.conv3   = ConvBlock(embed_dim // 4, embed_dim // 8)

        self.upconv4 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2)
        self.conv4   = ConvBlock(embed_dim // 8, embed_dim // 16)

        self.final_conv = nn.Conv2d(embed_dim // 16, num_classes, 1)

    def forward(self, x,upsampledFeature):

        d1 = upsampledFeature[0]  # shallowest
        d2 = upsampledFeature[1]
        d3 = upsampledFeature[2]
        d4 = upsampledFeature[3]

        # -----------------------------
        # CNN Encoder
        # -----------------------------
        enc1 = self.enc1(x)
        poo1 = self.poo1(enc1)

        enc2 = self.enc2(poo1)
        poo2 = self.poo2(enc2)

        enc3 = self.enc3(poo2)
        poo3 = self.poo3(enc3)

        enc4 = self.enc4(poo3)
        poo4 = self.poo4(enc4)  # shape: B, embed_dim//2, H, W

        # -----------------------------
        # DINO deep features
        # -----------------------------
        dino_deep, _ = self.encoder(x)
        # dino_deep: B, embed_dim, Hd, Wd

        # Match spatial size with poo4
        _, _, H, W = poo4.shape
        dino_resized = F.interpolate(dino_deep, size=(H, W), mode='bilinear', align_corners=False)

        # Project DINO channels: embed_dim → embed_dim // 2
        #dino_proj = self.dino_proj(dino_resized)

        # -----------------------------
        # Fusion at Bottleneck
        # -----------------------------
        # poo4 has embed_dim//2 channels
        # dino_proj has embed_dim//2 channels
        # concatenated → embed_dim channels
        #bottleneck_input = torch.cat([poo4, dino_proj], dim=1)
        bottleneck_input = torch.cat([poo4, dino_resized], dim=1)

        bneck = self.bottleneck(bottleneck_input)

        # -----------------------------
        # Decoder
        # -----------------------------
        
        up1 = self.upconv1(bneck)
        #print(f"Up1 shape: {up1.shape}, D4 shape: {d4.shape}")
        #print(f"skip proj shape: {self.dinoskip4_proj(d4).shape}")
        up1 = up1 + self.dinoskip4_proj(d4)
        merge1 = torch.cat([up1, enc4], dim=1)
        dec1 = self.conv1(merge1)

        up2 = self.upconv2(dec1)
        up2 = up2 + self.dinoskip3_proj(d3)
        merge2 = torch.cat([up2, enc3], dim=1)
        dec2 = self.conv2(merge2)

        up3 = self.upconv3(dec2)
        up3 = up3 + self.dinoskip2_proj(d2)
        merge3 = torch.cat([up3, enc2], dim=1)
        dec3 = self.conv3(merge3)

        up4 = self.upconv4(dec3)
        up4 = up4 + self.dinoskip1_proj(d1)
        merge4 = torch.cat([up4, enc1], dim=1)
        dec4 = self.conv4(merge4)

        # Final segmentation output
        x = self.final_conv(dec4)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)

        return x



class DINOv3UNetEncodeDecoderSkipsAnyUp_final2(nn.Module):
    def __init__(self, in_channels,dino_dim, embed_dim=1024 , num_classes=2, out_size=(256, 256)):
        super().__init__()
        self.out_size = out_size
        self.dino_dim = dino_dim
        self.embed_dim = embed_dim

        # define skip projection layers for DINO features
        self.dinoskip1_proj = nn.Conv2d(dino_dim, embed_dim // 16, kernel_size=1)
        self.dinoskip2_proj = nn.Conv2d(dino_dim, embed_dim // 8, kernel_size=1)
        self.dinoskip3_proj = nn.Conv2d(dino_dim, embed_dim // 4, kernel_size=1)
        self.dinoskip4_proj = nn.Conv2d(dino_dim, embed_dim // 2, kernel_size=1)

        # Encoder
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim//16)
        self.poo1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(in_channels=embed_dim//16, out_channels=embed_dim//8)
        self.poo2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(in_channels=embed_dim//8, out_channels=embed_dim//4)
        self.poo3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(in_channels=embed_dim//4, out_channels=embed_dim//2)
        self.poo4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = ConvBlock(in_channels=embed_dim//2, out_channels=embed_dim)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2)
        self.conv1   = ConvBlock(embed_dim, embed_dim//2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim//2, embed_dim//4, kernel_size=2, stride=2)
        self.conv2   = ConvBlock(embed_dim//2, embed_dim//4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim//4, embed_dim//8, kernel_size=2, stride=2)
        self.conv3   = ConvBlock(256, 128)

        self.upconv4 = nn.ConvTranspose2d(embed_dim//8, embed_dim//16, kernel_size=2, stride=2)
        self.conv4   = ConvBlock(embed_dim//8, embed_dim//16)

        # Final output layer
        self.final_conv = nn.Conv2d(embed_dim//16, num_classes, kernel_size=1)

    def forward(self, x,upfeatures):

        d1 = upfeatures[0]  # shallowest
        d2 = upfeatures[1]
        d3 = upfeatures[2]
        d4 = upfeatures[3]

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
        up1 = up1 + self.dinoskip4_proj(d4)
        merge1 = torch.cat([up1, enc4], dim=1)
        dec1 = self.conv1(merge1)

        up2 = self.upconv2(dec1)
        up2 = up2 + self.dinoskip3_proj(d3)
        merge2 = torch.cat([up2, enc3], dim=1)
        dec2 = self.conv2(merge2)

        up3 = self.upconv3(dec2)
        up3 = up3 + self.dinoskip2_proj(d2)
        merge3 = torch.cat([up3, enc2], dim=1)
        dec3 = self.conv3(merge3)

        up4 = self.upconv4(dec3)
        up4 = up4 + self.dinoskip1_proj(d1)
        merge4 = torch.cat([up4, enc1], dim=1)
        dec4 = self.conv4(merge4)

        x = self.final_conv(dec4)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        return x

class InputImgDinoFeatUNet(nn.Module):
    def __init__(self, img_channels, dino_dim,num_classes, out_size=(256, 256)):
        super().__init__()
        self.out_size = out_size
        #self.dimafterconcat = 3 + dino_dim  # after concatenating image and projected dino features
        #self.dimafterconcat = 2  # after concatenating image and projected dino features
        #self.dimafterconcat = 1 + 128  # after concatenating image and projected dino features
        self.dimafterconcat =  dino_dim  # after concatenating image and projected dino features


        # define dino channel reduction 
        #self.dino_proj = nn.Conv2d(dino_dim, 128, kernel_size=1)

        # Encoder
        self.enc1 = ConvBlock(in_channels=self.dimafterconcat, out_channels=64)
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

    def forward(self, x, dino_features):

        # layernorm dino features along channel dimension
        dino_features = F.layer_norm(dino_features, dino_features.shape[1:])
        #dino_features = torch.mean(dino_features, dim=1, keepdim=True)
        # make the input image grayscale to have 1 channel
        #x = torch.mean(x, dim=1, keepdim=True)

        #dino_features = self.dino_proj(dino_features)
        # normalize dino features 
        #dino_features = F.normalize(dino_features, dim=1)
        # avarage dino features to have 1 channel
        #dino_features = torch.mean(dino_features, dim=1, keepdim=True)

        # x is grayscale to get 1 channel
        #x = torch.mean(x, dim=1, keepdim=True)
        #print(f"Input image shape: {x.shape}, DINO features shape after projection: {dino_features.shape}")

        # print mean and std of input image and dino features
        #print(f"Input image mean: {x.mean().item()}, std: {x.std().item()}")
        #print(f"DINO features mean: {dino_features.mean().item()}, std: {dino_features.std().item()}")
        
        # concatenate input image and dino features along channel dimension
        #x = torch.cat([x, dino_features], dim=1)
        x = dino_features

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
    

class DinoAdapter(nn.Module):
    def __init__(self, dino_dim, out_dim=1024):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(dino_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.adapter(x)
    
class DinoAttentionGateFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, cnn_feat, dino_feat):
        gate = self.attention_gate(dino_feat)
        return cnn_feat * gate + cnn_feat

class DINOv3UNetEncodeDecoderSkipsAnyUpAttentionGate(nn.Module):
    def __init__(self, in_channels,dino_dim, embed_dim=1024 , num_classes=2, out_size=(256, 256)):
        super().__init__()
        self.out_size = out_size
        self.dino_dim = dino_dim
        self.embed_dim = embed_dim

        # define skip projection layers for DINO features
        self.dino1adapter = DinoAdapter(dino_dim, embed_dim // 2)
        self.dino2adapter = DinoAdapter(dino_dim, embed_dim // 4)
        self.dino3adapter = DinoAdapter(dino_dim, embed_dim // 8)
        self.dino4adapter = DinoAdapter(dino_dim, embed_dim // 16)

        self.fuse1attngate = DinoAttentionGateFusion(embed_dim // 2)
        self.fuse2attngate = DinoAttentionGateFusion(embed_dim // 4)
        self.fuse3attngate = DinoAttentionGateFusion(embed_dim // 8)
        self.fuse4attngate = DinoAttentionGateFusion(embed_dim // 16)

        # Encoder
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim//16)
        self.poo1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(in_channels=embed_dim//16, out_channels=embed_dim//8)
        self.poo2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(in_channels=embed_dim//8, out_channels=embed_dim//4)
        self.poo3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(in_channels=embed_dim//4, out_channels=embed_dim//2)
        self.poo4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = ConvBlock(in_channels=embed_dim//2, out_channels=embed_dim)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2)
        self.conv1   = ConvBlock(embed_dim, embed_dim//2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim//2, embed_dim//4, kernel_size=2, stride=2)
        self.conv2   = ConvBlock(embed_dim//2, embed_dim//4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim//4, embed_dim//8, kernel_size=2, stride=2)
        self.conv3   = ConvBlock(256, 128)

        self.upconv4 = nn.ConvTranspose2d(embed_dim//8, embed_dim//16, kernel_size=2, stride=2)
        self.conv4   = ConvBlock(embed_dim//8, embed_dim//16)

        # Final output layer
        self.final_conv = nn.Conv2d(embed_dim//16, num_classes, kernel_size=1)

    def forward(self, x,upfeatures):

        d1 = upfeatures[0]  # shallowest
        d2 = upfeatures[1]
        d3 = upfeatures[2]
        d4 = upfeatures[3]

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

        # skip connections with attention gate fusion
        d4_adapted = self.dino1adapter(d4)
        enc4 = self.fuse1attngate(enc4, d4_adapted)
        d3_adapted = self.dino2adapter(d3)
        enc3 = self.fuse2attngate(enc3, d3_adapted)
        d2_adapted = self.dino3adapter(d2)
        enc2 = self.fuse3attngate(enc2, d2_adapted)
        d1_adapted = self.dino4adapter(d1)
        enc1 = self.fuse4attngate(enc1, d1_adapted)

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
    

class DINOv3UNetEncodeDecoderSkipInterpolAttentionGate(nn.Module):
    def __init__(self, in_channels,dino_dim, embed_dim=1024 , num_classes=2, out_size=(256, 256)):
        super().__init__()
        self.out_size = out_size
        self.dino_dim = dino_dim
        self.embed_dim = embed_dim

        # define skip projection layers for DINO features
        self.dino1adapter = DinoAdapter(dino_dim, embed_dim // 2)
        self.dino2adapter = DinoAdapter(dino_dim, embed_dim // 4)
        self.dino3adapter = DinoAdapter(dino_dim, embed_dim // 8)
        self.dino4adapter = DinoAdapter(dino_dim, embed_dim // 16)

        self.fuse1attngate = DinoAttentionGateFusion(embed_dim // 2)
        self.fuse2attngate = DinoAttentionGateFusion(embed_dim // 4)
        self.fuse3attngate = DinoAttentionGateFusion(embed_dim // 8)
        self.fuse4attngate = DinoAttentionGateFusion(embed_dim // 16)

        # Encoder
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim//16)
        self.poo1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(in_channels=embed_dim//16, out_channels=embed_dim//8)
        self.poo2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(in_channels=embed_dim//8, out_channels=embed_dim//4)
        self.poo3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(in_channels=embed_dim//4, out_channels=embed_dim//2)
        self.poo4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = ConvBlock(in_channels=embed_dim//2, out_channels=embed_dim)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2)
        self.conv1   = ConvBlock(embed_dim, embed_dim//2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim//2, embed_dim//4, kernel_size=2, stride=2)
        self.conv2   = ConvBlock(embed_dim//2, embed_dim//4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim//4, embed_dim//8, kernel_size=2, stride=2)
        self.conv3   = ConvBlock(256, 128)

        self.upconv4 = nn.ConvTranspose2d(embed_dim//8, embed_dim//16, kernel_size=2, stride=2)
        self.conv4   = ConvBlock(embed_dim//8, embed_dim//16)

        # Final output layer
        self.final_conv = nn.Conv2d(embed_dim//16, num_classes, kernel_size=1)

    def forward(self, x,dinofeatures):

        d1 = dinofeatures[0]  # shallowest
        d2 = dinofeatures[1]
        d3 = dinofeatures[2]
        d4 = dinofeatures[3]

        # interpolate dino features to match encoder feature map sizes
        d1 = F.interpolate(d1, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        d2 = F.interpolate(d2, size=(x.shape[2]//2, x.shape[3]//2), mode='bilinear', align_corners=False)
        d3 = F.interpolate(d3, size=(x.shape[2]//4, x.shape[3]//4), mode='bilinear', align_corners=False)
        d4 = F.interpolate(d4, size=(x.shape[2]//8, x.shape[3]//8), mode='bilinear', align_corners=False)   


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

        # skip connections with attention gate fusion
        d4_adapted = self.dino1adapter(d4)
        # print shapes
        enc4 = self.fuse1attngate(enc4, d4_adapted)
        d3_adapted = self.dino2adapter(d3)
        enc3 = self.fuse2attngate(enc3, d3_adapted)
        d2_adapted = self.dino3adapter(d2)
        enc2 = self.fuse3attngate(enc2, d2_adapted)
        d1_adapted = self.dino4adapter(d1)
        enc1 = self.fuse4attngate(enc1, d1_adapted)

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


class DinoBottleneckAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, dino_feat):
        alpha = self.gate(dino_feat)
        return cnn_feat * alpha + cnn_feat, alpha
    
class DINOv3UNetEncodeDecoderAttentionGate(nn.Module):
    def __init__(self, in_channels, dino_dim, dinov3_model, embed_dim=1024, n_layers=4, num_classes=2, out_size=(256, 256)):
        super().__init__()

        self.out_size = out_size
        self.embed_dim = embed_dim
        self.dino_dim = dino_dim

        # ------------------------------------------
        # DINO Encoder
        # ------------------------------------------
        self.encoder = DINOv3Encoder(dinov3_model=dinov3_model, n_layers=n_layers)

        # DINO produces: B, C_dino, Hd, Wd
        # We project it down to match the UNet bottleneck input channels
        # CNN bottleneck input = embed_dim // 2
        self.dino_proj = nn.Conv2d(dino_dim, embed_dim // 2, kernel_size=1)
        self.bottleneck_fusion = DinoBottleneckAttention(embed_dim // 2)
 
        # ------------------------------------------
        # CNN Encoder (UNet-style)
        # ------------------------------------------
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim // 16)
        self.poo1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(embed_dim // 16, embed_dim // 8)
        self.poo2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(embed_dim // 8, embed_dim // 4)
        self.poo3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(embed_dim // 4, embed_dim // 2)
        self.poo4 = nn.MaxPool2d(2)

        # Bottleneck of the CNN
        # Input channels: embed_dim // 2  (from CNN)
        # After concatenation with projected DINO → embed_dim total
        self.bottleneck = ConvBlock(embed_dim//2, embed_dim)

        # ------------------------------------------
        # UNet Decoder
        # ------------------------------------------
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
        self.conv1   = ConvBlock(embed_dim, embed_dim // 2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2)
        self.conv2   = ConvBlock(embed_dim // 2, embed_dim // 4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
        self.conv3   = ConvBlock(embed_dim // 4, embed_dim // 8)

        self.upconv4 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2)
        self.conv4   = ConvBlock(embed_dim // 8, embed_dim // 16)

        self.final_conv = nn.Conv2d(embed_dim // 16, num_classes, 1)

    def forward(self, x):

        # -----------------------------
        # CNN Encoder
        # -----------------------------
        enc1 = self.enc1(x)
        poo1 = self.poo1(enc1)

        enc2 = self.enc2(poo1)
        poo2 = self.poo2(enc2)

        enc3 = self.enc3(poo2)
        poo3 = self.poo3(enc3)

        enc4 = self.enc4(poo3)
        poo4 = self.poo4(enc4)  # shape: B, embed_dim//2, H, W

        # -----------------------------
        # DINO deep features
        # -----------------------------
        dino_deep, _ = self.encoder(x)
        # dino_deep: B, embed_dim, Hd, Wd
        

        # Match spatial size with poo4
        _, _, H, W = poo4.shape
        dino_resized = F.interpolate(dino_deep, size=(H, W), mode='bilinear', align_corners=False)

        # Project DINO channels: embed_dim → embed_dim // 2
        dino_proj = self.dino_proj(dino_resized)


        # -----------------------------
        # Fusion at Bottleneck
        # -----------------------------
        
        fused_boo4, attention_map = self.bottleneck_fusion(poo4, dino_proj)
        bneck = self.bottleneck(fused_boo4)
        
        up1 = self.upconv1(bneck)
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

        # Final segmentation output
        x = self.final_conv(dec4)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        return x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, bias=False)
        self.W_x = nn.Conv2d(F_x, F_int, 1, bias=False)
        self.psi = nn.Conv2d(F_int, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        # g: UNet bottleneck feature
        # x: DINOv3 feature
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))

        return x * psi

class DINOv3UNetEncodeDecoderAttentionGateFullV1(nn.Module):
    def __init__(self, in_channels, embed_dim, dinov3_model, n_layers=4, num_classes=2, out_size=(256, 256)):
        super().__init__()

        self.out_size = out_size
        self.embed_dim = embed_dim
   

        # ------------------------------------------
        # DINO Encoder
        # ------------------------------------------
        self.encoder = DINOv3Encoder(dinov3_model=dinov3_model, n_layers=n_layers)


        # ------------------------------------------
        # Attention Gate for DINO features at Bottleneck
        # ------------------------------------------
        self.attention_gate = AttentionGate(F_g=embed_dim // 2 , F_x=embed_dim // 2, F_int=embed_dim // 4)
        
        
        # ------------------------------------------
        # CNN Encoder (UNet-style)
        # ------------------------------------------
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim // 16)
        self.poo1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(embed_dim // 16, embed_dim // 8)
        self.poo2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(embed_dim // 8, embed_dim // 4)
        self.poo3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(embed_dim // 4, embed_dim // 2)
        self.poo4 = nn.MaxPool2d(2)

        # Bottleneck of the CNN
        # Input channels: embed_dim // 2  (from CNN)
        # After concatenation with projected DINO → embed_dim total
        self.bottleneck = ConvBlock(embed_dim, embed_dim)

        # ------------------------------------------
        # UNet Decoder
        # ------------------------------------------
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
        self.conv1   = ConvBlock(embed_dim, embed_dim // 2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2)
        self.conv2   = ConvBlock(embed_dim // 2, embed_dim // 4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
        self.conv3   = ConvBlock(embed_dim // 4, embed_dim // 8)

        self.upconv4 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2)
        self.conv4   = ConvBlock(embed_dim // 8, embed_dim // 16)

        self.final_conv = nn.Conv2d(embed_dim // 16, num_classes, 1)

    def forward(self, x):

        # -----------------------------
        # CNN Encoder
        # -----------------------------
        enc1 = self.enc1(x)
        poo1 = self.poo1(enc1)

        enc2 = self.enc2(poo1)
        poo2 = self.poo2(enc2)

        enc3 = self.enc3(poo2)
        poo3 = self.poo3(enc3)

        enc4 = self.enc4(poo3)
        poo4 = self.poo4(enc4)  # shape: B, embed_dim//2, H, W

        # -----------------------------
        # DINO deep features
        # -----------------------------
        with torch.no_grad():
            dino_deep, _ = self.encoder(x)
        # dino_deep: B, embed_dim, Hd, Wd
        

        # Match spatial size with poo4
        _, _, H, W = poo4.shape
        dino_resized = F.interpolate(dino_deep, size=(H, W), mode='bilinear', align_corners=False)

        # Project DINO channels: embed_dim → embed_dim // 2
        #dino_proj = self.dino_proj(dino_resized)

        
        # Fusion at Bottleneck with Attention Gate
        gated_dino_feat = self.attention_gate(poo4, dino_resized)
        bottleneck_input = torch.cat([poo4, gated_dino_feat], dim=1)

        bneck = self.bottleneck(bottleneck_input)

        # -----------------------------
        # Decoder
        # -----------------------------
        up1 = self.upconv1(bneck)
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

        # Final segmentation output
        x = self.final_conv(dec4)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)

        return x
    

class DINOv3UNetEncodeDecoderAttentionGateFullV2(nn.Module):
    def __init__(self, in_channels,dino_dim, dinov3_model,embed_dim=1024, n_layers=4, num_classes=2, out_size=(256, 256)):
        super().__init__()

        self.out_size = out_size
        self.embed_dim = embed_dim
        self.dino_dim = dino_dim

        # ------------------------------------------
        # DINO Encoder
        # ------------------------------------------
        self.encoder = DINOv3Encoder(dinov3_model=dinov3_model, n_layers=n_layers)

        # ------------------------------------------
        # Attention Gate for DINO features at Bottleneck
        # ------------------------------------------
        self.attention_gate = AttentionGate(F_g=embed_dim // 2 , F_x=dino_dim, F_int=embed_dim // 4)
        
        
        # -----------------------------------------
        # CNN Encoder (UNet-style)
        # ------------------------------------------
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=embed_dim // 16)
        self.poo1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(embed_dim // 16, embed_dim // 8)
        self.poo2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(embed_dim // 8, embed_dim // 4)
        self.poo3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(embed_dim // 4, embed_dim // 2)
        self.poo4 = nn.MaxPool2d(2)

        # Bottleneck of the CNN
  
        #self.bneckdownproj = nn.Conv2d((embed_dim//2)+dino_dim, embed_dim, kernel_size=1)
        self.bottleneck = ConvBlock((embed_dim//2)+dino_dim, embed_dim)


        # ------------------------------------------
        # UNet Decoder
        # ------------------------------------------
        self.upconv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
        self.conv1   = ConvBlock(embed_dim, embed_dim // 2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2)
        self.conv2   = ConvBlock(embed_dim // 2, embed_dim // 4)

        self.upconv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
        self.conv3   = ConvBlock(embed_dim // 4, embed_dim // 8)

        self.upconv4 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2)
        self.conv4   = ConvBlock(embed_dim // 8, embed_dim // 16)

        self.final_conv = nn.Conv2d(embed_dim // 16, num_classes, 1)

    def forward(self, x):

        # -----------------------------
        # CNN Encoder
        # -----------------------------
        enc1 = self.enc1(x)
        poo1 = self.poo1(enc1)

        enc2 = self.enc2(poo1)
        poo2 = self.poo2(enc2)

        enc3 = self.enc3(poo2)
        poo3 = self.poo3(enc3)

        enc4 = self.enc4(poo3)
        poo4 = self.poo4(enc4)  # shape: B, embed_dim//2, H, W

        # -----------------------------
        # DINO deep features
        # -----------------------------
        with torch.no_grad():
            dino_deep, _ = self.encoder(x)
        # dino_deep: B, embed_dim, Hd, Wd
        

        # Match spatial size with poo4
        _, _, H, W = poo4.shape
        dino_resized = F.interpolate(dino_deep, size=(H, W), mode='bilinear', align_corners=False)

        # Project DINO channels: embed_dim → embed_dim // 2
        #dino_proj = self.dino_proj(dino_resized)

        
        # Fusion at Bottleneck with Attention Gate
        gated_dino_feat = self.attention_gate(poo4, dino_resized)
        bottleneck_input = torch.cat([poo4, gated_dino_feat], dim=1)

        #bottleneck_input = self.bneckdownproj(bottleneck_input)

        bneck = self.bottleneck(bottleneck_input)

        # -----------------------------
        # Decoder
        # -----------------------------
        up1 = self.upconv1(bneck)
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

        # Final segmentation output
        x = self.final_conv(dec4)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)

        return x
    

