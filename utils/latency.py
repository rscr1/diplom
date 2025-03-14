import torch
import segmentation_models_pytorch as smp
from typing import Tuple
import time

import urllib.request
from urllib.request import urlopen
import ssl
import json
ssl._create_default_https_context = ssl._create_unverified_context

def measure_latency(
    subnet,
    runs: int,
    image_size: Tuple[int, int],
    device: torch.device,
    seed: int,
    warmup_cut=5,
):
    cuda = device.type == "cuda"
    if cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    timings = torch.zeros(runs)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    with torch.no_grad():
        for i in range(runs):
            x = torch.randn(1, 3, *image_size, generator=g, device=device)

            if cuda:
                starter.record()
                _ = subnet.predict(x)
                ender.record()
                torch.cuda.synchronize()

                curr_time = starter.elapsed_time(ender)
                timings[i] = curr_time
            else:
                start = time.time()
                _ = subnet(x)
                end = time.time()

                timings[i] = 1000 * (end - start)

    # Я пробовал разные вещи, но первые 2-3 прогона были с сильным отклонением
    timings = timings[warmup_cut:] if runs > 15 else timings
    mean_latency = torch.mean(timings).item()
    std_latency = torch.std(timings).item()

    return mean_latency, std_latency


device = torch.device('cuda:1')


# model_1 = smp.FPN(
#     encoder_name='se_resnext50_32x4d', 
#     classes=8,
# )
# model_1 = model_1.to(device)
# print(measure_latency(model_1, 1000, (256, 1248), device, seed=42), 'fpn se_resnext50_32x4d')

# model_3 = smp.FPN(
#     encoder_name='mit_b1', 
#     classes=8,
#         )
# model_3 = model_3.to(device)
# print(measure_latency(model_3, 1000, (256, 1248), device, seed=42), 'fpn mit_b1')

# model_4 = smp.FPN(
#     encoder_name='mit_b2', 
#     classes=8,
#         )
# model_4 = model_4.to(device)
# print(measure_latency(model_4, 1000, (256, 1248), device, seed=42), 'fpn mit_b2')

# model_5 = smp.DeepLabV3(
#     encoder_name='se_resnext50_32x4d', 
#     classes=8,
# )
# model_5 = model_5.to(device)
# print(measure_latency(model_5, 1000, (256, 1248), device, seed=42), 'deeplabv3 se_resnext50_32x4d')

# model_7 = smp.DeepLabV3(
#     encoder_name='mit_b1', 
#     classes=8,
# )
# model_7 = model_7.to(device)
# print(measure_latency(model_7, 1000, (256, 1248), device, seed=42), 'deeplabv3 mit_b1')

# model_8 = smp.DeepLabV3(
#     encoder_name='mit_b2', 
#     classes=8,
# )
# model_8 = model_8.to(device)
# print(measure_latency(model_8, 1000, (256, 1248), device, seed=42), 'deeplabv3 mit_b2')

# model_9 = smp.DeepLabV3Plus(
#     encoder_name='se_resnext50_32x4d', 
#     classes=8,
# )
# model_9 = model_9.to(device)
# print(measure_latency(model_9, 1000, (256, 1248), device, seed=42), 'deeplabv3+ se_resnext50_32x4d')

# # model_11 = smp.DeepLabV3Plus(
# #     encoder_name='mit_b1', 
# #     classes=8,
# # )
# # model_11 = model_11.to(device)
# print('not supported', 'deeplabv3+ mit_b1')

# # model_12 = smp.DeepLabV3Plus(
# #     encoder_name='mit_b2', 
# #     classes=8,
# # )
# # model_12 = model_12.to(device)
# print('not supported', 'deeplabv3+ mit_b2')


class SegmFPNWithDepthHead(smp.FPN):
    def __init__(self, encoder_name='resnet34', classes=1, encoder_weights='imagenet'):
        super(SegmFPNWithDepthHead, self).__init__(encoder_name=encoder_name, classes=classes, encoder_weights=encoder_weights)
        self.depth_head = smp.base.heads.SegmentationHead(
            in_channels=128, 
            out_channels=1,
            kernel_size=1, 
            activation=None,
            upsampling=4
        )


    def forward(self, x):
        features = self.encoder(x)
        decoder_out = self.decoder(*features)

        depth = self.depth_head(decoder_out) # depth
        mask = self.segmentation_head(decoder_out) # segm

        return depth, mask


# segm_depth_model = SegmFPNWithDepthHead(encoder_name='se_resnext50_32x4d', classes=8, encoder_weights='imagenet')
# segm_depth_model.load_state_dict(torch.load('/AkhmetzyanovD/projects/hztfm/multitask_pipeline/last_runs/launch_0/segm_depth_model.pth'), strict=False)
# segm_depth_model = segm_depth_model.to(device)
# print(measure_latency(segm_depth_model, 1000, (256, 1248), device, seed=42), 'fpn se_resnext50_32x4d two heads')


# segm_depth_model = SegmFPNWithDepthHead(encoder_name='mit_b1', classes=8, encoder_weights='imagenet')
# segm_depth_model.load_state_dict(torch.load('/AkhmetzyanovD/projects/hztfm/multitask_pipeline/last_runs/launch_1/segm_depth_model.pth'), strict=False)
# segm_depth_model = segm_depth_model.to(device)
# print(measure_latency(segm_depth_model, 1000, (256, 1248), device, seed=42), 'fpn mit_b1 two heads')


class SegmFPNWithDepthDecoderHead(smp.FPN):
    def __init__(self, encoder_name='resnet34', classes=31, encoder_weights='imagenet'):
        super(SegmFPNWithDepthDecoderHead, self).__init__(encoder_name=encoder_name, classes=classes, encoder_weights=encoder_weights)

        # New second decoder
        self.depth_decoder = smp.decoders.fpn.decoder.FPNDecoder(
            encoder_channels=self.encoder.out_channels
        )

        # New second head
        self.depth_head = smp.base.heads.SegmentationHead(
            in_channels=128, 
            out_channels=1,
            kernel_size=1, 
            activation=None,
            upsampling=4
        )
        self.segmentation_head = smp.base.heads.SegmentationHead(
            in_channels=128, 
            out_channels=classes,
            kernel_size=1, 
            activation=None,
            upsampling=4
        )


    def forward(self, x):
        features = self.encoder(x)
        
        decoder_segm = self.decoder(*features)
        decoder_depth = self.depth_decoder(*features)

        mask = self.segmentation_head(decoder_segm) # segm
        depth = self.depth_head(decoder_depth) # depth

        return depth, mask
    

# segm_depth_model = SegmFPNWithDepthDecoderHead(encoder_name='se_resnext50_32x4d', classes=8, encoder_weights='imagenet')
# segm_depth_model.load_state_dict(torch.load('/AkhmetzyanovD/projects/hztfm/multitask_pipeline/last_runs/launch_2/segm_depth_decoder_model.pth'), strict=False)
# segm_depth_model = segm_depth_model.to(device)
# print(measure_latency(segm_depth_model, 1000, (256, 1248), device, seed=42), 'fpn se_resnext50_32x4d two decoders heads')


# segm_depth_model = SegmFPNWithDepthDecoderHead(encoder_name='mit_b1', classes=8, encoder_weights='imagenet')
# segm_depth_model.load_state_dict(torch.load('/AkhmetzyanovD/projects/hztfm/multitask_pipeline/last_runs/launch_3/segm_depth_decoder_model.pth'), strict=False)
# segm_depth_model = segm_depth_model.to(device)
# print(measure_latency(segm_depth_model, 1000, (256, 1248), device, seed=42), 'fpn mit_b1 two decoders heads')




class DepthWithSegmHead(smp.FPN):
    def __init__(self, encoder_name='resnet34', classes=1, encoder_weights='imagenet'):
        super(DepthWithSegmHead, self).__init__(encoder_name=encoder_name, classes=1, encoder_weights=encoder_weights)
        
        self.depth_head = smp.base.heads.SegmentationHead(
            in_channels=128, 
            out_channels=1,
            kernel_size=1, 
            activation=None,
            upsampling=4
        )

        # New second head
        self.segmentation_head = smp.base.heads.SegmentationHead(
            in_channels=128, 
            out_channels=classes,
            kernel_size=1, 
            activation=None,
            upsampling=4
        )


    def forward(self, x):
        features = self.encoder(x)
        decoder_out = self.decoder(*features)

        depth = self.depth_head(decoder_out) # depth
        mask = self.segmentation_head(decoder_out) # segm

        return depth, mask



# depth_segm_model = DepthWithSegmHead(encoder_name='se_resnext50_32x4d', classes=8, encoder_weights='imagenet')
# depth_segm_model.load_state_dict(torch.load('/AkhmetzyanovD/projects/hztfm/multitask_pipeline/last_models/depth_segm_head.pth'))
# depth_segm_model = depth_segm_model.to(device)
# print(measure_latency(depth_segm_model, 1000, (256, 1248), device, seed=42), 'fpn se_resnext50_32x4d two heads')


# depth_segm_model = DepthWithSegmHead(encoder_name='mit_b1', classes=8, encoder_weights='imagenet')
# depth_segm_model.load_state_dict(torch.load('/AkhmetzyanovD/projects/hztfm/multitask_pipeline/last_models/depth_segm_head_mit.pth'))
# depth_segm_model = depth_segm_model.to(device)
# print(measure_latency(depth_segm_model, 1000, (256, 1248), device, seed=42), 'fpn mit_b1 two heads')


class DepthWithSegmDecoderHead(smp.FPN):
    def __init__(self, encoder_name='resnet34', classes=1, encoder_weights='imagenet'):
        super(DepthWithSegmDecoderHead, self).__init__(encoder_name=encoder_name, classes=1, encoder_weights=encoder_weights)
        
        self.depth_head = smp.base.heads.SegmentationHead(
            in_channels=128,
            out_channels=1,
            kernel_size=1, 
            activation=None,
            upsampling=4
        )

        # New second decoder
        self.segmentation_decoder = smp.decoders.fpn.decoder.FPNDecoder(
            encoder_channels=self.encoder.out_channels
        )

        # New second head
        self.segmentation_head = smp.base.heads.SegmentationHead(
            in_channels=128, 
            out_channels=classes,
            kernel_size=1, 
            activation=None,
            upsampling=4
        )
 

    def forward(self, x):
        features = self.encoder(x)
        
        decoder_depth = self.decoder(*features)
        decoder_segm = self.segmentation_decoder(*features)

        depth = self.depth_head(decoder_depth) # depth
        mask = self.segmentation_head(decoder_segm) # segm

        return depth, mask
    

# depth_segm_model = DepthWithSegmDecoderHead(encoder_name='se_resnext50_32x4d', classes=8, encoder_weights='imagenet')
# depth_segm_model.load_state_dict(torch.load('/AkhmetzyanovD/projects/hztfm/multitask_pipeline/last_models/depth_segm_decoder_head.pth'))
# depth_segm_model = depth_segm_model.to(device)
# print(measure_latency(depth_segm_model, 1000, (256, 1248), device, seed=42), 'fpn se_resnext50_32x4d two decoders heads')


depth_segm_model = DepthWithSegmDecoderHead(encoder_name='mit_b1', classes=8, encoder_weights='imagenet')
depth_segm_model.load_state_dict(torch.load('/AkhmetzyanovD/projects/hztfm/multitask_pipeline/last_models/depth_segm_decoder_head_mit.pth'))
depth_segm_model = depth_segm_model.to(device)
print(measure_latency(depth_segm_model, 1000, (256, 1248), device, seed=42), 'fpn mit_b1 two decoders heads')