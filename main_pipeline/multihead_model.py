import segmentation_models_pytorch as smp


class SegmDeepLabV3WithDepthHead(smp.DeepLabV3):
    def __init__(self, encoder_name='resnet34', classes=8, encoder_weights='imagenet'):
        super(SegmDeepLabV3WithDepthHead, self).__init__(encoder_name=encoder_name, classes=classes, encoder_weights=encoder_weights)
        self.depth_head = smp.base.heads.SegmentationHead(
            in_channels=256, 
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


class SegmDeepLabV3WithDepthDecoderHead(smp.DeepLabV3):
    def __init__(self, encoder_name='resnet34', classes=8, encoder_weights='imagenet'):
        super(SegmDeepLabV3WithDepthDecoderHead, self).__init__(encoder_name=encoder_name, classes=classes, encoder_weights=encoder_weights)
        self.depth_decoder = smp.decoders.deeplabv3.decoder.DeepLabV3Decoder(
            encoder_channels=self.encoder.out_channels
        )

        self.depth_head = smp.base.heads.SegmentationHead(
            in_channels=256, 
            out_channels=1,
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


class SegmFPNWithDepthHead(smp.FPN):
    def __init__(self, encoder_name='resnet34', classes=8, encoder_weights='imagenet'):
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


class SegmFPNWithDepthDecoderHead(smp.FPN):
    def __init__(self, encoder_name='resnet34', classes=8, encoder_weights='imagenet'):
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

        # Changing defult layer.head to layer.segmentation_head
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


class DepthDeepLabV3WithSegmHead(smp.DeepLabV3):
    def __init__(self, encoder_name='resnet34', classes=8, encoder_weights='imagenet'):
        super(DepthDeepLabV3WithSegmHead, self).__init__(encoder_name=encoder_name, classes=1, encoder_weights=encoder_weights)
        
        self.depth_head = smp.base.heads.SegmentationHead(
            in_channels=256, 
            out_channels=1,
            kernel_size=1, 
            activation=None,
            upsampling=4
        )

        # New second head
        self.segmentation_head = smp.base.heads.SegmentationHead(
            in_channels=256, 
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


class DepthDeepLabV3WithSegmDecoderHead(smp.DeepLabV3):
    def __init__(self, encoder_name='resnet34', classes=8, encoder_weights='imagenet'):
        super(DepthDeepLabV3WithSegmDecoderHead, self).__init__(
            encoder_name=encoder_name, 
            classes=1, 
            encoder_weights=encoder_weights
        )
        
        self.depth_head = smp.base.heads.SegmentationHead(
            in_channels=256, 
            out_channels=1,
            kernel_size=1, 
            activation=None,
            upsampling=4
        )

        # New second decoder
        self.segmentation_decoder = smp.decoders.deeplabv3.decoder.DeepLabV3Decoder(
            encoder_channels=self.encoder.out_channels
        )

        # New second head
        self.segmentation_head = smp.base.heads.SegmentationHead(
            in_channels=256, 
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
    

class DepthFPNWithSegmHead(smp.FPN):
    def __init__(self, encoder_name='resnet34', classes=8, encoder_weights='imagenet'):
        super(DepthFPNWithSegmHead, self).__init__(encoder_name=encoder_name, classes=1, encoder_weights=encoder_weights)
        
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


class DepthFPNWithSegmDecoderHead(smp.FPN):
    def __init__(self, encoder_name='resnet34', classes=8, encoder_weights='imagenet'):
        super(DepthFPNWithSegmDecoderHead, self).__init__(encoder_name=encoder_name, classes=1, encoder_weights=encoder_weights)
        
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
