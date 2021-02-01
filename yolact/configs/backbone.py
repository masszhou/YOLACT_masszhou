from yolact.layers.backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone


class ResNet101:
    def __init__(self):
        self.name = 'ResNet101'
        self.path = 'resnet101_reducedfc.pth'
        self.type = ResNetBackbone
        self.args = ([3, 4, 23, 3],)
        self.transform = {
            'channel_order': 'RGB',
            'normalize': True,
            'subtract_means': False,
            'to_float': False,
        }
        self.selected_layers = list(range(2, 8))
        self.pred_scales = [[1]] * 6
        self.pred_aspect_ratios = [[[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]]] * 6
        self.use_pixel_scales = False
        self.preapply_sqrt = True
        self.use_square_anchors = False

    def dict(self):
        return vars(self)