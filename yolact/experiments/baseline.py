from yolact.configs.backbone import ResNet101
from yolact.configs.mask import MaskBase
from yolact.configs.fpn import FPNBase
from yolact.configs.coco import Coco2017
from yolact.configs.training import TrainingParameters


class ResNet101CocoBaseline:
    def __init__(self):
        # resnet backbone default transform
        self.backbone = ResNet101()

        # set protonet
        self.mask = MaskBase()
        self.mask.mask_type = 1  # lincomb
        self.mask.mask_proto_src = 0
        self.mask.mask_proto_net = [(256, 3, {'padding': 1})] * 3 + \
                                   [(None, -2, {}), (256, 3, {'padding': 1})] + \
                                   [(32, 1, {})]
        # Normalize the mask loss to emulate roi pooling's affect on loss
        self.mask.mask_proto_normalize_emulate_roi_pooling = True

        # dataset
        self.dataset = Coco2017()

        # Set this to a config object if you want an FPN (inherit from fpn_base). See fpn_base for details.
        self.fpn = FPNBase()
        self.fpn.use_conv_downsample = True  # Whether to down sample with a 3x3 stride 2 conv layer instead of just a stride 2 selection
        self.fpn.num_downsample = 2  # The number of extra layers to be produced by downsampling starting at P5, e.g. p6, p7 from paper

        # training parameters
        self.training_params = TrainingParameters()
        self.training_params.max_size = 550  # Input image size.
        self.training_params.lr_steps = (280000, 600000, 700000, 750000)
        self.training_params.max_iter = 800000
        self.training_params.mask_alpha = 6.125  # loss weight
        self.training_params.share_prediction_module = True  # Use the same weights for each network head
        # Uses the same network format as mask_proto_net, except this time it's for adding extra head layers
        # before the final prediction in prediction modules. If this is none, no extra layers will be added.
        self.training_params.extra_head_net = [(256, 3, {'padding': 1})]

    def dict(self):
        params = vars(self)
        params["backbone"] = params["backbone"].dict()
        params["mask"] = params["mask"].dict()
        params["fpn"] = params["fpn"].dict()
        params["dataset"] = params["dataset"].dict()
        params["training_params"] = params["training_params"].dict()
        return params
