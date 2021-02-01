class FPNBase:
    def __init__(self):
        # The number of features to have in each FPN layer
        self.num_features = 256

        # The upsampling mode used
        self.interpolation_mode = "bilinear"

        # The number of extra layers to be produced by downsampling starting at P5
        self.num_downsample = 1

        # Whether to down sample with a 3x3 stride 2 conv layer instead of just a stride 2 selection
        self.use_conv_downsample = False

        # Whether to pad the pred layers with 1 on each side (I forgot to add this at the start)
        # This is just here for backwards compatibility
        self.pad = True

        # Whether to add relu to the downsampled layers.
        self.relu_downsample_layers = False

        # Whether to add relu to the regular layers
        self.relu_pred_layers = True

    def dict(self):
        return vars(self)
