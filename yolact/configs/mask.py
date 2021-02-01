from typing import Optional, List, Tuple, Callable
from yolact.configs.activation import activation_func


class MaskBase:
    def __init__(self):
        # Direct produces masks directly as the output of each pred module.
        # This is denoted as fc-mask in the paper.
        # Parameters: mask_size, use_gt_bboxes
        # 0 -> direct, 1 -> lincomb
        self.mask_type = 1  # type: int

        self.mask_size = 16    # type: int

        # Since we're producing (near) full image masks, it'd take too much
        # vram to backprop on every single mask. Thus we select only a subset.
        self.masks_to_train = 100  # type: int

        # The input layer to the mask prototype generation network. This is an
        # index in backbone.layers. Use to use the image itself instead.
        self.mask_proto_src = None  # type: Optional[int]

        # A list of layers in the mask proto network with the last one
        # being where the masks are taken from. Each conv layer is in
        # the form (num_features, kernel_size, **kwdargs). An empty
        # list means to use the source for prototype masks. If the
        # kernel_size is negative, this creates a deconv layer instead.
        # If the kernel_size is negative and the num_features is None,
        # this creates a simple bilinear interpolation layer instead.
        self.mask_proto_net = [(256, 3, {}), (256, 3, {})]  # type: List[Tuple]

        # Whether to include an extra coefficient that corresponds to a proto
        # mask of all ones.
        self.mask_proto_bias = False  # type: bool

        # The activation to apply to each prototype mask.
        self.mask_proto_prototype_activation = activation_func["relu"]  # type: Callable
        # After summing the prototype masks with the predicted what activation to apply to the final mask.
        self.mask_proto_mask_activation = activation_func["sigmoid"]  # type: Callable
        # The activation to apply to the mask coefficients.
        self.mask_proto_coeff_activation = activation_func["tanh"]  # type: Callable

        # If True, crop the mask with the predicted bbox during training.
        self.mask_proto_crop = True  # type: bool

        # If cropping, the percent to expand the cropping bbox by
        # in each direction. This is to make the model less reliant
        # on perfect bbox predictions.
        self.mask_proto_crop_expand = 0.0  # type: float

        # If not None, apply an l1 or disjunctive regularization
        # loss directly to the prototype masks.
        self.mask_proto_loss = None  # type: Optional[str]

        # Binarize GT after dowsnampling during training?
        self.mask_proto_binarize_downsampled_gt = True  # type: bool

        # Whether to normalize mask loss by sqrt(sum(gt))
        self.mask_proto_normalize_mask_loss_by_sqrt_area = False  # type: bool

        # Reweight mask loss such that background is divided by
        # #background and foreground is divided by #foreground.
        self.mask_proto_reweight_mask_loss = False  # type: bool

        #  The path to the grid file to use with the next option.
        #  This should be a numpy.dump file with shape [numgrids, h, w]
        #  where h and w are w.r.t. the mask_proto_src convout.
        self.mask_proto_grid_file = 'data/grid.npy'  # type: str

        # Whether to add extra grid features to the proto_net input.
        self.mask_proto_use_grid = False  # type: bool

        # Add an extra set of sigmoided coefficients that is multiplied
        # into the predicted coefficients in order to "gate" them.
        self.mask_proto_coeff_gate = False  # type: bool

        # For each prediction module, downsample the prototypes
        # to the convout size of that module and supply the prototypes as input
        # in addition to the already supplied backbone features.
        self.mask_proto_prototypes_as_features = False  # type: bool

        # If the above is set, don't backprop gradients to
        # to the prototypes from the network head.
        self.mask_proto_prototypes_as_features_no_grad = False  # type: bool

        # Remove masks that are downsampled to 0 during loss calculations.
        self.mask_proto_remove_empty_masks = False  # type: bool

        # The coefficient to multiple the forground pixels with if reweighting.
        self.mask_proto_reweight_coeff = 1  # type: float

        # Apply coefficient diversity loss on the coefficients so that the same
        # instance has similar coefficients.
        self.mask_proto_coeff_diversity_loss = False  # type: bool

        # The weight to use for the coefficient diversity loss.
        self.mask_proto_coeff_diversity_alpha = 1  # type: float

        # Normalize the mask loss to emulate roi pooling's affect on loss.
        self.mask_proto_normalize_emulate_roi_pooling = False  # type: bool

        # Whether to use the old loss in addition to any special new losses.
        self.mask_proto_double_loss = False  # type: bool

        # The alpha to weight the above loss.
        self.mask_proto_double_loss_alpha = 1  # type: float

        # If true, this will give each prediction head its own prototypes.
        self.mask_proto_split_prototypes_by_head = False  # type: bool

        # Whether to crop with the predicted box or the gt box.
        self.mask_proto_crop_with_pred_box = False  # type: bool

    def dict(self):
        return vars(self)
