class TrainingParameters:
    def __init__(self):
        # self.dataset = "coco2017_dataset"
        # self.num_classes = 81  # This should include the background class
        self.max_iter = 400000

        # The maximum number of detections for evaluation
        self.max_num_detections = 100

        # dw' = momentum * dw - lr * (grad + decay * w)
        self.lr = 1e-3
        self.momentum = 0.9
        self.decay = 5e-4

        # For each lr step, what to multiply the lr with
        self.gamma = 0.1
        self.lr_steps = (280000, 360000, 400000)

        # Initial learning rate to linearly warmup from (if until > 0)
        self.lr_warmup_init = 1e-4

        # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
        self.lr_warmup_until = 500

        # The terms to scale the respective loss by
        self.conf_alpha = 1
        self.bbox_alpha = 1.5
        self.mask_alpha = 0.4 / 256 * 140 * 140  # Some funky equation. Don't worry about it.

        # Eval.py sets this if you just want to run YOLACT as a detector
        self.eval_mask_branch = True

        # Top_k examples to consider for NMS
        self.nms_top_k = 200
        # Examples with confidence less than this are not considered by NMS
        self.nms_conf_thresh = 0.05
        # Boxes with IoU overlap greater than this threshold will be culled during NMS
        self.nms_thresh = 0.5

        # SSD data augmentation parameters
        # Randomize hue, vibrance, etc.
        self.augment_photometric_distort = True
        # Have a chance to scale down the image and pad (to emulate smaller detections)
        self.augment_expand = True
        # Potentialy sample a random crop from the image and put it in a random place
        self.augment_random_sample_crop = True
        # Mirror the image with a probability of 1/2
        self.augment_random_mirror = True
        # Flip the image vertically with a probability of 1/2
        self.augment_random_flip = False
        # With uniform probability, rotate the image [0,90,180,270] degrees
        self.augment_random_rot90 = False

        # Discard detections with width and height smaller than this (in absolute width and height)
        self.discard_box_width = 4 / 550
        self.discard_box_height = 4 / 550

        # If using batchnorm anywhere in the backbone, freeze the batchnorm layer during training.
        # Note: any additional batch norm layers after the backbone will not be frozen.
        self.freeze_bn = False

        # Use the same weights for each network head
        self.share_prediction_module = False,

        # For hard negative mining, instead of using the negatives that are leastl confidently background,
        # use negatives that are most confidently not background.
        self.ohem_use_most_confident = False

        # Use focal loss as described in https://arxiv.org/pdf/1708.02002.pdf instead of OHEM
        self.use_focal_loss = False
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2

        # The initial bias toward forground objects, as specified in the focal loss paper
        self.focal_loss_init_pi = 0.01

        # Keeps track of the average number of examples for each class, and weights the loss for that class accordingly.
        self.use_class_balanced_conf = False

        # Whether to use sigmoid focal loss instead of softmax, all else being the same.
        self.use_sigmoid_focal_loss = False

        # Use class[0] to be the objectness score and class[1:] to be the softmax predicted class.
        # Note: at the moment this is only implemented if use_focal_loss is on.
        self.use_objectness_score = False

        # Adds a global pool + fc layer to the smallest selected layer that predicts the existence of
        # each of the 80 classes. This branch is only evaluated during training time and is just there
        # for multitask learning.
        self.use_class_existence_loss = False
        self.class_existence_alpha = 1

        # Adds a 1x1 convolution directly to the biggest selected layer that predicts a semantic segmentations
        # for each of the 80 classes.
        # This branch is only evaluated during training time and is just there for multitask learning.
        self.use_semantic_segmentation_loss = False
        self.semantic_segmentation_alpha = 1

        # Adds another branch to the netwok to predict Mask IoU.
        self.use_mask_scoring = False
        self.mask_scoring_alpha = 1

        # Match gt boxes using the Box2Pix change metric instead of the standard IoU metric.
        # Note that the threshold you set for iou_threshold should be negative with this setting on.
        self.use_change_matching = False

        # Uses the same network format as mask_proto_net, except this time it's for adding extra head layers
        # before the final prediction in prediction modules. If this is none, no extra layers will be added.
        self.extra_head_net = None

        # What params should the final head layers have (the ones that predict box, confidence, and mask coeffs)
        self.head_layer_params = {'kernel_size': 3, 'padding': 1}

        # Add extra layers between the backbone and the network heads
        # The order is (bbox, conf, mask)
        self.extra_layers = (0, 0, 0)

        # During training, to match detections with gt, first compute the maximum gt IoU for each prior.
        # Then, any of those priors whose maximum overlap is over the positive threshold, mark as positive.
        # For any priors whose maximum is less than the negative iou threshold, mark them as negative.
        # The rest are neutral and not used in calculating the loss.
        self.positive_iou_threshold = 0.5
        self.negative_iou_threshold = 0.5

        # When using ohem, the ratio between positives and negatives (3 means 3 negatives to 1 positive)
        self.ohem_negpos_ratio = 3

        # If less than 1, anchors treated as a negative that have a crowd iou over this threshold with
        # the crowd boxes will be treated as a neutral.
        self.crowd_iou_threshold = 1

        # This is filled in at runtime by Yolact's __init__, so don't touch it
        self.mask_dim = None

        # Input image size.
        self.max_size = 300

        # Whether or not to do post processing on the cpu at test time
        self.force_cpu_nms = True

        # Whether to use mask coefficient cosine similarity nms instead of bbox iou nms
        self.use_coeff_nms = False

        # Whether or not to have a separate branch whose sole purpose is to act as the coefficients for
        # coeff_diversity_loss Remember to turn on coeff_diversity_loss, or these extra coefficients won't do anything!
        # To see their effect, also remember to turn on use_coeff_nms.
        self.use_instance_coeff = False
        self.num_instance_coeffs = 64

        # Whether or not to tie the mask loss / box loss to 0
        self.train_masks = True
        self.train_boxes = True
        # If enabled, the gt masks will be cropped using the gt bboxes instead of the predicted ones.
        # This speeds up training time considerably but results in much worse mAP at test time.
        self.use_gt_bboxes = False

        # Whether or not to preserve aspect ratio when resizing the image.
        # If True, this will resize all images to be max_size^2 pixels in area while keeping aspect ratio.
        # If False, all images are resized to max_size x max_size
        self.preserve_aspect_ratio = False

        # Whether or not to use the prediction module (c) from DSSD
        self.use_prediction_module = False

        # Whether or not to use the predicted coordinate scheme from Yolo v2
        self.use_yolo_regressors = False

        # For training, bboxes are considered "positive" if their anchors have a 0.5 IoU overlap
        # or greater with a ground truth box. If this is true, instead of using the anchor boxes
        # for this IoU computation, the matching function will use the predicted bbox coordinates.
        # Don't turn this on if you're not using yolo regressors!
        self.use_prediction_matching = False

        # A list of settings to apply after the specified iteration. Each element of the list should look like
        # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
        self.delayed_settings = []

        # Use command-line arguments to set this.
        self.no_jit = False

        self.backbone = None
        self.name = 'base_config'

        # Fast Mask Re-scoring Network
        # Inspried by Mask Scoring R-CNN (https://arxiv.org/abs/1903.00241)
        # Do not crop out the mask with bbox but slide a convnet on the image-size mask,
        # then use global pooling to get the final mask score
        self.use_maskiou = False

        # Archecture for the mask iou network. A (num_classes-1, 1, {}) layer is appended to the end.
        self.maskiou_net = []

        # Discard predicted masks whose area is less than this
        self.discard_mask_area = -1

        self.maskiou_alpha = 1.0
        self.rescore_mask = False
        self.rescore_bbox = False
        self.maskious_to_train = -1

    def dict(self):
        return vars(self)
