import torch
import cv2
import argparse
import torch.backends.cudnn as cudnn

from yolact.data import cfg, set_cfg
from yolact.utils.functions import SavePath
from yolact.utils.augmentations import BaseTransform, FastBaseTransform
from yolact.utils.visualization import prep_display
from yolact import Yolact


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation', add_help=False)
    parser.add_argument('--cuda', default=True, type=str2bool)
    parser.add_argument('--trained_model', default='weights/yolact_base_54_800000.pth', type=str)
    parser.add_argument('--config', default=None, help='The config object to use.')
    parser.add_argument('--score_threshold', default=0.15, type=float)
    parser.add_argument('--top_k', default=15, type=int)
    parser.add_argument('--images', default="./images/frame001737.jpg", type=str)

    # visualization
    parser.add_argument('--display_lincomb', default=False, type=str2bool)
    parser.add_argument('--display_masks', default=True, type=str2bool)
    parser.add_argument('--display_text', default=True, type=str2bool)
    parser.add_argument('--display_bboxes', default=True, type=str2bool)
    parser.add_argument('--display_scores', default=True, type=str2bool)

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                        shuffle=False, benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False,
                        crop=True, detect=False, display_fps=False, emulate_playback=False)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Yolact evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    with torch.no_grad():
        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        net = net.cuda()
        print(' Done.')

        img = cv2.imread(args.images)
        frame = torch.from_numpy(img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)

        img_numpy = prep_display(preds, frame, None, None, args, undo_transform=False)
        cv2.imshow("img", img_numpy)
        cv2.waitKey()

