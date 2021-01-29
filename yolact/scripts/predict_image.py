import torch
import cv2
import argparse
import matplotlib.pyplot as plt

from yolact.data import cfg, set_cfg
from yolact.utils.functions import SavePath
from yolact.utils.augmentations import BaseTransform, FastBaseTransform
from yolact.utils.visualization import prep_display
from yolact import Yolact


def args_parser():
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model', default='weights/yolact_base_54_800000.pth', type=str)
    parser.add_argument('--config', default=None, help='The config object to use.')
    parser.add_argument('--score_threshold', default=0.15, type=float)
    parser.add_argument('--top_k', default=15, type=int)
    parser.add_argument('--images', default="./images/frame001737.jpg", type=str,
                        help='the input folder of images')
    return parser.parse_args()


def evalimage(net: Yolact, path: str, extra_args, save_path: str = None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_numpy = prep_display(preds, frame, None, None, extra_args, undo_transform=False)

    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]

    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)


if __name__ == "__main__":
    args = args_parser()
    print(args)
    if args.config is not None:
        set_cfg(args.config)
    else:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    print('Loading model...', end='')
    net = Yolact()
    net.load_weights(args.trained_model)
    net.eval()
    print(' Done.')

    img = cv2.imread(args.images)
    frame = torch.from_numpy(img).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_numpy = prep_display(preds, frame, None, None, args, undo_transform=False)

    plt.imshow(img)
    plt.title(args.images)
    plt.show()
