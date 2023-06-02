import torch
import json
import cv2
import yaml
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from loguru import logger

from core.utils import to_tensors
from datasets.epic_kitchen import EpicKitchenSub
from datasets.misc import collate_fn_epicsub


def get_parser() -> None:
    parser = argparse.ArgumentParser(description="E2FGVI")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, choices=['e2fgvi_hq'])
    # parser.add_argument("--neighbor_stride", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--basedir",
                        default="/mnt/seagate12t/EPIC-KITCHEN/EPIC-KITCHEN",)
    parser.add_argument("--part", type=int, required=True)

    # args for e2fgvi_hq (which can handle videos with arbitrary resolution)
    # parser.add_argument("--set_size", action='store_true', default=False)
    # parser.add_argument("--width", type=int)
    # parser.add_argument("--height", type=int)

    return parser


def main() -> None:
    _DEVICE = "cuda"
    parser = get_parser()
    args = parser.parse_args()
    args.part = f'P{int(args.part):02d}'

    logger.info(f"Arguments Configuration: \n {yaml.dump(vars(args))}")

    net = importlib.import_module('model.' + args.model)
    logger.info(f"Imported {args.model} model")
    model = net.InpaintGenerator().to(_DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=_DEVICE))
    model.eval()

    # itreate over all clips in a part
    info = json.load(open(os.path.join(args.basedir, "info.json")))[args.part]
    for clip in info.keys():
        logger.info(f"Processing [{clip}] clip in part [{args.part}]")
        # prepare dataset for the clip
        dataloader = EpicKitchenSub(basedir=args.basedir, part=args.part, clip=clip,
                                    ).get_dataloader(batch_size=args.batch_size,
                                                     collate_fn=collate_fn_epicsub,
                                                     num_workers=8,
                                                     pin_memory=True,
                                                     shuffle=False,
                                                     )
        length_frames_per_ins = dataloader.dataset._LENGTH_FRAMES_PER_INS
        with torch.no_grad():
            for i_b, batch in enumerate(dataloader):
                logger.info(f"Processing batch [{i_b}] of {len(dataloader)}")
                masked_frames = batch["masked_imgs"].to(_DEVICE) # (B, T, C, H, W)
                outpaths = batch["outpaths"]
                pred_frames, _ = model(masked_frames, length_frames_per_ins)
                pred_frames = pred_frames.reshape(*masked_frames.shape)
                pred_frames = pred_frames[:, :, :, :dataloader.dataset._HEIGHT, :dataloader.dataset._WIDTH]
                pred_frames = (pred_frames + 1) / 2
                # save the results
                pred_frames = pred_frames.cpu().numpy()
                for i_ins in range(pred_frames.shape[0]):
                    ins_outpaths = outpaths[i_ins]
                    for i_frame in range(pred_frames.shape[1]):
                        i_frame_outpath = ins_outpaths[i_frame]
                        frame = pred_frames[i_ins, i_frame]
                        frame = np.transpose(frame, (1, 2, 0))
                        frame = (frame * 255).astype(np.uint8)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        # cv2.imshow('Image', frame)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        cv2.imwrite(i_frame_outpath, frame)


if __name__ == '__main__':
    main()