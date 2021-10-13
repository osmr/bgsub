"""
    Image Human Segmenter.
    Based on:
    - https://github.com/ternaus/people_segmentation,
    - https://github.com/hkchengrex/CascadePSP.
"""

__all__ = ['ImageHumanSegmenter']

import numpy as np
import cv2
from segmentation_models_pytorch import Unet
from torch.utils import model_zoo
import segmentation_refinement as refine
import torch
from utils.image import resize_image, resize_image_with_max_size, pad_image_to_size, pad_image_by_factor, crop_image,\
    normalize_image
from utils.pytorch import rename_layer_names_in_state_dict, create_tensor_from_rgb_image


class ImageHumanSegmenter(object):
    """
    Image Human Segmenter.

    Parameters:
    ----------
    use_cuda : bool, default False
        Whether to use CUDA.
    """
    def __init__(self,
                 use_cuda=False):
        super(ImageHumanSegmenter, self).__init__()
        self.use_cuda = use_cuda
        device = "cuda:0" if use_cuda else "cpu"

        self.segm_net = self.create_human_segm_net()
        if self.use_cuda:
            self.segm_net = self.segm_net.cuda()

        self.refiner = refine.Refiner(device=device)

    def __call__(self, bgr_image: np.array) -> np.array:
        """
        Process an image.

        Parameters:
        ----------
        bgr_image : np.array
            BGR image.

        Returns:
        -------
        np.array
            Binary mask.
        """
        rgb_image = cv2.cvtColor(bgr_image, code=cv2.COLOR_BGR2RGB)

        base_size = 800
        rgb_image, rgb_image_size = resize_image_with_max_size(rgb_image, max_size=base_size)
        rgb_image, rgb_image_pad_params = pad_image_to_size(
            image=rgb_image,
            dst_image_size=(base_size, base_size),
            border_type=cv2.BORDER_CONSTANT)

        padded_image, factor_pad_params = pad_image_by_factor(
            image=rgb_image,
            factor=32,
            border_type=cv2.BORDER_CONSTANT)
        x = normalize_image(image=padded_image)
        x = create_tensor_from_rgb_image(x)
        x = torch.unsqueeze(x, dim=0)

        if self.use_cuda:
            x = x.cuda()

        with torch.no_grad():
            y = self.segm_net(x)[0][0]

        mask = (y > 0).cpu().detach().numpy().astype(np.uint8)
        mask = mask * 255

        mask = self.refiner.refine(rgb_image, mask, fast=True, L=1200)

        mask = crop_image(mask, factor_pad_params)

        mask = crop_image(mask, rgb_image_pad_params)
        mask = resize_image(mask, image_size=rgb_image_size)

        mask = self.refiner.refine(bgr_image, mask, fast=True, L=1800)
        mask = (mask > 127).astype(np.uint8) * 255
        assert (mask.dtype == np.uint8)

        return mask

    @staticmethod
    def create_human_segm_net():
        """
        Create human segmentation Pytorch network.

        Returns:
        -------
        net : nn.Module
            Target net with loaded weights.
        """
        model_weights_url = "https://github.com/ternaus/people_segmentation/releases/download/0.0.1/2020-09-23a.zip"
        net = Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None)
        state_dict = model_zoo.load_url(url=model_weights_url, progress=True, map_location="cpu")["state_dict"]
        state_dict = rename_layer_names_in_state_dict(state_dict=state_dict, layer_name_dict={"model.": ""})
        net.load_state_dict(state_dict)
        net.eval()
        return net
