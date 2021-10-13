"""
    Matting Background Subtractor.
    Based on https://github.com/PeterL1n/BackgroundMattingV2.
"""

__all__ = ['MatthingBgSubtractor']

import numpy as np
from torch.utils import model_zoo
from utils.pytorch import create_tensor_from_rgb_image


class MatthingBgSubtractor(object):
    """
    Matting Background Subtractor.

    Parameters:
    ----------
    threshold : int, default 127
        Threshold for matting mask binarization.
    use_cuda : bool, default False
        Whether to use CUDA.
    """
    def __init__(self,
                 threshold=127,
                 use_cuda=False):
        super(MatthingBgSubtractor, self).__init__()
        self.threshold = threshold
        self.use_cuda = use_cuda

        model_pt_url = "https://github.com/osmr/bgsub/releases/download/v0.0.1/bg_mattingv2_torchscript_resnet50_fp32-0000-5083f7ac.pth.zip"  # noqa
        self.net = model_zoo.load_url(url=model_pt_url, progress=True, map_location="cpu")

        if self.use_cuda:
            self.net = self.net.cuda()

        self.net = self.net.eval()

    def __call__(self,
                 image: np.array,
                 bg_image: np.array) -> np.array:
        """
        Process an image.

        Parameters:
        ----------
        image : np.array
            Processed image.
        bg_image : np.array
            Background image.

        Returns:
        -------
        np.array
            Binary mask.
        """
        image = create_tensor_from_rgb_image(image)
        bg_image = create_tensor_from_rgb_image(bg_image)

        if self.use_cuda:
            image = image.cuda()
            bg_image = bg_image.cuda()

        image = image.unsqueeze(0)
        bg_image = bg_image.unsqueeze(0)

        if (image.size(2) <= 2048) and (image.size(3) <= 2048):
            self.net.backbone_scale = 1 / 4
            self.net.refine_sample_pixels = 80_000
        else:
            self.net.backbone_scale = 1 / 8
            self.net.refine_sample_pixels = 320_000

        pha = self.net(image, bg_image)[0]

        mask = pha[0].mul(255).byte().cpu().numpy().transpose((1, 2, 0))
        mask = (mask[:, :, 0] > self.threshold).astype(np.uint8) * 255
        assert (mask.dtype == np.uint8)

        return mask
