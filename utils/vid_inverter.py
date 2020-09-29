"""Utility functions to invert frames in a video back to latent code"""
from tqdm import tqdm
import cv2
import numpy as np

import torch

from models.stylegan_generator import StyleGANGenerator
from models.stylegan_encoder import StyleGANEncoder
from models.perceptual_model import PerceptualModel
from utils.inverter import StyleGANInverter

__all__ = ['StyleGANVidInverter']

def _get_tensor_value(tensor):
    """Gets the value of a torch Tensor."""
    return tensor.cpu().detach().numpy()


class StyleGANVidInverter(StyleGANInverter):
    """Defines the class for StyleGAN inversion.

    Even having the encoder, the output latent code is not good enough to recover
    the target image satisfyingly. To this end, this class optimize the latent
    code based on gradient descent algorithm. In the optimization process,
    following loss functions will be considered:

    (1) Pixel-wise reconstruction loss. (required)
    (2) Perceptual loss. (optional, but recommended)
    (3) Regularization loss from encoder. (optional, but recommended for in-domain
      inversion)

    NOTE: The encoder can be missing for inversion, in which case the latent code
    will be randomly initialized and the regularization loss will be ignored.
    """

    def __init__(self,
               model_name,
               learning_rate=1e-2,
               iteration=100,
               reconstruction_loss_weight=1.0,
               perceptual_loss_weight=5e-5,
               regularization_loss_weight=2.0,
               logger=None):
        """Initializes the inverter.

        NOTE: Only Adam optimizer is supported in the optimization process. We can try other optimizer later.

        Args:
          model_name: Name of the model on which the inverted is based. The model
            should be first registered in `models/model_settings.py`.
          logger: Logger to record the log message.
          learning_rate: Learning rate for optimization. (default: 1e-2)
          iteration: Number of iterations for optimization. (default: 100)
          reconstruction_loss_weight: Weight for reconstruction loss. Should always
            be a positive number. (default: 1.0)
          perceptual_loss_weight: Weight for perceptual loss. 0 disables perceptual
            loss. (default: 5e-5)
          regularization_loss_weight: Weight for regularization loss from encoder.
            This is essential for in-domain inversion. However, this loss will
            automatically ignored if the generative model does not include a valid
            encoder. 0 disables regularization loss. (default: 2.0)
        """
        super(StyleGANVidInverter, self).__init__(
                model_name,
               learning_rate=1e-2,
               iteration=100,
               reconstruction_loss_weight=1.0,
               perceptual_loss_weight=5e-5,
               regularization_loss_weight=2.0,
               logger=None
        )

    def vid_invert(self, images):
        """
            Invert given frames.
            Images: a list of images
            
        """
        zs = []  # to store latent code z for each image
        for image in images:
            x = image[np.newaxis]  # TODO
            x = self.G.to_tensor(x.astype(np.float32))
            x.requires_grad = False
            init_z = self.get_init_code(image)
            z = torch.Tensor(init_z).to(self.run_device)
            zs.append(z)
        
        optimizer = torch.optim.Adam(zs, lr=self.learning_rate)
        
        pbar = tqdm(range(1, self.iteration + 1), leave=True)
        
        for itr in range(len(images)):
            loss = 0.0
            for step in pbar:
                
            
            
        pass