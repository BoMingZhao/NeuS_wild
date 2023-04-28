import os
import torchvision.transforms as T
import torch
import numpy as np
import cv2
from PIL import Image
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

def visualize_visibility(lvis, output_path):
    frames = []
    for i in range(lvis.shape[-1]): # for each light pixel
        frame = denormalize_float(lvis[:, :, i])
        frame = np.dstack([frame] * 3)
        frames.append(frame)
    make_video(frames, fps=32, outpath=output_path)

def denormalize_float(arr, uint_type='uint8'):
    r"""De-normalizes the input ``float`` array such that :math:`1` becomes
    the target ``uint`` maximum.
    Args:
        arr (numpy.ndarray): Input array of type ``float``.
        uint_type (str, optional): Target ``uint`` type.
    Returns:
        numpy.ndarray: De-normalized array of the target type.
    """
    if uint_type not in ('uint8', 'uint16'):
        raise TypeError(uint_type)
    maxv = np.iinfo(uint_type).max
    arr_ = arr * maxv
    arr_ = arr_.astype(uint_type)
    return arr_

def make_video(
        imgs, fps=24, outpath=None, method='matplotlib', dpi=96, bitrate=-1):
    """Writes a list of images into a grayscale or color video.
    Args:
        imgs (list(numpy.ndarray)): Each image should be of type ``uint8`` or
            ``uint16`` and of shape H-by-W (grayscale) or H-by-W-by-3 (RGB).
        fps (int, optional): Frame rate.
        outpath (str, optional): Where to write the video to (a .mp4 file).
            ``None`` means
            ``os.path.join(const.Dir.tmp, 'make_video.mp4')``.
        method (str, optional): Method to use: ``'matplotlib'``, ``'opencv'``,
            ``'video_api'``.
        dpi (int, optional): Dots per inch when using ``matplotlib``.
        bitrate (int, optional): Bit rate in kilobits per second when using
            ``matplotlib``; reasonable values include 7200.
    Writes
        - A video of the images.
    """

    assert imgs, "Frame list is empty"
    for frame in imgs:
        assert np.issubdtype(frame.dtype, np.unsignedinteger), \
            "Image type must be unsigned integer"

    h, w = imgs[0].shape[:2]
    for frame in imgs[1:]:
        assert frame.shape[:2] == (h, w), \
            "All frames must have the same shape"

    if method == 'matplotlib':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import animation

        w_in, h_in = w / dpi, h / dpi
        fig = plt.figure(figsize=(w_in, h_in))
        Writer = animation.writers['ffmpeg'] # may require you to specify path
        writer = Writer(fps=fps, bitrate=bitrate)

        def img_plt(arr):
            img_plt_ = plt.imshow(arr)
            ax = plt.gca()
            ax.set_position([0, 0, 1, 1])
            ax.set_axis_off()
            return img_plt_

        anim = animation.ArtistAnimation(fig, [(img_plt(x),) for x in imgs])
        anim.save(outpath, writer=writer)
        # If obscure error like "ValueError: Invalid file object: <_io.Buff..."
        # occurs, consider upgrading matplotlib so that it prints out the real,
        # underlying ffmpeg error

        plt.close('all')

    elif method == 'opencv':

        # TODO: debug codecs (see http://www.fourcc.org/codecs.php)
        if outpath.endswith('.mp4'):
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # fourcc = cv2.VideoWriter_fourcc(*'X264')
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            # fourcc = 0x00000021
        elif outpath.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            raise NotImplementedError("Video type of\n\t%s" % outpath)

        vw = cv2.VideoWriter(outpath, fourcc, fps, (w, h))

        for frame in imgs:
            if frame.ndim == 3:
                frame = frame[:, :, ::-1] # cv2 uses BGR
            vw.write(frame)

        vw.release()

    else:
        raise ValueError(method)