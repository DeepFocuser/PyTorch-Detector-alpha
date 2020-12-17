import random
import numpy as np
__all__ = ["image_random_color_distort", "random_flip", "np", "random"]

def image_random_color_distort(src, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                               saturation_low=0.5, saturation_high=1.5, hue_delta=18):

    def brightness(src, delta, p=0.5):
        """Brightness distortion."""
        if np.random.uniform(0, 1) > p:
            delta = np.random.uniform(-delta, delta)
            src += delta
            return src
        return src

    def contrast(src, low, high, p=0.5):
        """Contrast distortion"""
        if np.random.uniform(0, 1) > p:
            alpha = np.random.uniform(low, high)
            src *= alpha
            return src
        return src

    def saturation(src, low, high, p=0.5):
        """Saturation distortion."""
        if np.random.uniform(0, 1) > p:
            alpha = np.random.uniform(low, high)
            gray = src * np.array([[[0.299, 0.587, 0.114]]])
            gray = np.sum(gray, axis=2, keepdims=True)
            gray *= (1.0 - alpha)
            src *= alpha
            src += gray
            return src
        return src

    def hue(src, delta, p=0.5):
        """Hue distortion"""
        if np.random.uniform(0, 1) > p:
            alpha = random.uniform(-delta, delta)
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                           [0.0, u, -w],
                           [0.0, w, u]])
            tyiq = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]])
            ityiq = np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]])
            t = np.dot(np.dot(ityiq, bt), tyiq).T
            src = np.dot(src, t)
            return src
        return src

    src = src.astype('float32')

    # brightness
    src = brightness(src, brightness_delta)

    # color jitter
    if np.random.randint(0, 2):
        src = contrast(src, contrast_low, contrast_high)
        src = saturation(src, saturation_low, saturation_high)
        src = hue(src, hue_delta)
    else:
        src = saturation(src, saturation_low, saturation_high)
        src = hue(src, hue_delta)
        src = contrast(src, contrast_low, contrast_high)
    return src

def random_flip(src, px=0, py=0, copy=False):
    flip_y = np.random.choice([False, True], p=[1 - py, py])
    flip_x = np.random.choice([False, True], p=[1 - px, px])
    if flip_y:
        src = np.flip(src, axis=0)
    if flip_x:
        src = np.flip(src, axis=1)
    if copy:
        src = src.copy()
    return src, (flip_x, flip_y)
