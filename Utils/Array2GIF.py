from PIL import Image
import numpy as np


def array2gif(array: [np.ndarray], filename: str, duration:[float,int]=0.1):
    imgs = []
    print('array shape:', array[-1].shape)
    print('array dtype:', array[-1].dtype)
    print('frames number:', len(array))
    for i, frame in enumerate(array):
        # frame = frame[:,:,::-1]
        # im = Image.fromarray((frame * 255 % 255).astype(np.uint8))
        im = Image.fromarray(frame)
        imgs.append(im)

    imgs[0].save(filename, save_all=True, append_images=imgs, duration=duration, transparency=0, disposal=2)


def array2gif2(array: [np.ndarray], filename: str, duration: [float, int] = 0.1):
    imgs = []
    print('array shape:', array[-1].shape)
    print('array dtype:', array[-1].dtype)
    print('frames number:', len(array))

    for i, frame in enumerate(array):
        # 将透明度转换为白色（或其他背景色）
        # 假设透明度为0的部分是背景
        if frame.shape[2] == 4:  # 如果有Alpha通道
            alpha = frame[:, :, 3] / 255.0  # 归一化Alpha通道
            frame_rgb = frame[:, :, :3]  # 提取RGB通道
            background = np.ones_like(frame_rgb) * 255  # 白色背景
            frame = (frame_rgb * alpha[..., None] + background * (1 - alpha[..., None])).astype(np.uint8)

        # 创建PIL图像
        im = Image.fromarray(frame)
        imgs.append(im)

    # 保存为GIF
    imgs[0].save(filename, save_all=True, append_images=imgs, duration=duration, transparency=0, disposal=2)