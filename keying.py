import os
import cv2
import numpy as np
import argparse
import tkinter as tk
from tkinter import filedialog


def info():
    args = argparse.ArgumentParser()
    args.add_argument('--save_name', type=str,
                      default='./result/test_result.png',
                      help='保存的文件名')
    args.add_argument('--threshold_B', type=int,
                      default=200,
                      help='B通道的阈值')
    args.add_argument('--threshold_G', type=int,
                      default=200,
                      help='G通道的阈值')
    args.add_argument('--threshold_R', type=int,
                      default=200,
                      help='R通道的阈值')

    args.add_argument('--save_size', type=tuple,
                      default=(200, 200),
                      help='图片保存的大小，（W，H）')

    parser = args.parse_args()
    return parser


def keying(img_path, args):
    """
    实现抠图功能
    :param img_path: 图片路径
    :param args: 参数
    :return: 无返回
    """

    threshold_B = args.threshold_B
    threshold_G = args.threshold_G
    threshold_R = args.threshold_R
    save_path = args.save_name
    size = args.save_size

    img = cv2_imread(img_path)
    img = cv2.resize(img, size)
    # cv2.imshow('img', img)

    # 如果输入图片是PNG格式，则无需新建通道
    result = img
    if img.shape[-1] == 3:
        result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # print(f'img.shape:{img.shape}')
    # print(f'result.shape:{result.shape}')
    # print(type(img_array), len(img_array), img_array.shape)

    result_array = np.array(result)  # 转换成ndarray格式
    # 分离图片通道
    B, G, R, A = result_array[:, :, 0], result_array[:, :, 1], result_array[:, :, 2], result_array[:, :, -1]
    # 创建一个同格式的通道，用于实现与操作
    E = np.empty_like(B) * 0 + 255

    # 分别获取BGR三通道满足条件的坐标
    B_index = np.where(B > threshold_B)
    G_index = np.where(G > threshold_G)
    R_index = np.where(R > threshold_R)

    # 先 或操作
    E[B_index] = 0
    E[G_index] = 0
    E[R_index] = 0
    # 再 非操作
    E[np.where(R <= threshold_B)] = 255
    E[np.where(G <= threshold_G)] = 255
    E[np.where(B <= threshold_R)] = 255

    # 对 PNG格式 的A通道进行替换，即抠图的区域变透明
    A[np.where(E == 0)] = 0
    result_array[:, :, -1] = A  # 赋值回原图片内

    # print('result')
    # print(result, result != result_array)
    # cv2.imshow('result', result_array)
    # print(save_path)

    # 保存图片
    rec = cv2.imwrite(save_path, result_array, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    if rec:
        print("抠图成功，保存在：{0}".format(save_path))

    # cv2.waitKey(0)


def cv2_imread(img_path):
    """
    opencv读取中文命名的图片
    :param img_path: 图片的路径
    :return: 返回读取的图片
    """
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    return img


def cv2_imwrite(img_path, img):
    """
    opencv保存中文命名的图片
    :param img_path: 图片的路径
    :param img: 需要保存的图片
    :return:
    """
    cv2.imencode('.png', img)[1].tofile(img_path)  # 保存图片


def use1():
    # 实例化
    arg = info()
    root = tk.Tk()
    root.withdraw()
    # 获取文件夹路径
    f_path = filedialog.askopenfilename()
    print('\n获取的文件地址：', f_path)
    jpg_path = f_path
    png_path = arg.save_name
    # print(png_path)
    keying(jpg_path, png_path)


def use2():
    arg = info()
    src_path = "./src"
    save_path = "./result"
    i = 0
    file_name = os.listdir(src_path)

    arg.save_name = os.path.join(save_path, str(i) + '.png')
    # print(file_name)
    for file in file_name:
        file_path = os.path.join(src_path, file)
        # print(file_path, arg.save_name)
        i += 1
        arg.save_name = os.path.join(save_path, str(i) + '.png')
        keying(file_path, arg)


if __name__ == '__main__':
    # use1()
    use2()

