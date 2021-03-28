from matplotlib.animation import FuncAnimation, writers
import logging
from IPython.display import HTML
import pdb
import json
import os
import scipy.signal as signal
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from typing import (List, Dict)

matplotlib.rcParams['animation.embed_limit'] = 2 ** 128

names = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel"
]
dic_from_names_to_index = {x: i for i, x in enumerate(names)}
colors = [[1.0, 0.0, 0.0],
          [1.0, 0.3333333333333333, 0.0],
          [1.0, 0.6666666666666666, 0.0],
          [1.0, 1.0, 0.0],
          [0.6666666666666666, 1.0, 0.0],
          [0.3333333333333333, 1.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 1.0, 0.3333333333333333],
          [0.0, 1.0, 0.6666666666666666],
          [0.0, 1.0, 1.0],
          [0.0, 0.6666666666666666, 1.0],
          [0.0, 0.3333333333333333, 1.0],
          [0.0, 0.0, 1.0],
          [0.3333333333333333, 0.0, 1.0],
          [0.6666666666666666, 0.0, 1.0],
          [1.0, 0.0, 1.0],
          [1.0, 0.0, 0.6666666666666666],
          [1.0, 0.0, 0.3333333333333333],
          [1.0, 1.0, 0.0],
          [1.0, 1.0, 0.3333333333333333],
          [1.0, 1.0, 0.6666666666666666],
          [1.0, 1.0, 1.0],
          [0.6666666666666666, 1.0, 1.0],
          [0.3333333333333333, 1.0, 1.0],
          [0.0, 1.0, 1.0]]
Body_25_Pairs = [(1, 2),
                 (1, 5),
                 (2, 3),
                 (3, 4),
                 (5, 6),
                 (6, 7),
                 (1, 8),
                 (8, 9),
                 (9, 10),
                 (8, 12),
                 (10, 11),
                 (11, 22),
                 (11, 24),
                 (22, 23),
                 (12, 13),
                 (1, 0),
                 (13, 14),
                 (14, 21),
                 (14, 19),
                 (0, 15),
                 (0, 16),
                 (15, 17),
                 (16, 18),
                 (19, 20)]


# 装饰器统计函数运行时间
def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result

    return func_wrapper


def from_path_get_poseList(path):
    '''从json path提取2d位置为list'''
    with open(path) as f:
        t = json.loads(f.read())
        try:
            return t['people'][0]['pose_keypoints_2d']
        except IndexError:
            # 忽略IndexError
            return None
        else:
            return None


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'valid')
    return re


def from_partIndex_get_poseTuple(index, list, only_y=False):
    '''获得对于身体部位的坐标'''
    position = (index) * 3
    return (list[position], list[position + 1], list[position + 2])


def completion_path(index) -> str:
    return str(index).rjust(3, '0')


def get_pack_from_aniPose(ani):
    '''将x，y分别提取成list
    0->x,1->y'''
    return list(zip(*ani))


def get_pose_y(index, all_list):
    '''index为dict里的index'''
    return np.array(get_pack_from_aniPose(all_list)[index * 3 + 1])


def get_mean_pose_position(json_list):
    if isinstance(json_list, list):
        json_list = np.array(json_list)
    return json_list.mean(axis=1)


def shift(json_list, json_stander, POSITION='RWrist'):
    '''相对标准视频的json list，做归一化，这里以MidHip为标准'''
    mean = get_mean_pose_position(json_list)
    mean_s = get_mean_pose_position(json_stander)
    global dic_from_names_to_index
    x, y, _ = from_partIndex_get_poseTuple(dic_from_names_to_index[POSITION], mean)
    xs, ys, _ = from_partIndex_get_poseTuple(dic_from_names_to_index[POSITION], mean_s)
    shifted_json_list = json_list
    for i in range(len(json_list)):
        for k in range(25):
            shifted_json_list[i][3 * k] -= (x - xs)
            shifted_json_list[i][3 * k + 1] -= (y - ys)
    return shifted_json_list


def split(json_list):
    '''先以10窗口平滑，再7窗口二次平滑,效果大概百分之七八十，还带写更多if-else手动判断
    由于openpose结果震荡，俯卧撑判断正确度大概80%'''
    single_pose = get_pose_y(dic_from_names_to_index["MidHip"], json_list)
    # 去0值
    single_pose[single_pose == 0] = single_pose.mean()
    windowsize = 10
    guassed = moving_average(single_pose, windowsize)
    guassed = moving_average(guassed, 7)
    extre_points, = signal.argrelextrema(guassed, np.greater)
    correct_points = [extre_points[0]]
    for i in range(len(extre_points) - 1):
        if extre_points[i + 1] - correct_points[-1] >= 10:
            correct_points.append(extre_points[i + 1])
    return np.array(correct_points)


# 计算夹角
def Angle(vec1, vec2, deg=True):
    _angle = np.arctan2(np.cross(vec1, vec2), np.dot(vec1, vec2))
    if deg:
        _angle = np.rad2deg(_angle)
    return _angle


def from_3Index_get_Angle(index, json_timestep):
    '''计算向量（1,0）与（1,2）之间的夹角'''
    Pose = []
    for i in range(3):
        Pose += [from_partIndex_get_poseTuple(index[i], json_timestep)]
    vec1 = np.array([Pose[0][0] - Pose[1][0], Pose[0][1] - Pose[1][1]])
    vec2 = np.array([Pose[2][0] - Pose[1][0], Pose[2][1] - Pose[1][1]])
    return Angle(vec1, vec2)


def Outliers_clean(data_list, threshold=50):
    '''清除list中突变的数据点'''

    # pose_list是整个序列（列表），threshold表示判断一个点是否为异常点的阈值（该点和前一个时间的数值的差值）
    def chazhi(chazhi_list):
        # 作为确定值的插值点
        n = len(chazhi_list)
        x = [i for i in range(n)]
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i][j] = x[i] ** (n - 1 - j)
        B = np.zeros((n, 1))
        for b in range(n):
            B[b][0] = chazhi_list[b]
        a = np.linalg.solve(A, B)  # 返回多项式插值函数的参数

        def f(a, x):
            f = 0
            for m in range(len(a)):
                f += a[m][0] * x ** (n - m - 1)
            return f

        return f(a, n + 1)  # 返回下一个点（异常点）的插值

    # 默认第0个点是正常的
    for i in range(1, len(data_list)):
        delta = abs(data_list[i] - data_list[i - 1])
        # 如果第1,2个点异常，默认为前一个点的值
        if i < 3 and delta > threshold:
            data_list[i] = data_list[i - 1]
            continue
        # 如果后续的第i个点异常，通过前三个点得到的插值函数计算出该点的值
        if delta > threshold:
            data_list[i] = chazhi(data_list[i - 2:i])
    return data_list


def Clean_outline(line, n):
    line = np.array(line)
    for i in range(n):
        outNum = line.diff().abs().idxmax()
        line[outNum] = line[outNum - 1]


def getAllsinglePose_y(user_list, pose="MidHip", clean=1):
    want = get_pack_from_aniPose(user_list)[dic_from_names_to_index[pose] * 3 + 1]
    if clean:
        want = moving_average(Outliers_clean(list(want)), 4)
    return want


class abnormality():
    def __init__(self, u, a):
        self.u = u
        self.a = a
        self.bias = 1 / stats.norm.pdf(0)

    def __call__(self, x):
        return stats.norm.pdf((x - self.u) / self.a) * self.bias


# def split(json_list):
#     '''分割json list 为不同的运动周期，还是以MidHip为标准，不过还是有待提升，提取后需要手动查看分割的各个index'''
#     single_pose = get_pose_y(dic_from_names_to_index["MidHip"],json_list)
#     split_cut_min = np.sort(single_pose)[int(len(single_pose)*0.3)]#去除异常点
#     normal_min = 400
#     split_cut_max = np.sort(single_pose)[-int(len(single_pose)*0.3)]
#     normal_max = 1000
#     extre_points, = signal.argrelextrema(single_pose,np.greater)
#     return extre_points[((single_pose[extre_points]>=split_cut_max) & (single_pose[extre_points]<=normal_max))]#使用极大极值点分割
if __name__ == '__main__':
    # json_list = np.array([0, 20, 30, 40, 50, 70, 90, 120, 150, 154, 159, 162, 160, 155, 158, 155, 140, 120, 110])
    # print(split(json_list))
    pass
