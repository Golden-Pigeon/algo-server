import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

matplotlib.rcParams['animation.embed_limit'] = 2 ** 128
from matplotlib.animation import FuncAnimation, writers
import logging
from IPython.display import HTML
import pdb
import json
import os

print(os.getcwd())

from util import *


def from_json_plot(
        json_file=r'C:\Users\lmsZs\Desktop\Desktop\Projects\python\BodyPose\json_data\tiaoyuan\tiaoyuan.correct-openpose.mp4#',
        save_path='./test.html',
        time_start=0,
        cut_down=1000):
    '''从含有json文件的文件夹中提取并画图，返回html文件等视频动图'''
    _, _, filenames = next(os.walk(json_file))
    filenames = [os.path.join(json_file, i) for i in filenames if i[-4:] == 'json']
    pose_list_all = []
    for filename in filenames:
        pose_list = from_path_get_poseList(filename)
        if pose_list is not None:
            pose_list_all.append(pose_list)
    end = time_start + cut_down if time_start + cut_down <= len(pose_list_all) else len(pose_list_all) - 1
    pose_list_all = pose_list_all[time_start:end]
    fig = plt.figure(figsize=(15, 12))
    xlim = 1280
    ylim = -720
    plt.xlim(0, xlim)
    plt.ylim(ylim, 0)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, xlim), ylim=(ylim, 0))

    def animate(i):
        global Body_25_Pairs
        global colors
        ax.cla()
        ax.set_xlim(0, xlim)
        ax.set_ylim(ylim, 0)
        pose_list = pose_list_all[i]
        # 每一帧的所有pose
        for pose_part in Body_25_Pairs:
            # 每一个关节连接点
            x1, y1 = pose_list[pose_part[0] * 3], pose_list[pose_part[0] * 3 + 1]
            x2, y2 = pose_list[pose_part[1] * 3], pose_list[pose_part[1] * 3 + 1]
            if x1 and x2 and y1 and y2:
                # -y让图方向倒过来
                ax.plot([x1, x2], [-y1, -y2], 'o-', lw=2, color=colors[pose_part[0]])
                ax.text(10, 10, str(i))

    frame_num = len(pose_list_all)
    ani = FuncAnimation(fig, animate, frames=range(frame_num), interval=300)
    # print('Begin saving gif')
    # ani.save('test.gif', writer='imagemagick', fps=None)
    # print('Finished.')
    with open(save_path, 'w') as f:
        # f.write('<!DOCTYPE html> <html> <head> <meta charset="UTF-8"> <title>Test</title> </head> <body> ')
        f.write(ani.to_html5_video())
    return ani


def from_2json_plot(
        json_file1='./json_data/hip/openpose/output.json',
        json_file2='./json_data/Std1-openpose.mp4#-20210131T074012Z-001/Std1-openpose.mp4#',
        save_path='./test_hip.html',
        video='html'):
    '''同上，但画两个图'''
    # TODO: 范围, -y， ect
    _, _, filenames1 = next(os.walk(json_file1))
    filenames1 = [os.path.join(json_file1, i) for i in filenames1 if i[-4:] == 'json']
    pose_list_all1 = []
    for filename in filenames1:
        pose_list = from_path_get_poseList(filename)
        if pose_list is not None:
            pose_list_all1.append(pose_list)

        _, _, filenames2 = next(os.walk(json_file2))
    filenames2 = [os.path.join(json_file2, i) for i in filenames2 if i[-4:] == 'json']
    pose_list_all2 = []
    for filename in filenames2:
        pose_list = from_path_get_poseList(filename)
        if pose_list is not None:
            pose_list_all2.append(pose_list)

    pose_list_all1 = np.array(pose_list_all1)
    pose_list_all2 = np.array(pose_list_all2)
    slices = split(pose_list_all1)
    # 在这里调试中断，查看slices，手动判断选取哪两个index作为分割帧
    slice1_fir = slices[0]
    slice1_end = slices[-1]
    slice2 = split(pose_list_all2)
    slice2_fir = slices[0]
    slice2_end = slices[-1]
    # 抽出动图可以看出，确实都是两个俯卧撑周期，而且时间帧数完全一样，效果较好，但openpose结果抖动太大
    pose_list_all1 = pose_list_all1[slice1_fir:slice1_end]
    pose_list_all2 = pose_list_all2[slice2_fir:slice2_end]
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 2000), ylim=(0, 1000))

    def animate(i):
        global Body_25_Pairs
        global colors
        ax.cla()
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 1000)
        pose_list1 = pose_list_all1[i]
        pose_list2 = pose_list_all2[i]
        # 每一帧的所有pose
        for pose_part in Body_25_Pairs:
            # 每一个关节连接点
            pose_list = pose_list1
            x1, y1 = pose_list[pose_part[0] * 3], pose_list[pose_part[0] * 3 + 1]
            x2, y2 = pose_list[pose_part[1] * 3], pose_list[pose_part[1] * 3 + 1]
            if (x1 and x2 and y1 and y2):
                ax.plot([x1, x2], [y1, y2], 'o-', lw=2, color='red')  # color=colors[pose_part[0]])
        for pose_part in Body_25_Pairs:
            pose_list = pose_list2
            x1, y1 = pose_list[pose_part[0] * 3], pose_list[pose_part[0] * 3 + 1]
            x2, y2 = pose_list[pose_part[1] * 3], pose_list[pose_part[1] * 3 + 1]
            if (x1 and x2 and y1 and y2):
                ax.plot([x1, x2], [y1, y2], 'o-', lw=2, color='blue')  # colors[pose_part[0]])

    frame_num = min(len(pose_list_all1), len(pose_list_all2))
    ani = FuncAnimation(fig, animate, frames=range(frame_num), interval=250)
    if video != 'html':
        print('Begin saving gif')
        ani.save('test.gif', writer='imagemagick', fps=None)
        print('Finished.')
    else:
        with open(save_path, 'w') as f:
            # f.write('<!DOCTYPE html> <html> <head> <meta charset="UTF-8"> <title>Test</title> </head> <body> ')
            f.write(ani.to_html5_video())
    return ani


if __name__ == '__main__':
    # from_json_plot(save_path='./test3.html')
    ani2 = from_json_plot()
    # HTML(ani2.to_jshtml())
