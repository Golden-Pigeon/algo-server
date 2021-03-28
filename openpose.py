#这个脚本返回算法产生的json文件path

import os
from PATH import *

def OpenposeAPI(video_path):
    return json_path


def get_json_file(video_path):
    # colab_video_path = '123.mp4_20210221_234444.mkv'
    openpose_video_path = video_path.replace('.mp4', '') + '-openpose.mp4'

    os.system(OPEN_POSE_BIN + " --number_people_max 12 --render_pose 0 --video " + video_path + " --display 0 --write_json " + openpose_video_path)
    if os.path.exists(openpose_video_path):
        return openpose_video_path
    else:
        return ""

if __name__ == '__main__':
    test()