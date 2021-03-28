from util import *


class Yangwoqizuo:
    def __init__(self, now_json_path, std_json_path=r'BodyPose\json_data\tiaoyuan.correct-openpose.mp4#'):
        self.std_json = std_json_path
        self.user_json_path = now_json_path
        # assert os.path.exists(self.std_json)

    def data_process(self):
        _, _, filenames1 = next(os.walk(self.user_json_path))
        filenames1 = [os.path.join(self.user_json_path, i) for i in filenames1 if i[-4:] == 'json']
        pose_list_all1 = []
        for filename in filenames1:
            pose_list = from_path_get_poseList(filename)
            if pose_list is not None:
                pose_list_all1.append(pose_list)
            # pose_list_all1 = pose_list_all1[50:cut_down+50]
        pose_list_all2 = None
        #     _, _, filenames2 = next(os.walk(self.std_json))
        # filenames2 = [os.path.join(self.std_json, i) for i in filenames2 if i[-4:] == 'json']
        # pose_list_all2 = []
        # for filename in filenames2:
        #     pose_list = from_path_get_poseList(filename)
        #     if pose_list is not None:
        #         pose_list_all2.append(pose_list)
        # # pose_list_all2 = pose_list_all2[50:cut_down+50]

        # pose_list_all1 = np.array(pose_list_all1)
        # pose_list_all2 = np.array(pose_list_all2)

        return pose_list_all2, pose_list_all1

    def Shouzhou(self, json_list, dis_threshold=30, acc_th=0.9):
        knee_idx = dic_from_names_to_index['LKnee']
        elbow_idx = dic_from_names_to_index['LElbow']

        dis = []
        for json_timestep in json_list:
            A0, A1, _ = from_partIndex_get_poseTuple(knee_idx, json_timestep)
            B0, B1, _ = from_partIndex_get_poseTuple(elbow_idx, json_timestep)
            dis.append(np.sqrt((A0 - B0) ** 2 + (A1 - B1) ** 2))
        dis = moving_average(Outliers_clean(dis, threshold=50), 4)
        dis = np.array(dis)
        split_index, = signal.argrelextrema(dis, np.less)
        cnt_correct = np.count_nonzero(dis[split_index] <= dis_threshold)
        if cnt_correct < acc_th * split_index.shape[0]:
            # TODO 多个最大值可能出错
            return 1, split_index[np.argmax(dis[split_index])], 0
        else:
            return 0, None, 1

    def Jianbang(self, json_list, height_threshold=20, acc_threshold=0.7):  # 肩膀没触地
        # height_threshold肩膀高度必须低于的阈值，acc_threshold 合格次数占比的阈值
        jianbang_index = dic_from_names_to_index["LShoulder"]
        hip_index = dic_from_names_to_index["LHip"]
        height = []
        for json_timestep in json_list:
            _, height_y, _ = from_partIndex_get_poseTuple(jianbang_index, json_timestep)
            _, height_hip, _ = from_partIndex_get_poseTuple(hip_index, json_timestep)
            height.append(height_hip - height_y)  # 获得左肩的y值的时间序列

        height = np.array(height)
        windowsize = 10
        height = moving_average(Outliers_clean(height, 50), windowsize)
        index_list = signal.argrelextrema(height, np.less)  # 肩膀到达最低点的高度时的坐标
        height_list = height[index_list]  # 肩膀到达最低点时肩膀和屁股的高度值差
        wrong_index = height_list.argmax()  # 错误的帧数
        accuracy = 0  # 准确次数占比
        correct_list = [i <= height_threshold for i in height_list]
        accuracy = len(correct_list) / len(height_list)
        if accuracy >= acc_threshold:
            return 0, None, 1
        else:
            return 1, wrong_index, 0

    def Jiaolidi(self, json_list, height_threshold=30):  # 脚离地,height_threshold脚离地高度的阈值
        Lheel_index = dic_from_names_to_index["LHeel"]
        height = []
        for json_timestep in json_list:
            _, height_y, _ = from_partIndex_get_poseTuple(Lheel_index, json_timestep)
            height.append(height_y)  # 获得左脚的y值的时间序列值
        wrong_index = height.index(min(height))  # 脚离地最高的帧数

        if max(height) - min(height) < height_threshold:
            return 0, None, 1
        else:
            return 1, wrong_index, 0

    def split(self, json_list):
        head_idx = dic_from_names_to_index['Nose']
        height = []
        for json_timestep in json_list:
            height.append(from_partIndex_get_poseTuple(json_timestep, head_idx)[1])
        head_idx = moving_average(Outliers_clean(head_idx, threshold=50), 4)
        head_idx = np.array(head_idx)
        split_index = signal.argrelextrema(head_idx, np.less)
        return split_index

    # def split_fall(self, pose_list_all, jump_index):

    def judge(self):
        METHOD = [self.Jianbang, self.Shouzhou, self.Jiaolidi]
        total_error = ['肩未触地', '手肘没碰到膝盖', '脚离地']

        std_list, user_list = self.data_process()

        # for method in METHOD:
        #     method(Qitiao_list)
        error_res = ''
        advice_res = ''
        error_idx = []

        errors = []
        fitness = 0
        for i, method in enumerate(METHOD):
            state, idx, fit = method(user_list)
            if state:
                errors.append([total_error[i], idx])
            fitness += fit
        if len(errors) == 0:
            error_res = '未检测到问题。'
            advice_res = '您的动作很标准。'
        else:
            error_res = '您动作中的问题有：\n'
            for s in errors:
                error_res += s[0] + '\n'
                error_idx.append(s[1])
            if len(errors) / len(total_error) <= 0.4:
                advice_res = '您的动作稍有不标准，希望您能多加练习。'
            else:
                advice_res = '您的动作问题较多，请参照提供视频多加练习!'
        fitness = fitness / len(METHOD)
        return error_res, advice_res, error_idx, fitness



if __name__ == '__main__':
    action = Yangwoqizuo(
        r'C:\Users\lmsZs\Desktop\Desktop\Projects\python\BodyPose\json_data\tiaoyuan\tiaoyuan.correct-openpose.mp4#')
    res = action.judge()
    print(res[0])
    print((res[1]))
    print((res[2]))
# ani=from_json_plot()
# ani2 = from_2json_plot()
# HTML(ani2.to_jshtml()
