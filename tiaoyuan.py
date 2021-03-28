from util import *
import pandas as pd


class Tiaoyuan:
    def __init__(self, now_json_path, std_json_path='BodyPose\json_data\tiaoyuan.correct-openpose.mp4#'):
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

    def Quxi(self, json_list, angle_threshold=90):
        Quxi_abnormality = abnormality(angle_threshold, 10 * len(str(abs(angle_threshold))))
        Quxi_index = [dic_from_names_to_index["MidHip"], dic_from_names_to_index["LKnee"],
                      dic_from_names_to_index["LAnkle"]]
        angle = []
        for json_timestep in json_list:
            angle.append(from_3Index_get_Angle(Quxi_index, json_timestep))
        angle = Outliers_clean(angle)
        MinAngle = min(angle)
        if MinAngle > angle_threshold:
            return 1, angle.index(MinAngle), Quxi_abnormality(MinAngle)
        else:
            return 0, None, 1

    def Xiongkou(self, json_list, angle_threshold=50):
        xiougkou_abnormality = abnormality(angle_threshold, 10 * len(str(abs(angle_threshold))))
        Xiongkou_index = [dic_from_names_to_index["LKnee"], dic_from_names_to_index["MidHip"],
                          dic_from_names_to_index["Neck"]]
        angle = []
        for json_timestep in json_list:
            angle.append(from_3Index_get_Angle(Xiongkou_index, json_timestep))
        angle = Outliers_clean(angle)
        MinAngle = min(angle)
        if MinAngle < angle_threshold:
            return 1, angle.index(MinAngle), xiougkou_abnormality(MinAngle)
        else:
            return 0, None, 1

    def Zhongxinggao(self, json_list, angle_threshold=100):
        Zhongxinggao_index = [dic_from_names_to_index["LKnee"], dic_from_names_to_index["MidHip"],
                              dic_from_names_to_index["Neck"]]
        angle = []
        for json_timestep in json_list:
            angle.append(from_3Index_get_Angle(Zhongxinggao_index, json_timestep))
        angle = Outliers_clean(angle)
        MinAngle = min(angle)
        if MinAngle > angle_threshold:
            return 1, angle.index(MinAngle), 0
        else:
            return 0, None, 1

    def split(self, pose_list_all, height_threshold=10):
        init_height = (pose_list_all[0][19 * 3 + 1] + pose_list_all[0][22 * 3 + 1]) / 2  # 左右脚大拇指的平均高度
        div1 = None  # 起跳帧号
        div2 = None  # 落地帧号
        for i in range(len(pose_list_all)):
            toe_height = (pose_list_all[i][19 * 3 + 1] + pose_list_all[i][22 * 3 + 1]) / 2
            if init_height - toe_height >= height_threshold:
                div1 = i
                break
        if div1 is None:
            return None, None
        last_height = (pose_list_all[div1][19 * 3 + 1] + pose_list_all[div1][22 * 3 + 1]) / 2
        for i in range(div1, len(pose_list_all)):
            toe_height = (pose_list_all[i][19 * 3 + 1] + pose_list_all[i][22 * 3 + 1]) / 2
            if last_height - toe_height >= 0:
                div2 = i
                break
            else:
                last_height = toe_height
        return div1, div2

    # def split_fall(self, pose_list_all, jump_index):

    def judge(self):
        '''
        跳远的只输入一个运动周期，分割起跳，跳中，落地三个阶段
        '''

        METHOD = [self.Quxi, self.Xiongkou, self.Zhongxinggao]
        total_error = ['屈膝幅度过大', '起跳时胸部过低', '重心过高']

        std_list, user_list = self.data_process()

        Qitiao_idx, _ = self.split(user_list)
        Qitiao_list = user_list[:Qitiao_idx]
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
    action = Tiaoyuan(
        r'C:\Users\lmsZs\Desktop\Desktop\Projects\python\BodyPose\json_data\tiaoyuan\outputtiaoyuan_liyudagun\openpose\output.json#')
    res = action.judge()
    print(res[0])
    print((res[1]))
    print((res[2]))
# ani=from_json_plot()
# ani2 = from_2json_plot()
# HTML(ani2.to_jshtml()
