from util import *


class Shixinqiu:
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

    def judge_left_right(self, pose_list_all, left="LKnee", right="RKnee"):
        '''返回0为左在前'''
        mean = get_mean_pose_position(pose_list_all)
        if mean[dic_from_names_to_index[left]] < mean[dic_from_names_to_index[right]]:
            return 0
        else:
            return 1

    def Xiaobiguowan(self, json_list, angle_threshold=95):
        Quxi_index = [dic_from_names_to_index["LWrist"], dic_from_names_to_index["LElbow"],
                      dic_from_names_to_index["LShoulder"]]
        angle = []
        for json_timestep in json_list:
            angle.append(from_3Index_get_Angle(Quxi_index, json_timestep))
        angle = [i + 360 if i < 0 else i for i in angle]
        angle = Outliers_clean(angle)
        MinAngle = min(angle)
        if MinAngle < angle_threshold:
            return 1, angle.index(MinAngle), 0
        else:
            return 0, None, 1

    def Yaobuhouyang(self, json_list, angle_threshold=90):
        houyang_abnormality = abnormality(angle_threshold, 10 * len(str(abs(angle_threshold))))
        L_R = self.judge_left_right(json_list)
        houyang_index = [dic_from_names_to_index["Neck"], dic_from_names_to_index["MidHip"],
                         dic_from_names_to_index["RKnee" if L_R else "LKnee"]]
        angle = []
        for json_timestep in json_list:
            angle.append(from_3Index_get_Angle(houyang_index, json_timestep))
        angle = np.array(angle)
        angle[angle >= 0] = angle[angle >= 0] - 360
        angle = moving_average(Outliers_clean(angle, threshold=30), 4)
        MinAngle = min(angle)
        if MinAngle > angle_threshold:
            return 1, angle.tolist().index(MinAngle), houyang_abnormality(MinAngle)
        else:
            return 0, None, 1

    def Qianhoujio(self, json_list, threshold=5):
        jio_abnormality = abnormality(threshold, len(str(abs(threshold))) * 10)
        L_R = self.judge_left_right(json_list)
        if L_R == 1:
            front_pose = "RHeel"
            back_pose = "LHeel"
        else:
            front_pose = "LHeel"
            back_pose = "RHeel"
        front_pose = get_pack_from_aniPose(json_list)[dic_from_names_to_index[front_pose] * 3]
        back_pose = get_pack_from_aniPose(json_list)[dic_from_names_to_index[back_pose] * 3]
        if min(back_pose) - max(front_pose) <= threshold:
            return 1, np.argmin(np.array(back_pose)), jio_abnormality(min(back_pose) - max(front_pose))
        else:
            return 0, None, 1

    # def split(self, pose_list_all, height_threshold=10):
    #     init_height = (pose_list_all[0][19 * 3 + 1] + pose_list_all[0][22 * 3 + 1]) / 2  # 左右脚大拇指的平均高度
    #     div1 = None  # 起跳帧号
    #     div2 = None  # 落地帧号
    #     for i in range(len(pose_list_all)):
    #         toe_height = (pose_list_all[i][19 * 3 + 1] + pose_list_all[i][22 * 3 + 1]) / 2
    #         if init_height - toe_height >= height_threshold:
    #             div1 = i
    #             break
    #     if div1 is None:
    #         return None, None
    #     last_height = (pose_list_all[div1][19 * 3 + 1] + pose_list_all[div1][22 * 3 + 1]) / 2
    #     for i in range(div1, len(pose_list_all)):
    #         toe_height = (pose_list_all[i][19 * 3 + 1] + pose_list_all[i][22 * 3 + 1]) / 2
    #         if last_height - toe_height >= 0:
    #             div2 = i
    #             break
    #         else:
    #             last_height = toe_height
    #     return div1, div2

    # def split_fall(self, pose_list_all, jump_index):

    def judge(self):
        '''
        跳远的只输入一个运动周期，分割起跳，跳中，落地三个阶段
        '''

        METHOD = [self.Xiaobiguowan, self.Yaobuhouyang, self.Qianhoujio]
        total_error = ['小臂后举时弯曲幅度过大', '背弓不足', '后脚移动越线']

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
    action = Shixinqiu(
        r'C:\Users\lmsZs\Desktop\Desktop\Projects\python\BodyPose\json_data\shixinqiu\shixinqiu-correct\openpose\output.json#')
    res = action.judge()
    print(res[0])
    print((res[1]))
    print((res[2]))
# ani=from_json_plot()
# ani2 = from_2json_plot()
# HTML(ani2.to_jshtml()
