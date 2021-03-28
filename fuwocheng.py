from util import *


class Fuwocheng:
    def __init__(self, now_json_path, std_json_path=r'C:\Users\lmsZs\Desktop\Desktop\Projects\python\BodyPose\json_data\fuwocheng_yangwoqizuo\Std1-openpose.mp4#-20210131T074012Z-001\Std1-openpose.mp4#'):
        self.std_json = std_json_path
        self.user_json_path = now_json_path
        assert os.path.exists(self.std_json)

    def data_process(self):
        _, _, filenames1 = next(os.walk(self.user_json_path))
        filenames1 = [os.path.join(self.user_json_path, i) for i in filenames1 if i[-4:] == 'json']
        pose_list_all1 = []
        for filename in filenames1:
            pose_list = from_path_get_poseList(filename)
            if pose_list is not None:
                pose_list_all1.append(pose_list)
            # pose_list_all1 = pose_list_all1[50:cut_down+50]

            _, _, filenames2 = next(os.walk(self.std_json))
        filenames2 = [os.path.join(self.std_json, i) for i in filenames2 if i[-4:] == 'json']
        pose_list_all2 = []
        for filename in filenames2:
            pose_list = from_path_get_poseList(filename)
            if pose_list is not None:
                pose_list_all2.append(pose_list)
        # pose_list_all2 = pose_list_all2[50:cut_down+50]

        pose_list_all1 = np.array(pose_list_all1)
        pose_list_all2 = np.array(pose_list_all2)

        return pose_list_all2, pose_list_all1

    def split(self, json_list):
        """先以10窗口平滑，再7窗口二次平滑,效果大概百分之七八十，还带写更多if-else手动判断
        由于openpose结果震荡，俯卧撑判断正确度大概80%"""
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

    def shift(self, json_list, json_stander, POSITION='RWrist'):
        """相对标准视频的json list，做归一化，这里以MidHip为标准"""
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

    def Too_ast(self, shifted_slice_json, json_stander, threshold=0.7):
        fast_index = [1, 2, 5]  # 哪几个部位更快
        count = 0
        all_count = 0
        for timestep in np.linspace(0, 0.5, 20):
            json_timestep = shifted_slice_json[int((len(shifted_slice_json) - 1) * timestep)]
            stander_json_timestep = json_stander[int((len(json_stander) - 1) * timestep)]
            for index in fast_index:
                if from_partIndex_get_poseTuple(index, json_timestep)[1] < \
                        from_partIndex_get_poseTuple(index, stander_json_timestep)[1]:
                    count += 1
                all_count += 1
        if count / all_count >= threshold:
            return True
        else:
            return False

    def Hip_high(self, shifted_slice_json, json_stander, threshold=0.7):
        compare_index = [dic_from_names_to_index['MidHip'], dic_from_names_to_index["RWrist"],
                         dic_from_names_to_index["LWrist"]]
        count = 0
        all_count = 0
        for timestep in np.linspace(0, 0.5, 20):
            json_timestep = shifted_slice_json[int((len(shifted_slice_json) - 1) * timestep)]
            stander_json_timestep = json_stander[int((len(json_stander) - 1) * timestep)]

            hip = from_partIndex_get_poseTuple(compare_index[0], json_timestep)
            RWrist = from_partIndex_get_poseTuple(compare_index[1], json_timestep)
            LWrist = from_partIndex_get_poseTuple(compare_index[2], json_timestep)
            hip_S = from_partIndex_get_poseTuple(compare_index[0], stander_json_timestep)
            RWrist_S = from_partIndex_get_poseTuple(compare_index[1], stander_json_timestep)
            LWrist_S = from_partIndex_get_poseTuple(compare_index[2], stander_json_timestep)
            if ((hip[0] - (RWrist[0] + LWrist[0]) / 2) ** 2 + (hip[1] - (RWrist[1] + LWrist[1]) / 2) ** 2) > (
                    (hip_S[0] - (RWrist_S[0] + LWrist_S[0]) / 2) ** 2 + (
                    hip_S[1] - (RWrist_S[1] + LWrist_S[1]) / 2) ** 2):
                count += 1
            all_count += 1

        if count / all_count >= threshold:
            return True
        else:
            return False

    def Head_back(self, shifted_slice_json, json_stander, threshold=0.7):
        compare_index = [dic_from_names_to_index['Nose'], dic_from_names_to_index["Neck"],
                         dic_from_names_to_index["MidHip"]]
        count = 0
        all_count = 0
        for timestep in np.linspace(0, 0.5, 20):
            json_timestep = shifted_slice_json[int((len(shifted_slice_json) - 1) * timestep)]
            stander_json_timestep = json_stander[int((len(json_stander) - 1) * timestep)]
            Nose = from_partIndex_get_poseTuple(compare_index[0], json_timestep)
            Neck = from_partIndex_get_poseTuple(compare_index[1], json_timestep)
            MidHip = from_partIndex_get_poseTuple(compare_index[2], json_timestep)
            vec1 = np.array([Nose[0] - MidHip[0], Nose[1] - MidHip[1]])
            vec2 = np.array([Neck[0] - MidHip[0], Neck[1] - MidHip[1]])
            angle1 = Angle(vec2, vec1)
            Nose_S = from_partIndex_get_poseTuple(compare_index[0], stander_json_timestep)
            Neck_S = from_partIndex_get_poseTuple(compare_index[1], stander_json_timestep)
            MidHip_S = from_partIndex_get_poseTuple(compare_index[2], stander_json_timestep)
            vec1 = np.array([Nose_S[0] - MidHip_S[0], Nose_S[1] - MidHip_S[1]])
            vec2 = np.array([Neck_S[0] - MidHip_S[0], Neck_S[1] - MidHip_S[1]])
            angle2 = Angle(vec2, vec1)
            if angle1 > 0.1:  # angle2:
                count += 1
            all_count += 1
        if count / all_count >= threshold:
            return True
        else:
            return False

    def judge(self):
        '''
        默认已经split,对一个周期的动作判断,对于每种错误依次判断
        '''

        METHOD = [self.Too_ast, self.Hip_high, self.Head_back]
        total_error = ['动作过快', '臀部过高', '头部后仰']

        std_list, user_list = self.data_process()

        user_list = self.shift(user_list, std_list)
        user_slices = self.split(user_list)
        std_slices = self.split(std_list)

        # ERROR_COUNT = 0
        # ALL_TIMES = 0
        # for i in range(min(len(user_slices), len(std_slices)) - 1):
        #     # 在这里同样手动判断提取间隔适当的index
        #     slice1_fir = user_slices[i]
        #     slice1_end = user_slices[i + 1]
        #     slice2_fir = std_slices[i]
        #     slice2_end = std_slices[i + 1]
        #     ERROR_COUNT += Judge(user_list[slice1_fir:slice1_end], std_list[slice2_fir:slice2_end])
        #     ALL_TIMES += 1
        # last_judge_confidence = ERROR_COUNT / ALL_TIMES

        errors = []
        for i, method in enumerate(METHOD):
            ERROR_COUNT = 0
            ALL_TIMES = 0
            for j in range(min(len(user_slices), len(std_slices)) - 1):
                # 在这里同样手动判断提取间隔适当的index
                slice1_fir = user_slices[j]
                slice1_end = user_slices[j + 1]
                slice2_fir = std_slices[j]
                slice2_end = std_slices[j + 1]
                ERROR_COUNT += method(user_list[slice1_fir:slice1_end], std_list[slice2_fir:slice2_end])
                ALL_TIMES += 1
            last_judge_confidence = ERROR_COUNT / ALL_TIMES
            if last_judge_confidence >= 0.4:
                errors.append(total_error[i])

        error_res = ''
        advice_res = ''
        if len(errors) == 0:
            error_res = '未检测到问题。'
            advice_res = '您的动作很标准。'
        else:
            error_res = '您动作中的问题有：\n'
            for s in errors:
                error_res += s + '\n'
            if len(errors) / len(total_error) <= 0.4:
                advice_res = '您的动作稍有不标准，希望您能多加练习。'
            else:
                advice_res = '您的动作问题较多，请参照提供视频多加练习!'
        fitness = len(error_res) / len(METHOD)
        fitness = fitness if fitness <= 1 else 1
        return error_res, advice_res, 18, fitness


if __name__ == '__main__':
    # action = Fuwocheng('./json_data/Quick1-openpose.mp4#-20210131T074007Z-001/Quick1-openpose.mp4#')
    action = Fuwocheng(r'C:\Users\lmsZs\Desktop\Desktop\Projects\python\BodyPose\json_data\fuwocheng_yangwoqizuo\Quick1-openpose.mp4#-20210131T074007Z-001\Quick1-openpose.mp4#')
    res = action.judge()
# ani=from_json_plot()
# ani2 = from_2json_plot()
# HTML(ani2.to_jshtml()
