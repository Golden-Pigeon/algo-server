from judge import Fuwocheng
from util import *
from judge_old import Judge

# def test():
#     '''测试函数，对json list作shift-split-judge流程，通过'''
#     json_file1='json_data\Quick1-openpose.mp4#-20210131T074007Z-001\Quick1-openpose.mp4#'
#     json_file2='json_data\Std1-openpose.mp4#-20210131T074012Z-001\Std1-openpose.mp4#'
#     _,_,filenames1=next(os.walk(json_file1))
#     filenames1 = [os.path.join(json_file1,i) for i in filenames1 if i[-4:]=='json']
#     pose_list_all1=[]
#     for filename in filenames1:
#         pose_list = from_path_get_poseList(filename)
#         if pose_list is not None:
#             pose_list_all1.append(pose_list)
#     # pose_list_all1 = pose_list_all1[50:cut_down+50]
#
#         _,_,filenames2=next(os.walk(json_file2))
#     filenames2 = [os.path.join(json_file2,i) for i in filenames2 if i[-4:]=='json']
#     pose_list_all2=[]
#     for filename in filenames2:
#         pose_list = from_path_get_poseList(filename)
#         if pose_list is not None:
#             pose_list_all2.append(pose_list)
#     # pose_list_all2 = pose_list_all2[50:cut_down+50]
#
#     pose_list_all1 = np.array(pose_list_all1)
#     pose_list_all2 = np.array(pose_list_all2)
#
#     pose_list_all1 = shift(pose_list_all1,pose_list_all2)
#     slices = split(pose_list_all1)
#     i=0
#     #在这里同样手动判断提取间隔适当的index
#     slice1_fir = slices[0]
#     slice1_end = slices[1]
#     while slice1_end-slice1_fir<=7:
#         i+=1
#         slice1_fir = slices[i]
#         slice1_end = slices[i+1]
#
#     slices = split(pose_list_all2)
#     i=0
#     slice2_fir = slices[0]
#     slice2_end = slices[1]
#     while slice2_end-slice2_fir<=7:
#         i+=1
#         slice2_fir = slices[i]
#
#         slice2_end = slices[i+1]
#     judge(pose_list_all1[slice1_fir:slice1_end],pose_list_all2[slice2_fir:slice2_end])


def Test(json_file1='./json_data/Quick1-openpose.mp4#-20210131T074007Z-001/Quick1-openpose.mp4#',
    json_file2='./json_data/Std1-openpose.mp4#-20210131T074012Z-001/Std1-openpose.mp4#'):
    '''第二版的代码测试流程'''


    _,_,filenames1=next(os.walk(json_file1))
    filenames1 = [os.path.join(json_file1,i) for i in filenames1 if i[-4:]=='json']
    pose_list_all1=[]
    for filename in filenames1:
        pose_list = from_path_get_poseList(filename)
        if pose_list is not None:
            pose_list_all1.append(pose_list)
    # pose_list_all1 = pose_list_all1[50:cut_down+50]

        _,_,filenames2=next(os.walk(json_file2))
    filenames2 = [os.path.join(json_file2,i) for i in filenames2 if i[-4:]=='json']
    pose_list_all2=[]
    for filename in filenames2:
        pose_list = from_path_get_poseList(filename)
        if pose_list is not None:
            pose_list_all2.append(pose_list)
    # pose_list_all2 = pose_list_all2[50:cut_down+50]

    pose_list_all1 = np.array(pose_list_all1)
    pose_list_all2 = np.array(pose_list_all2)

    pose_list_all1 = shift(pose_list_all1,pose_list_all2)
    slices1 = split(pose_list_all1)
    slices2 = split(pose_list_all2)
    ERROR_COUNT=0
    ALL_TIMES=0
    for i in range(min(len(slices1),len(slices2))-1):
        #在这里同样手动判断提取间隔适当的index
        slice1_fir = slices1[i]
        slice1_end = slices1[i+1]
        slice2_fir = slices2[i]
        slice2_end = slices2[i+1]
        ERROR_COUNT+=Judge(pose_list_all1[slice1_fir:slice1_end],pose_list_all2[slice2_fir:slice2_end])
        ALL_TIMES+=1
    last_judge_confidence =ERROR_COUNT/ALL_TIMES
    return last_judge_confidence

if __name__ == '__main__':
    # Test(json_file1 = './json_data/hip/openpose/output.json')
    # action = Fuwocheng(r'C:\Users\lmsZs\Desktop\Desktop\Projects\python\BodyPose\json_data\Quick1-openpose.mp4'
    #                    r'#-20210131T074007Z-001\Quick1-openpose.mp4#')
    # action = Fuwocheng(r'C:\Users\lmsZs\Desktop\Desktop\Projects\python\BodyPose\json_data\head_back\openpose\output'
    #                    r'.json')
    action = Fuwocheng(r'C:\Users\lmsZs\Desktop\Desktop\Projects\python\BodyPose\json_data\head_back\openpose\output.json')
    res = action.judge()
    print(res[0])
    print(res[1])
    # ani=from_json_plot()
    # ani2 = from_2json_plot()
    # HTML(ani2.to_jshtml())
