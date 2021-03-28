使用body_25人体姿态数据格式

数据存储格式：1.调试格式 List:"time" : [ dict:"body_part" : NameTuple(x,y,)]
            2.实际格式 np.array[json_list]

时间帧间隔： 40-50毫秒 一秒20帧