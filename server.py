from flask import Flask, request, make_response
import os
from openpose import get_json_file
from tiaoyuan import Tiaoyuan
from fuwocheng import Fuwocheng
from shixinqiu import Shixinqiu
from yangwoqizuo import Yangwoqizuo
from video_get_pic import getVideoJpg
import cv2
import random

app = Flask(__name__)


@app.route('/for_judge', methods=['POST'])
def for_judge():
    global judger
    if request.method == 'POST':
        params_file = request.files['video']
        action = request.form['action']
        video_path = "./videos/" + params_file.name + "-openpose.json#"
        dst = os.path.join(video_path)
        params_file.save(dst)
        if not os.path.exists(video_path):
            return {
                "code": 30000,
                "message": '视频上传失败'
            }
        json_file_path = get_json_file(video_path)

        if not os.path.exists(json_file_path):
            return {
                "code": 30001,
                "message": 'json文件生成失败'
            }
        if action == 'tiaoyuan':
            judger = Tiaoyuan(json_file_path)
        elif action == 'fuwocheng':
            judger = Fuwocheng(json_file_path)
        elif action == 'shixinqiu':
            judger = Shixinqiu(json_file_path)
        elif action == 'yangwoqizuo':
            judger = Yangwoqizuo(json_file_path)
        res = judger.judge()
        idx = res[3]
        if len(idx) > 0:
        	frame = getVideoJpg(video_path, idx[0])
        	frame_path = "./imgs/" + params_file.name + str(random.randint(1, 100000000)) + ".png"
        	cv2.imwrite(frame_path, frame)
        	correct_path = res[4][0]

	        return {
	            "code": 200,
	            "message": '成功返回',
	            "error_res": res[0],
	            "advice_res": res[1],
	            "fitness": "%.2f%%" % (float(res[2]) * 100.0,),
	            "error_img": "http://38r3144h99.zicp.vip:58980/show_img?path=" + frame_path
	            "correct_path": "http://38r3144h99.zicp.vip:58980/show_img?path=" + correct_path
	        }
	    else:
	    	return {
	            "code": 200,
	            "message": '成功返回',
	            "error_res": res[0],
	            "advice_res": res[1],
	            "fitness": "%.2f%%" % (float(res[2]) * 100.0,),
	            # "error_img": "http://38r3144h99.zicp.vip:58980/show_img?path=" + frame_path
	            # "correct_path": "http://38r3144h99.zicp.vip:58980/show_img?path=" + correct_path
	        }
    else:
        return {
            "code": 500,
            "message": '不支持的HTTP方法'
        }


@app.route("/show_img", methods=['GET'])
def show_img():
    cdir = request.args['path']

    image_data = open(cdir, 'rb').read()
    res = make_response(image_data)
    res.headers['Content-Type'] = 'image/png'
    return res




if __name__ == '__main__':
    app.run(port=8082)