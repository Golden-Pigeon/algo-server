import cv2


def getVideoJpg(_videoPath, index):
    # vidcap = cv2.VideoCapture("C:\\Users\\sswc\\Desktop\\gai2\\public\\showPdf\\6f3db1f9-e247-4aa5-bca7-93bae62a0079.mp4")
    vidcap = cv2.VideoCapture(_videoPath)
    vidcap.set(1, index)
    success, image = vidcap.read()

    # name = "C:\\Users\\sswc\\Desktop\\gai2\\public\\pdfImages\\6f3db1f9-e247-4aa5-bca7-93bae62a0079.png"
    # name = _pngPath
    return image
    # cv2.imwrite(_pngPath, image)

if __name__ == '__main__':
    video = r"C:\Users\lmsZs\Desktop\Desktop\Projects\python\BodyPose\视频\实心球\vlc-record-2021-03-14-20h01m51s-实心球（原）66529032-1-192--openpose.mp4"
    n = getVideoJpg(video, 25)
    cv2.imshow('1', n)
    cv2.waitKey()