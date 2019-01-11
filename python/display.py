import cv2
import os
import time

times = 0
start_time = time.time()
#cap = cv2.VideoCapture('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/display_image.jpg')
while True:
    while True:
        if os.path.exists('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/gn_finished.txt') == 1:
            break
    img = cv2.imread('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/display_image.jpg')
    end_time = time.time()
    times+=1
    average = (end_time - start_time) / times
    print('times:' + str(times))
    print('average_time:' + str(average))
    #os.remove('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/display_go.txt')
    cv2.imshow('capture',img)
    cv2.waitKey(1)
cap.release()
cap.destoryAllWindows()
