import cv2
import os
import time

filelist = ['pic_ty.jpg','pic_gn.jpg','ty_graph.txt','go.txt','ty_go.txt','cap_gn_go.txt','cap_ty_go.txt','ty_finished.txt','gn_finished.txt','display_go.txt']
for each in filelist:
    filename = "/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/" + each
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

with open('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/cap_go.txt', 'w') as f:
    f.write('go')
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
start_time = time.time()

cv2.imwrite('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/pic_ty.jpg',frame)
cv2.imwrite('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/pic_gn.jpg',frame)
with open('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/cap_ty_go.txt', 'w') as f:
    f.write('go')
with open('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/cap_gn_go.txt', 'w') as f:
    f.write('go')
print('Ready to go')
times = 0
while True:
    ret, frame = cap.read()
    if os.path.exists('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/ty_finished.txt') == 1:
        time.sleep(0.005)
        cv2.imwrite('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/pic_ty.jpg',frame)
        os.remove('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/ty_finished.txt')
        while True:
            try:
                with open('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/ty_go.txt', 'w') as f:
                    f.write('go')
                break
            except FileNotFoundError:
                pass
        while True:
            try:
                with open('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/cap_ty_go.txt', 'w') as f:
                    f.write('go')
                break
            except FileNotFoundError:
                pass
    if os.path.exists('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/gn_finished.txt') == 1:
        time.sleep(0.005)
        cv2.imwrite('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/pic_gn.jpg',frame)
        os.remove('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/gn_finished.txt')
        while True:
            try:
                with open('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/cap_gn_go.txt', 'w') as f:
                    f.write('go')
                break
            except FileNotFoundError:
                pass
    #cv2.imshow('capture',frame)
    times+=1
    end_time = time.time()
    print(times)
    print((end_time-start_time)/times)
    cv2.waitKey(1)
cap.release()
cap.destoryAllWindows()
