# 环境要求: python 3.6 + opencv-python 3.4.14.51
# 功能: 简易人脸识别系统(采集-训练-识别-界面)
import cv2
import numpy as np
import os
import shutil
import threading
import tkinter as tk
from PIL import Image, ImageTk

# ------------------ 基础数据结构 ------------------
id_dict = {}  # id->name
Total_face_num = 0

def init():
    if not os.path.exists('config.txt'):
        with open('config.txt', 'w') as f:
            f.write('0\n')
    with open('config.txt') as f:
        global Total_face_num
        Total_face_num = int(f.readline())
        for _ in range(Total_face_num):
            line = f.readline()
            if not line.strip():
                continue
            id_name = line.split()
            if len(id_name) >= 2:
                id_dict[int(id_name[0])] = id_name[1]

init()

# ------------------ OpenCV初始化 ------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
camera = cv2.VideoCapture(0)
success, img = camera.read()
W_size = 0.1 * camera.get(3)
H_size = 0.1 * camera.get(4)

system_state_lock = 0  # 0空闲 1刷脸中 2录入中

# ------------------ 人脸采集 ------------------
def Get_new_face():
    print("正在从摄像头录入新人脸信息")
    filepath = "data"
    if os.path.exists(filepath): shutil.rmtree(filepath)
    os.mkdir(filepath)
    sample_num = 0
    pictur_num = 30

    while True:
        global success, img
        success, img = camera.read()
        if not success: break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 0))
            sample_num += 1
            T = Total_face_num
            cv2.imwrite("./data/User.{}.{}.jpg".format(T, sample_num), gray[y:y+h, x:x+w])
        if sample_num > pictur_num: break
        l = int(sample_num / pictur_num * 50)
        r = 50 - l
        print("\r%3d%% %s->%s" % (sample_num / pictur_num * 100, "="*l, "_"*r), end="")
        var.set("%3d%%" % (sample_num / pictur_num * 100))
        window.update()

# ------------------ 人脸训练 ------------------
def Train_new_face():
    print("\n正在训练")
    path = 'data'
    recog = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels(path)
    recog.train(faces, np.array(ids))
    yml = str(Total_face_num) + ".yml"
    recog.save(yml)

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples, ids = [], []
    for image_path in image_paths:
        if not image_path.endswith('.jpg'): continue
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, 'uint8')
        id = int(os.path.split(image_path)[-1].split('.')[1])
        faces = face_cascade.detectMultiScale(img_np)
        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(id)
    return face_samples, ids

def write_config():
    print("新人脸训练结束")
    T = Total_face_num
    with open('config.txt', "a") as f:
        f.write(str(T) + " User" + str(T) + "\n")
    id_dict[T] = "User" + str(T)
    with open('config.txt', 'r+') as f:
        flist = f.readlines()
        flist[0] = str(int(flist[0]) + 1) + "\n"
    with open('config.txt', 'w') as f:
        f.writelines(flist)

# ------------------ 人脸识别 ------------------
def scan_face():
    for i in range(Total_face_num):
        recognizer.read(str(i+1) + ".yml")
        ave_poss = 0
        for times in range(10):
            global success, img
            while system_state_lock == 2: pass
            success, img = camera.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(W_size), int(H_size))
            )
            cur_poss = 0
            for (x, y, w, h) in faces:
                while system_state_lock == 2: pass
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                conf = confidence
                if confidence < 100:
                    user_name = id_dict.get(idnum, "Untagged user:{}".format(idnum))
                    confidence = "{:.0f}%".format(100 - confidence)
                else:
                    user_name = "unknown"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(user_name), (x+5, y-5), font, 1, (0,0,255), 1)
                cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (0,0,0), 1)
                if 0 < conf < 15 or 35 < conf < 60:
                    cur_poss = 1
                else:
                    cur_poss = 0
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
            ave_poss += cur_poss
        if ave_poss >= 5:
            return i + 1
    return 0

# ------------------ 多线程交互 ------------------
def f_scan_face_thread():
    var.set('刷脸')
    ans = scan_face()
    if ans == 0:
        var.set("最终结果：无法识别")
    else:
        var.set("最终结果：{} {}".format(ans, id_dict[ans]))
    global system_state_lock
    system_state_lock = 0

def f_scan_face():
    global system_state_lock
    if system_state_lock in (1, 2): return
    system_state_lock = 1
    threading.Thread(target=f_scan_face_thread, daemon=True).start()

def f_rec_face_thread():
    var.set('录入')
    cv2.destroyAllWindows()
    global Total_face_num
    Total_face_num += 1
    Get_new_face()
    global system_state_lock
    system_state_lock = 0
    Train_new_face()
    write_config()

def f_rec_face():
    global system_state_lock
    if system_state_lock == 2: return
    system_state_lock = 2
    threading.Thread(target=f_rec_face_thread, daemon=True).start()

def f_exit():
    camera.release()
    cv2.destroyAllWindows()
    window.quit()
    window.destroy()  # 完全退出

# ------------------ 图形界面 ------------------
window = tk.Tk()
window.title('Cheney\' Face_rec 3.0')
window.geometry('1000x500')
var = tk.StringVar()
l = tk.Label(window, textvariable=var, bg='green', fg='white', font=('Arial', 12), width=50, height=4)
l.pack()
tk.Button(window, text='开始刷脸', font=('Arial', 12), width=10, height=2, command=f_scan_face).place(x=800, y=120)
tk.Button(window, text='录入人脸', font=('Arial', 12), width=10, height=2, command=f_rec_face).place(x=800, y=220)
tk.Button(window, text='退出', font=('Arial', 12), width=10, height=2, command=f_exit).place(x=800, y=320)
panel = tk.Label(window, width=500, height=350)
panel.place(x=10, y=100)
window.config(cursor="arrow")

def video_loop():
    global success, img
    if success:
        cv2.waitKey(1)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
    window.after(10, video_loop)

video_loop()
window.mainloop()