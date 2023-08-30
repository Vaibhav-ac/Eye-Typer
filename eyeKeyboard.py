import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import tkinter as tk
from tkinter import *

#print(cv2.__version__) #4.7.0
#print(mp.__version__) #0.10.1
#print(np.__version__) #1.24.1
#print(pyautogui.__version__) #0.9.54
#print(tkinter.TkVersion) #8.6

#getting the screensize
screensize = pyautogui.size()

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

#setting up the alphabets on the screen
up_width, up_height = screensize
alphx = int(up_width * 0.5)
alphy = int(up_height * 0.3)
startx = int((up_width - alphx) / 2)
starty = int((up_height - alphy) / 2)

#dividing the alphabets into 4 rows
#each row has 28/4 = 7 letters per row
hi = int(alphy / 4)
wid = int(alphx / 7)
blank = np.ones((alphy, alphx, 3), dtype = np.uint8)
blank[:,:] = (255, 255, 255)

for i in range(4):
    vary = (hi * i) + (hi // 2)
    for j in range(7):
        varx = (j * wid) + (wid // 2)
        var = chr(ord('A') + (7 * i) + j) #type-casting to print letters
        t = ""
        t += var
        if (7 * i) + j == 26:
            t = "_"
        if (7 * i) + j == 27:
            t = "<"
        cv2.putText(blank, t, (varx, vary), cv2.FONT_HERSHEY_DUPLEX, float(up_width)/float(up_height), (0, 0, 0), 3)

#displaying the alphabets on the screen
cv2.imshow("BLANK", blank)
cv2.moveWindow("BLANK", startx, starty)

root = Tk()
# specify size of window.
root.geometry("1200x150+400+100")
 
# Create text widget and specify size.
T = Text(root, height = 100, width = 1100)
l = Label(root, text = "Letters")
l.config(font =("Courier", 20))

#use relative coordinate system based on the first value
ans = ' '
prevchar = '1'
cnt = 0
cap = cv2.VideoCapture(0)
v = 0
#can make use of z axis coordinates if there are multiple faces
while True:
    l.pack()
    T.pack()
    v+=1
    success, img = cap.read()
    img = cv2.flip(img , 1)
    img = cv2.resize(img, (up_width, up_height))
    if not success:
        break
    a = 0
    b = 0
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)
    if results.multi_face_landmarks:
        for id, facelms in enumerate(results.multi_face_landmarks):
            h, w, c = img.shape
            #for reference
            if v == 1 : 
                a = facelms.landmark[10].z - facelms.landmark[152].z
                b = facelms.landmark[123].z - facelms.landmark[352].z
                break
            
            #difference between the topmost point and bottom most point of the face with respect to camera tells whether face moving up or down
            newa = facelms.landmark[10].z - facelms.landmark[152].z

            #difference between the rightmost and leftmost point of the face with respect to camera tells whether face moving left or right
            newb = facelms.landmark[123].z - facelms.landmark[352].z
            
            #accordingly the difference is scaled 
            scale = float(up_width)/float(up_height)
            #scale = 0
            diffy = ((a - newa) * w) * scale
            diffx = ((b - newb) * w) * scale

            #relative scaling for better movement
            #adding the difference from the reference on the current coordinates
            midptx = facelms.landmark[9].x * w + diffx
            midpty = facelms.landmark[9].y * h + diffy

            pyautogui.moveTo(midptx, midpty) #moving cursor

            #identifying the letter on which the cursor is present
            refx = int(midptx - startx)
            refy = int(midpty- starty)
            for i in range(5) :
                if hi * i > refy :
                    if i == 0 :
                        cnt = 0
                    else :
                        for j in range(8) :
                            if wid * j > refx :
                                if j == 0 :
                                    cnt = 0
                                else :
                                    ans = chr(ord('A') + (7 * (i - 1)) + (j - 1)) #type-casting
                                    if (7 * (i - 1)) + (j - 1) == 26 :
                                        ans = '_'
                                    if (7 * (i - 1)) + (j - 1) == 27 :
                                        ans = '<'
                                break
                            elif j == 7 :
                                cnt = 0
                    break
                elif i == 4 : 
                    cnt = 0
            if cnt == 0 :
                prevchar = ans
                cnt+=1
            elif prevchar != ans :
                cnt = 0
            else :
                cnt+=1
            if cnt == 8 : #if the cursor is on the same letter for 8 frames then that letter is printed
                if ans == '<' :
                    T.delete("end-2c")
                else :
                    T.insert(tk.END, ans)
                cnt = 0
    T.update()

    img = cv2.resize(img, (200, 150))
    cv2.imshow("FACE", img)
    cv2.moveWindow("FACE", 10, 10)

    if cv2.waitKey(30) & 0xFF == ord('q'): #30 millisecond wait time and pressing q stops the code
        break

cap.release()
cv2.destroyAllWindows()
tk.mainloop()

 