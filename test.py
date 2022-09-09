import time
import threading
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.figure import Figure

import seaborn as sns #importするだけでスタイルがSeabornになる

from pygame import mixer

mixer.init()        #初期化
mixer.music.load("C:/Users/0414i/Downloads/hito_ge_paku01.mp3")



f = [0]
sigre = [0]
sigim = [0]
sigabs = [0]

sns.set_style("white")
class GUI(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.value = 0

    def start(self):
        self.root = tk.Tk()
        self.root.configure(bg='white')
        self.root.geometry("700x650")
        # StringVarをフィールドに定義する
        self.sv = tk.StringVar()
        self.sv.set("iinokei")
        # ラベルの表示 データはStringVarをバインドする
        # self.label1 = tk.Label(self.root, textvariable=self.sv)
        # self.label1.place(x=150, y=228)
        # # self.label1.pack()
        # self.label = tk.Label(self.root, textvariable=self.sv)
        # self.label.place(x=150, y=2)
        # self.label.pack()
        # ボタンの表示
        self.button = tk.Button(self.root, text='push', command=self.change_value_callback, height=20,width=30)
        self.button.pack(anchor=tk.NW, padx=70, pady=20)
        
        # Canvasを生成
        self.F = Figure(figsize=(10, 3), dpi=100)
        self.re = self.F.add_subplot(121)
        self.im = self.F.add_subplot(122)
        # self.abso = self.F.add_subplot(133)
        self.canvas = FigureCanvasTkAgg(self.F, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, expand=0)
        self.canvas._tkcanvas.pack(side=tk.BOTTOM, expand=0)
        
        # キャンバス作成
        self.canvas2 = tk.Canvas(self.root, bg="white", height=320, width=350)
        # キャンバス表示
        self.canvas2.place(x=300, y=20)
        
        # イメージ作成
        self.imgsss = tk.PhotoImage(file="C:/Users/0414i/Downloads/画像1.png", width=320, height=320, master=self.root)
        # キャンバスにイメージを表示
        self.canvas2.create_image(30, 30, image=self.imgsss, anchor=tk.NW)
        
        self.root.mainloop()
    
    # change_valueを別スレッドで実行するコールバック
    def change_value_callback(self):
        th = threading.Thread(target=self.change_value, args=())
        th.start()

    # StringVarを更新するように変更する
    def change_value(self):

        for value in range(100):
            time.sleep(0.5)
            # StringVarを変更するとGUIスレッドでラベル文字列が更新される
            self.sv.set(str(value))
            # ラベルに表示されるだろう値を表示
            print(value)
            
            if value % 3 == 0:
                mixer.music.play(1)
                # イメージ作成
                self.imgsss = tk.PhotoImage(file="C:/Users/0414i/Downloads/画像2.png", width=320, height=320, master=self.root)
                # キャンバスにイメージを表示
                self.canvas2.create_image(30, 30, image=self.imgsss, anchor=tk.NW)
                appender = f[-1]+1
                appender2 = f[-1]
            elif value%7 == 0:
                appender2 = f[-1]+2
                appender = f[-1]
                # イメージ作成
                self.imgsss = tk.PhotoImage(file="C:/Users/0414i/Downloads/画像3.png", width=320, height=320, master=self.root)
                # キャンバスにイメージを表示
                self.canvas2.create_image(30, 30, image=self.imgsss, anchor=tk.NW)
                
            f.append(value)
            sigre.append(appender)
            sigim.append(appender2)
            # sigabs.append(appender)

            # データをClear
            self.re.cla()
            self.im.cla()
            # self.abso.cla()

            self.re.plot(f, sigre, "r")
            self.im.plot(f, sigim, "b")
            # self.abso.plot(f, sigabs, "r")
            self.canvas.draw()

if __name__ == '__main__':
    gui = GUI()
    gui.start()