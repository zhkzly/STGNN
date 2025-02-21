

import tkinter as tk
import numpy as np

import math

'''
基本功能:
1:能够产生一个方形的数字格子
2:可以进行基本的刷新操作(也就是更新格子中的数字，并且可以选自格子的大小)
3:

一个初始化为5*5的数字格子


'''


class Game_window():
    def __init__(self, init_windows):
        self.init_windows = init_windows
        array_number = np.arange(5*5) + 1
        index_=np.arange(5*2)
        game_window.rowconfigure(index_.tolist(), weight=1)
        game_window.columnconfigure(index_.tolist(), weight=1)
        self.output = np.random.choice(a=array_number, size=(5,5), replace=False)
        # self.set_init_window()



    def set_init_window(self):
        self.right_position=tk.Label(master=self.init_windows,text='h')
        self.right_position.grid(rows=5*2-1,column=5*2-1)
        self.data_entry = tk.Entry(master=self.init_windows,bg='orange')
        self.data_entry.grid(row=5+2,column=2,sticky='nsew',columnspan=5)
        self.init_flash_button = tk.Button(master=self.init_windows, text='flash button', bg='lightblue',command=self.flash_button_command)
        self.init_flash_button.grid(row=5+4,column=2,sticky='nsew',columnspan=5)
        self.init_input_button=tk.Button(master=self.init_windows,text='确定',command=self.input_button_command)
        self.init_input_button.grid(row=5+3,column=2,sticky='nsew',columnspan=5)
        self.label_list=[[tk.Label(master=self.init_windows,text=f'{self.output[i,j]}') for j in range(5)] for i in range(5)]
        for i in range(5):
            for j in range(5):
                self.label_list[i][j].grid(row=i+2,column=j+2,sticky='nsew')


        # self.label_list[0][0]['text']=
    @property
    def array_size(self):
        return int(float(self.data_entry.get()))

    def clear_wigets(self,wigets:list):
        for row in wigets:
            for col in row:
                col['text']=' '


    def flash_button_command(self):
        self.input_button_command()
        self.right_position.destroy()
        self.right_position=tk.Label(master=self.init_windows,text='h')
        self.right_position.grid(rows=self.array_size*2-1,column=self.array_size*2-1)

        self.clear_wigets(self.label_list)
        print(self.label_list)
        self.init_windows.update()
        for i in range(self.array_size):
            self.label_list = []
            for j in range(self.array_size):
                # print(type(self.output))
                # print(self.label_list)
                # print(type(self.label_list[i][j]['text']))
                cols=[]
                _label=tk.Label(master=self.init_windows,text=self.output[i,j])
                _label.grid(row=i+math.ceil(self.array_size/2),column=j+math.ceil(self.array_size/2),sticky='nsew')
                cols.append(_label)
            self.label_list.append(cols)

        self.init_input_button.grid(row=self.array_size+math.ceil(self.array_size/2)+2,column=math.ceil(self.array_size/2),sticky='nsew',columnspan=self.array_size)
        self.init_windows.update()

        self.init_flash_button.grid(row=self.array_size+math.ceil(self.array_size/2)+3,column=math.ceil(self.array_size/2),sticky='nsew',columnspan=self.array_size)

        self.init_windows.update()
        self.data_entry.grid(row=self.array_size+math.ceil(self.array_size/2)+1,column=math.ceil(self.array_size/2),sticky='nsew',columnspan=self.array_size)
        self.init_windows.update()

    def input_button_command(self):
        array_number = np.arange(self.array_size*self.array_size) + 1

        index_=np.arange(self.array_size*2)
        game_window.rowconfigure(index_.tolist(), weight=1)
        game_window.columnconfigure(index_.tolist(), weight=1)
        self.init_windows.update()
        self.output = np.random.choice(a=array_number, size=(self.array_size ,self.array_size ), replace=False)



game_window=tk.Tk()

my_window=Game_window(game_window)
game_window.update()
game_window.mainloop()





# class T():
#     def __init__(self):
#         pass
#     def get_(self):
#         print(type(self.a))
#
#
# a=T()
# a.get_()




