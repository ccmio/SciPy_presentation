﻿import tkinter as tk
import youhua_interface
import chazhi_interface
import jifen_interface
import xinhao_interface
import tuxiang_interface
import tongji_interface


def create1():
	top = tk.Toplevel()
	top.geometry('313x261')
	top.title('优化')
	top.configure(background='#d9d9d9')
	top.resizable(0, 0)

	Label1 = tk.Label(top)
	Label1.place(relx=0.19, rely=0.04, height=60, width=160)
	Label1.configure(background='#d9d9d9')
	Label1.configure(disabledforeground='#a3a3a3')
	Label1.configure(foreground='#000000')
	Label1.configure(text='请选择需要处理的问题')

	Labelframe1 = tk.LabelFrame(top)
	Labelframe1.place(relx=0.19, rely=0.19, relheight=0.63, relwidth=0.58)
	Labelframe1.configure(relief=tk.GROOVE)
	Labelframe1.configure(foreground='black')
	Labelframe1.configure(text='优化类型问题')
	Labelframe1.configure(background='#d9d9d9')
	Labelframe1.configure(width=180)

	Button1 = tk.Button(Labelframe1)
	Button1.place(relx=0.11, rely=0.2, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button1.configure(activebackground='#d9d9d9')
	Button1.configure(activeforeground='#000000')
	Button1.configure(background='#0bafd8')
	Button1.configure(disabledforeground='#a3a3a3')
	Button1.configure(foreground='#000000')
	Button1.configure(highlightbackground='#d9d9d9')
	Button1.configure(highlightcolor='black')
	Button1.configure(command=youhua_interface.c1)
	Button1.configure(text='最小二乘法拟合')

	Button2 = tk.Button(Labelframe1)
	Button2.place(relx=0.11, rely=0.47, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button2.configure(activebackground='#d9d9d9')
	Button2.configure(activeforeground='#000000')
	Button2.configure(background='#0bafd8')
	Button2.configure(disabledforeground='#a3a3a3')
	Button2.configure(foreground='#000000')
	Button2.configure(highlightbackground='#d9d9d9')
	Button2.configure(highlightcolor='black')
	Button2.configure(command=youhua_interface.c2)
	Button2.configure(text='求函数的最小值')

	Button3 = tk.Button(Labelframe1)
	Button3.place(relx=0.11, rely=0.74, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button3.configure(activebackground='#d9d9d9')
	Button3.configure(activeforeground='#000000')
	Button3.configure(background='#0bafd8')
	Button3.configure(disabledforeground='#a3a3a3')
	Button3.configure(foreground='#000000')
	Button3.configure(highlightbackground='#d9d9d9')
	Button3.configure(highlightcolor='black')
	Button3.configure(command=youhua_interface.c3)
	Button3.configure(text='非线性方程组求解')


def create2():
	top = tk.Toplevel()
	top.geometry('313x265')
	top.title('插值')
	top.configure(background='#d9d9d9')
	top.resizable(0, 0)

	Label1 = tk.Label(top)
	Label1.place(relx=0.19, rely=0.04, height=26, width=156)
	Label1.configure(background='#d9d9d9')
	Label1.configure(disabledforeground='#a3a3a3')
	Label1.configure(foreground='#000000')
	Label1.configure(text='请选择需要处理的问题')

	Labelframe1 = tk.LabelFrame(top)
	Labelframe1.place(relx=0.19, rely=0.19, relheight=0.63, relwidth=0.58)
	Labelframe1.configure(relief=tk.GROOVE)
	Labelframe1.configure(foreground='black')
	Labelframe1.configure(text='插值类型问题')
	Labelframe1.configure(background='#d9d9d9')
	Labelframe1.configure(width=180)

	Button1 = tk.Button(Labelframe1)
	Button1.place(relx=0.11, rely=0.2, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button1.configure(activebackground='#d9d9d9')
	Button1.configure(activeforeground='#000000')
	Button1.configure(background='#0bafd8')
	Button1.configure(disabledforeground='#a3a3a3')
	Button1.configure(foreground='#000000')
	Button1.configure(highlightbackground='#d9d9d9')
	Button1.configure(highlightcolor='black')
	Button1.configure(command=chazhi_interface.c1)
	Button1.configure(text='插值')

	Button2 = tk.Button(Labelframe1)
	Button2.place(relx=0.11, rely=0.47, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button2.configure(activebackground='#d9d9d9')
	Button2.configure(activeforeground='#000000')
	Button2.configure(background='#0bafd8')
	Button2.configure(disabledforeground='#a3a3a3')
	Button2.configure(foreground='#000000')
	Button2.configure(highlightbackground='#d9d9d9')
	Button2.configure(highlightcolor='black')
	Button2.configure(command=chazhi_interface.c2)
	Button2.configure(text='外推和拟合曲线')

	Button3 = tk.Button(Labelframe1)
	Button3.place(relx=0.11, rely=0.74, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button3.configure(activebackground='#d9d9d9')
	Button3.configure(activeforeground='#000000')
	Button3.configure(background='#0bafd8')
	Button3.configure(disabledforeground='#a3a3a3')
	Button3.configure(foreground='#000000')
	Button3.configure(highlightbackground='#d9d9d9')
	Button3.configure(highlightcolor='black')
	Button3.configure(command=chazhi_interface.c3)
	Button3.configure(text='二维插值')


def create3():
	top = tk.Toplevel()
	top.geometry('313x265')
	top.title('积分')
	top.configure(background='#d9d9d9')
	top.resizable(0, 0)
	Label1 = tk.Label(top)
	Label1.place(relx=0.19, rely=0.04, height=26, width=156)
	Label1.configure(background='#d9d9d9')
	Label1.configure(disabledforeground='#a3a3a3')
	Label1.configure(foreground='#000000')
	Label1.configure(text='请选择需要处理的问题')

	Labelframe1 = tk.LabelFrame(top)
	Labelframe1.place(relx=0.19, rely=0.19, relheight=0.63, relwidth=0.58)
	Labelframe1.configure(relief=tk.GROOVE)
	Labelframe1.configure(foreground='black')
	Labelframe1.configure(text='积分类型问题')
	Labelframe1.configure(background='#d9d9d9')
	Labelframe1.configure(width=180)

	Button1 = tk.Button(Labelframe1)
	Button1.place(relx=0.11, rely=0.2, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button1.configure(activebackground='#d9d9d9')
	Button1.configure(activeforeground='#000000')
	Button1.configure(background='#0bafd8')
	Button1.configure(disabledforeground='#a3a3a3')
	Button1.configure(foreground='#000000')
	Button1.configure(highlightbackground='#d9d9d9')
	Button1.configure(highlightcolor='black')
	Button1.configure(command=jifen_interface.c1)
	Button1.configure(text='求定积分')

	Button2 = tk.Button(Labelframe1)
	Button2.place(relx=0.11, rely=0.74, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button2.configure(activebackground='#d9d9d9')
	Button2.configure(activeforeground='#000000')
	Button2.configure(background='#0bafd8')
	Button2.configure(disabledforeground='#a3a3a3')
	Button2.configure(foreground='#000000')
	Button2.configure(highlightbackground='#d9d9d9')
	Button2.configure(highlightcolor='black')
	Button2.configure(command=jifen_interface.c2)
	Button2.configure(text='解常微分方程组')


def create4():
	top = tk.Toplevel()
	top.geometry('313x265')
	top.title('信号处理')
	top.configure(background='#d9d9d9')
	top.resizable(0, 0)
	Label1 = tk.Label(top)
	Label1.place(relx=0.19, rely=0.04, height=26, width=156)
	Label1.configure(background='#d9d9d9')
	Label1.configure(disabledforeground='#a3a3a3')
	Label1.configure(foreground='#000000')
	Label1.configure(text='请选择需要处理的问题')

	Labelframe1 = tk.LabelFrame(top)
	Labelframe1.place(relx=0.19, rely=0.19, relheight=0.63, relwidth=0.58)
	Labelframe1.configure(relief=tk.GROOVE)
	Labelframe1.configure(foreground='black')
	Labelframe1.configure(text='信号处理问题')
	Labelframe1.configure(background='#d9d9d9')
	Labelframe1.configure(width=180)

	Button1 = tk.Button(Labelframe1)
	Button1.place(relx=0.11, rely=0.47, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button1.configure(activebackground='#d9d9d9')
	Button1.configure(activeforeground='#000000')
	Button1.configure(background='#0bafd8')
	Button1.configure(disabledforeground='#a3a3a3')
	Button1.configure(foreground='#000000')
	Button1.configure(highlightbackground='#d9d9d9')
	Button1.configure(highlightcolor='black')
	Button1.configure(command=xinhao_interface.c1)
	Button1.configure(text='中值滤波')


def create5():
	top = tk.Toplevel()
	top.geometry('313x265')
	top.title('图像处理')
	top.configure(background='#d9d9d9')
	top.resizable(0, 0)
	Label1 = tk.Label(top)
	Label1.place(relx=0.19, rely=0.04, height=26, width=156)
	Label1.configure(background='#d9d9d9')
	Label1.configure(disabledforeground='#a3a3a3')
	Label1.configure(foreground='#000000')
	Label1.configure(text='请选择需要处理的问题')

	Labelframe1 = tk.LabelFrame(top)
	Labelframe1.place(relx=0.19, rely=0.19, relheight=0.63, relwidth=0.58)
	Labelframe1.configure(relief=tk.GROOVE)
	Labelframe1.configure(foreground='black')
	Labelframe1.configure(text='图像处理问题')
	Labelframe1.configure(background='#d9d9d9')
	Labelframe1.configure(width=180)

	Button1 = tk.Button(Labelframe1)
	Button1.place(relx=0.11, rely=0.2, relheight=0.15, relwidth=0.8, y=-16, h=8)

	Button1.configure(activebackground='#d9d9d9')
	Button1.configure(activeforeground='#000000')
	Button1.configure(background='#0bafd8')
	Button1.configure(disabledforeground='#a3a3a3')
	Button1.configure(foreground='#000000')
	Button1.configure(highlightbackground='#d9d9d9')
	Button1.configure(highlightcolor='black')
	Button1.configure(command=tuxiang_interface.c1)
	Button1.configure(text='膨胀和腐蚀')

	Button2 = tk.Button(Labelframe1)
	Button2.place(relx=0.11, rely=0.74, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button2.configure(activebackground='#d9d9d9')
	Button2.configure(activeforeground='#000000')
	Button2.configure(background='#0bafd8')
	Button2.configure(disabledforeground='#a3a3a3')
	Button2.configure(foreground='#000000')
	Button2.configure(highlightbackground='#d9d9d9')
	Button2.configure(highlightcolor='black')
	Button2.configure(command=tuxiang_interface.c2)
	Button2.configure(text='HIT和MISS')


def create6():
	top = tk.Toplevel()
	top.geometry('313x265')
	top.title('统计')
	top.configure(background='#d9d9d9')
	top.resizable(0, 0)
	Label1 = tk.Label(top)
	Label1.place(relx=0.19, rely=0.04, height=26, width=156)
	Label1.configure(background='#d9d9d9')
	Label1.configure(disabledforeground='#a3a3a3')
	Label1.configure(foreground='#000000')
	Label1.configure(text='请选择需要处理的问题')

	Labelframe1 = tk.LabelFrame(top)
	Labelframe1.place(relx=0.19, rely=0.19, relheight=0.63, relwidth=0.58)
	Labelframe1.configure(relief=tk.GROOVE)
	Labelframe1.configure(foreground='black')
	Labelframe1.configure(text='统计类型问题')
	Labelframe1.configure(background='#d9d9d9')
	Labelframe1.configure(width=180)

	Button1 = tk.Button(Labelframe1)
	Button1.place(relx=0.11, rely=0.2, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button1.configure(activebackground='#d9d9d9')
	Button1.configure(activeforeground='#000000')
	Button1.configure(background='#0bafd8')
	Button1.configure(disabledforeground='#a3a3a3')
	Button1.configure(foreground='#000000')
	Button1.configure(highlightbackground='#d9d9d9')
	Button1.configure(highlightcolor='black')
	Button1.configure(command=tongji_interface.c1)
	Button1.configure(text='连续和离散概率分布')

	Button2 = tk.Button(Labelframe1)
	Button2.place(relx=0.11, rely=0.74, relheight=0.15, relwidth=0.8, y=-16, h=8)
	Button2.configure(activebackground='#d9d9d9')
	Button2.configure(activeforeground='#000000')
	Button2.configure(background='#0bafd8')
	Button2.configure(disabledforeground='#a3a3a3')
	Button2.configure(foreground='#000000')
	Button2.configure(highlightbackground='#d9d9d9')
	Button2.configure(highlightcolor='black')
	Button2.configure(command=tongji_interface.c2)
	Button2.configure(text='二项分布')


root = tk.Tk()
root.geometry('800x600')
root.title('科学计算教学演示系统')
root.configure(background='#d9d9d9')
root.resizable(1, 1)

menubar = tk.Menu(root)

# 在顶级菜单实例下创建子菜单实例
amenu = tk.Menu(menubar)
for each in ['点', '数值']:
	amenu.add_command(label=each)

bmenu = tk.Menu(menubar)
# 为每个子菜单实例添加菜单项
for each in ['使用说明', '联系我们']:
	bmenu.add_command(label=each)

# 为顶级菜单实例添加菜单，并级联相应的子菜单实例
menubar.add_cascade(label='样本数据', menu=amenu)
menubar.add_cascade(label='关于', menu=bmenu)

# 顶级菜单实例应用到大窗口中
root['menu'] = menubar

Label1 = tk.Label(root)
Label1.place(relx=0.35, rely=0.06, height=26, width=255)
Label1.configure(activebackground='#f9f9f9')
Label1.configure(activeforeground='black')
Label1.configure(background='#d9d9d9')
Label1.configure(disabledforeground='#a3a3a3')
Label1.configure(foreground='#000000')
Label1.configure(highlightbackground='#d9d9d9')
Label1.configure(highlightcolor='black')
Label1.configure(text='欢迎使用科学计算教学演示系统!')

Labelframe1 = tk.LabelFrame(root)
Labelframe1.place(relx=0.07, rely=0.2, relheight=0.66, relwidth=0.83)
Labelframe1.configure(relief=tk.GROOVE)
Labelframe1.configure(foreground='black')
Labelframe1.configure(text='请选择科学计算问题类型')
Labelframe1.configure(background='#d9d9d9')
Labelframe1.configure(highlightbackground='#d9d9d9')
Labelframe1.configure(highlightcolor='black')

Button1 = tk.Button(Labelframe1)
Button1.place(relx=0.09, rely=0.23, relheight=0.05, relwidth=0.2, y=-16, h=8)
Button1.configure(activebackground='#d9d9d9')
Button1.configure(activeforeground='#000000')
Button1.configure(background='#16b8d8')
Button1.configure(disabledforeground='#a3a3a3')
Button1.configure(foreground='#000000')
Button1.configure(highlightbackground='#d9d9d9')
Button1.configure(highlightcolor='black')
Button1.configure(command=create1)
Button1.configure(text='优化')

Button2 = tk.Button(Labelframe1)
Button2.place(relx=0.4, rely=0.23, relheight=0.05, relwidth=0.2, y=-16, h=8)
Button2.configure(activebackground='#d9d9d9')
Button2.configure(activeforeground='#000000')
Button2.configure(background='#16b8d8')
Button2.configure(disabledforeground='#a3a3a3')
Button2.configure(foreground='#000000')
Button2.configure(highlightbackground='#d9d9d9')
Button2.configure(highlightcolor='black')
Button2.configure(command=create2)
Button2.configure(text='插值')

Button3 = tk.Button(Labelframe1)
Button3.place(relx=0.09, rely=0.69, relheight=0.05, relwidth=0.2, y=-16, h=8)
Button3.configure(activebackground='#d9d9d9')
Button3.configure(activeforeground='#000000')
Button3.configure(background='#16b8d8')
Button3.configure(disabledforeground='#a3a3a3')
Button3.configure(foreground='#000000')
Button3.configure(highlightbackground='#d9d9d9')
Button3.configure(highlightcolor='black')
Button3.configure(command=create3)
Button3.configure(text='积分')

Button4 = tk.Button(Labelframe1)
Button4.place(relx=0.4, rely=0.69, relheight=0.05, relwidth=0.2, y=-16, h=8)
Button4.configure(activebackground='#d9d9d9')
Button4.configure(activeforeground='#000000')
Button4.configure(background='#16b8d8')
Button4.configure(disabledforeground='#a3a3a3')
Button4.configure(foreground='#000000')
Button4.configure(highlightbackground='#d9d9d9')
Button4.configure(highlightcolor='black')
Button4.configure(command=create4)
Button4.configure(text='信号处理')

Button5 = tk.Button(Labelframe1)
Button5.place(relx=0.71, rely=0.23, relheight=0.05, relwidth=0.2, y=-16, h=8)
Button5.configure(activebackground='#d9d9d9')
Button5.configure(activeforeground='#000000')
Button5.configure(background='#16b8d8')
Button5.configure(disabledforeground='#a3a3a3')
Button5.configure(foreground='#000000')
Button5.configure(highlightbackground='#d9d9d9')
Button5.configure(highlightcolor='black')
Button5.configure(command=create5)
Button5.configure(text='图像处理')

Button6 = tk.Button(Labelframe1)
Button6.place(relx=0.71, rely=0.69, relheight=0.05, relwidth=0.2, y=-16, h=8)
Button6.configure(activebackground='#d9d9d9')
Button6.configure(activeforeground='#000000')
Button6.configure(background='#16b8d8')
Button6.configure(disabledforeground='#a3a3a3')
Button6.configure(foreground='#000000')
Button6.configure(highlightbackground='#d9d9d9')
Button6.configure(highlightcolor='black')
Button6.configure(command=create6)
Button6.configure(text='统计')

root.mainloop()
