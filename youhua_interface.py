﻿import tkinter as tk


def c1():
	top = tk.Toplevel()
	top.geometry('284x187')
	top.title('最小二乘拟合')
	Frame1 = tk.Frame(top)
	Frame1.place(relx=0.04, rely=0.05, relheight=0.56, relwidth=0.83)
	Frame1.configure(relief=tk.GROOVE)
	Frame1.configure(borderwidth='2')
	Frame1.configure(relief=tk.GROOVE)
	Frame1.configure(background='#d9d9d9')
	Frame1.configure(width=235)
	v1 = tk.StringVar()
	e1 = tk.Entry(Frame1, textvariable=v1, width=10)
	e1.grid(row=1, column=1, padx=1, pady=1)
	l1 = tk.Label(Frame1, text='样本数据').grid(row=1, column=0, padx=1, pady=1)
	v2 = tk.StringVar()
	e2 = tk.Entry(Frame1, textvariable=v2, width=10)
	e2.grid(row=2, column=1, padx=1, pady=1)
	l2 = tk.Label(Frame1, text='含参方程').grid(row=2, column=0, padx=1, pady=1)
	Button1 = tk.Button(top)
	Button1.place(relx=0.04, rely=0.64, height=33, width=104)
	Button1.configure(activebackground='#d9d9d9')
	Button1.configure(activeforeground='#000000')
	Button1.configure(background='#d9d9d9')
	Button1.configure(disabledforeground='#a3a3a3')
	Button1.configure(foreground='#000000')
	Button1.configure(highlightbackground='#d9d9d9')
	Button1.configure(highlightcolor='black')
	Button1.configure(pady='0')
	Button1.configure(text='样本数据绘图')

	Button2 = tk.Button(top)
	Button2.place(relx=0.46, rely=0.64, height=33, width=104)
	Button2.configure(activebackground='#d9d9d9')
	Button2.configure(activeforeground='#000000')
	Button2.configure(background='#d9d9d9')
	Button2.configure(disabledforeground='#a3a3a3')
	Button2.configure(foreground='#000000')
	Button2.configure(highlightbackground='#d9d9d9')
	Button2.configure(highlightcolor='black')
	Button2.configure(pady='0')
	Button2.configure(text='拟合并绘制曲线')


def c2():
	top = tk.Toplevel()
	top.geometry('284x187')
	top.title('求函数最小值')
	Frame1 = tk.Frame(top)
	Frame1.place(relx=0.04, rely=0.05, relheight=0.56, relwidth=0.83)
	Frame1.configure(relief=tk.GROOVE)
	Frame1.configure(borderwidth='2')
	Frame1.configure(relief=tk.GROOVE)
	Frame1.configure(background='#d9d9d9')
	Frame1.configure(width=235)
	v1 = tk.StringVar()
	e1 = tk.Entry(Frame1, textvariable=v1, width=10)
	e1.grid(row=1, column=1, padx=1, pady=1)
	l1 = tk.Label(Frame1, text='请输入函数f(x,y)').grid(row=1, column=0, padx=1, pady=1)
	Button1 = tk.Button(top)
	Button1.place(relx=0.04, rely=0.64, height=33, width=104)
	Button1.configure(activebackground='#d9d9d9')
	Button1.configure(activeforeground='#000000')
	Button1.configure(background='#d9d9d9')
	Button1.configure(disabledforeground='#a3a3a3')
	Button1.configure(foreground='#000000')
	Button1.configure(highlightbackground='#d9d9d9')
	Button1.configure(highlightcolor='black')
	Button1.configure(pady='0')
	Button1.configure(text='计算最小点')
	Button2 = tk.Button(top)
	Button2.place(relx=0.46, rely=0.64, height=33, width=104)
	Button2.configure(activebackground='#d9d9d9')
	Button2.configure(activeforeground='#000000')
	Button2.configure(background='#d9d9d9')
	Button2.configure(disabledforeground='#a3a3a3')
	Button2.configure(foreground='#000000')
	Button2.configure(highlightbackground='#d9d9d9')
	Button2.configure(highlightcolor='black')
	Button2.configure(pady='0')
	Button2.configure(text='绘制计算路径')


def c3():
	top = tk.Toplevel()
	top.geometry('284x187')
	top.title('非线性方程组求解')
	Frame1 = tk.Frame(top)
	Frame1.place(relx=0.04, rely=0.05, relheight=0.56, relwidth=0.83)
	Frame1.configure(relief=tk.GROOVE)
	Frame1.configure(borderwidth='2')
	Frame1.configure(relief=tk.GROOVE)
	Frame1.configure(background='#d9d9d9')
	Frame1.configure(width=235)
	v1 = tk.StringVar()
	e1 = tk.Entry(Frame1, textvariable=v1, width=10)
	e1.grid(row=1, column=1, padx=1, pady=1)
	l1 = tk.Label(Frame1, text='f1').grid(row=1, column=0, padx=1, pady=1)
	v2 = tk.StringVar()
	e2 = tk.Entry(Frame1, textvariable=v2, width=10)
	e2.grid(row=2, column=1, padx=1, pady=1)
	l2 = tk.Label(Frame1, text='f2').grid(row=2, column=0, padx=1, pady=1)
	v3 = tk.StringVar()
	e3 = tk.Entry(Frame1, textvariable=v2, width=10)
	e3.grid(row=3, column=1, padx=1, pady=1)
	l3 = tk.Label(Frame1, text='f3').grid(row=3, column=0, padx=1, pady=1)
	Button1 = tk.Button(top)
	Button1.place(relx=0.04, rely=0.64, height=33, width=104)
	Button1.configure(activebackground='#d9d9d9')
	Button1.configure(activeforeground='#000000')
	Button1.configure(background='#d9d9d9')
	Button1.configure(disabledforeground='#a3a3a3')
	Button1.configure(foreground='#000000')
	Button1.configure(highlightbackground='#d9d9d9')
	Button1.configure(highlightcolor='black')
	Button1.configure(pady='0')
	Button1.configure(text='方程组求解')

	Button2 = tk.Button(top)
	Button2.place(relx=0.46, rely=0.64, height=33, width=104)
	Button2.configure(activebackground='#d9d9d9')
	Button2.configure(activeforeground='#000000')
	Button2.configure(background='#d9d9d9')
	Button2.configure(disabledforeground='#a3a3a3')
	Button2.configure(foreground='#000000')
	Button2.configure(highlightbackground='#d9d9d9')
	Button2.configure(highlightcolor='black')
	Button2.configure(pady='0')
	Button2.configure(text='绘图')
