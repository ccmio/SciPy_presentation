﻿import tkinter as tk


def c1():
	top = tk.Toplevel()
	top.geometry('284x187')
	top.title('膨胀和腐蚀')
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
	l1 = tk.Label(Frame1, text='选择图像').grid(row=1, column=0, padx=1, pady=1)
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
	Button1.configure(text='确定')


def c2():
	top = tk.Toplevel()
	top.geometry('284x187')
	top.title('Hit和Miss')
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
	l1 = tk.Label(Frame1, text='图像').grid(row=1, column=0, padx=1, pady=1)

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
	Button1.configure(text='确定')
