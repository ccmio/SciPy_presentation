import pylab as pl
import numpy as np
import scipy.optimize as opt
from scipy.optimize import fsolve
from scipy import constants as c
from math import sin, cos
from scipy import integrate
from scipy import interpolate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.ndimage import morphology
from scipy.ndimage import binary_hit_or_miss
from scipy import stats
from sympy import Symbol
import sympy


# 直线最小二乘拟合
def least_square(str):
    def ls_line():

        x = np.linspace(0, 10, 100)
        y_noise = 1.5*x - 0.5 + np.random.standard_normal(len(x)) * 1.5

        def residuals(p):
            k, b = p
            return 1.5*x - 0.5 - (k * x + b)

        r = opt.leastsq(residuals, [1.3, -0.3])
        k, b = r[0]
        y = k * x + b
        plt.scatter(x, y_noise, color="r", marker='o', s=50)
        plt.plot(x, y, color="b", linewidth=3)
        plt.show()

        def residuals(p):
            k, b = p
            return f(x) - (k*x + b)

        def draw_noise_data():
            plt.scatter(x, y_noise, color="r", marker='o', s=50)
            plt.show()

        def draw_final():
            r = opt.leastsq(residuals, [1.3, -0.3])
            k, b = r[0]
            y = k * x + b
            plt.scatter(x, y_noise, color="r", marker='o', s=50)
            plt.plot(x, y, color="b", linewidth=3)
            plt.show()

    def ls_sin():
        def func(x, p):
            a, k, theta = p
            return a * np.sin(2 * np.pi * k * x + theta)

        def residuals(p, y, x):
            return y - func(x, p)
        x = np.linspace(0, 2 * c.pi, 100)
        a, k, theta = 10, 0.34, c.pi / 6
        y0 = func(x, [a, k, theta])
        y1 = y0 + 2*np.random.randn(len(x))
        p0 = [9, 0.3, c.pi / 5]
        plt.scatter(x, y1, color="r", marker='o', s=50)
        plsq = opt.leastsq(residuals, p0, args=(y1, x))
        plt.plot(x, func(x, plsq[0]), color="b", linewidth=3)
        plt.show()

    if str == '直线':
        ls_line()
    elif str == '正弦曲线':
        ls_sin()
    else:
        return

# 正弦最小二乘拟合
# least_square(input('请输入类型'))


# 求函数最小值
def fmin(str):
    x = Symbol('x')
    y = Symbol('y')

    # 超级巨坑，sympy的int和float类型numpy不认，只能用lambdify把表达式转换为函数

    function = eval(str)
    f_func = sympy.lambdify((x, y), function)
    points = []
    init_point = [10.0, 10.0]

    def f(p):
        x_value, y_value = p
        z = f_func(x_value, y_value)
        points.append((x_value, y_value, z))
        return z

    def fprime(p):
        x_value, y_value = p
        x_prime = sympy.diff(function, x)
        y_prime = sympy.diff(function, y)
        dx = sympy.lambdify((x, y), x_prime)
        dy = sympy.lambdify((x, y), y_prime)
        array = np.array([dx(x_value, y_value), dy(x_value, y_value)])
        return array

# 优化方法任意选择，可以用if分支选
    # 这两种优化方法没用到偏导
    # result = opt.fmin(f, init_point)
    #result = opt.fmin_powell(f, init_point)

    # 用到偏导的：
    result = opt.fmin_cg(f, np.array(init_point), fprime=fprime)
    # result = opt.fmin_bfgs(f, init_point, fprime=fprime)
    # result = opt.fmin_tnc(f, init_point, fprime=fprime)
    # result = opt.fmin_l_bfgs_b(f, init_point, fprime=fprime)

    # 其它
    # result = opt.fmin_cobyla(f, init_point, [])

    p = np.array(points)
    xmin, xmax = np.min(p[:, 0]-1), np.max(p[:, 0])+1
    ymin, ymax = np.min(p[:, 1]-1), np.max(p[:, 1])+1
    y_new, x_new = np.ogrid[ymin:ymax:500j, xmin:xmax:500j]
    aa = f([x_new, y_new])
    z = np.log10(aa)
    print(aa, type(aa))
    zmin, zmax = np.min(z), np.max(z)
    pl.imshow(z, extent=(xmin, xmax, ymin, ymax), origin='bottom', aspect='auto')
    pl.plot(p[:, 0], p[:, 1])
    pl.scatter(p[:, 0], p[:, 1], c=range(len(p)))
    pl.scatter(xmax, ymax, c='b')
    pl.xlim(xmin, xmax)
    pl.ylim(ymin, ymax)
    pl.show()
    print(result, '函数最小值为: ', 10**zmin)


# 求解非线性方程组5x1+3=0, 4x0^2 - 2sin(x1x2)=0, x1x2-1.5=0
def scipy_fsolve1():
    def f(x):
        x0, x1, x2 = x.tolist()
        return[
            5*x1 + 3,
            4*x0*x0 - 2*sin(x1*x2),
            x1*x2 - 1.5
        ]
    result = fsolve(f, [1, 1, 1])
    print(result)
    print(f(result))


# 求解非线性方程组cos(a) = 1 - d^2 / (2*R^2), L = a * r
def scipy_fsolve2():
    def f(x):
        d = 140
        l = 156
        a, r = x.tolist()
        return [
            cos(a) - 1 + (d*d)/(2*r*r),
            l - r * a
        ]
    def j(x):
        d = 140
        l = 156
        a, r = x.tolist()
        return [
            [-sin(a), (d*d)/4*r],
            [-r, -a]
        ]
    result = opt.fsolve(f, [1, 1], fprime=j)
    print(result)
    print(f(result))


# B样条曲线插值
def scipy_interpld():
    # 决定插值曲线的数据点有11个，插值之后的数据点有101个
    x = np.linspace(0, 10, 11)
    y = np.sin(x)
    xnew = np.linspace(0, 10, 101)
    plt.plot(x, y, 'ro')
    for kind in ['nearest', 'zero', 'slinear', 'quadratic']:
        f = interpolate.interp1d(x, y, kind=kind)
        # interpld可以计算x的取值范围内的任意点的函数值
        ynew = f(xnew)
        plt.plot(xnew, ynew, label=str(kind))
    plt.legend()
    plt.show()
    # plt.savefig("c:\\figure1.png")

# 外推插值
def scipy_uspline():
    x1 = np.linspace(0, 10, 20)
    y1 = np.sin(x1)
    sx1 = np.linspace(0, 12, 100)
    sy1 = interpolate.UnivariateSpline(x1, y1, s=0)(sx1)
    # 外推运算使得输入数据x的值没有大于10的点但是能计算出在10到12之间的数值
    x2 = np.linspace(0, 20, 200)
    y2 = np.sin(x2) + np.random.standard_normal(len(x2)) * 0.2
    sx2 = np.linspace(0, 20, 2000)
    spline2 = interpolate.UnivariateSpline(x2, y2, s=8)
    sy2 = spline2(sx2)
    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    plt.plot(x1, y1, ".", label='data')
    plt.plot(sx1, sy1, label='spline_curve')
    plt.legend()
    plt.subplot(212)
    plt.plot(x2, y2, ".", label='data')
    plt.plot(sx2, sy2, linewidth=2, label='spline_curve')
    plt.plot(x2, np.sin(x2), label='')
    plt.legend()

    # 计算曲线和横线的交点
    def root_at(self, v):
        coeff = self.get_coeffs()
        coeff -= v
        try:
            root = self.roots()
            return root
        finally:
            coeff += v

    interpolate.UnivariateSpline.roots_at = root_at
    plt.plot(sx2, sy2, lw=2, label='spline_curve')
    ax = plt.gca()
    for level in [0.5, 0.75, -0.5, -0.75]:
        ax.axhline(level, ls=':', color='k')
        xr = spline2.roots_at(level)
        plt.plot(xr, spline2(xr), 'ro')
    plt.show()
    # plt.savefig("c:\\figure1.png")


# 二维插值
def scipy_interp2d():
    def func(x, y):
        return (x + y) * np.exp(-5.0 * (x ** 2 + y ** 2))
    # X-Y轴分为15*15的网格
    y, x = np.mgrid[-1:1:15j, -1:1:15j]
    fvals = func(x, y)  # 计算每个网格点上的函数值  15*15的值
    print(len(fvals[0]))
    # 三次样条二维插值
    newfunc = interpolate.interp2d(x, y, fvals, kind='cubic')
    # 计算100*100的网格上的插值
    xnew = np.linspace(-1, 1, 100)  # x
    ynew = np.linspace(-1, 1, 100)  # y
    fnew = newfunc(xnew, ynew)  # 仅仅是y值   100*100的值
    # 为了更明显地比较插值前后的区别，使用关键字参数interpolation='nearest'
    # 关闭imshow()内置的插值运算。
    pl.subplot(121)
    im1 = pl.imshow(fvals, extent=[-1, 1, -1, 1], interpolation='nearest', origin="lower")  # pl.cm.jet
    # extent=[-1,1,-1,1]为x,y范围  favals为
    pl.colorbar(im1)
    pl.subplot(122)
    im2 = pl.imshow(fnew, extent=[-1, 1, -1, 1], interpolation='nearest', origin="lower")
    pl.colorbar(im2)
    pl.show()


# 圆面积积分
def integrate_circle():
    def half_circle(x):
        return (1 - x ** 2) ** 0.5
    pi_half, err = integrate.quad(half_circle, -1, 1)
    print(pi_half * 2)


# 球体积积分
def integrate_ball():
    def half_circle(x):
        return (1 - x ** 2) ** 0.5

    def half_sphere(x, y):
        return (1 - x ** 2 - y ** 2) ** 0.5
    v, err = integrate.dblquad(half_sphere, -1, 1, lambda x: -half_circle(x), lambda x: half_circle(x))
    print(v, err)


# 解常微分方程组(洛伦兹吸引子）其他实例：https://blog.csdn.net/weixin_42376039/article/details/86485817
def odeint_lorenz():
    def lorenz(w, t, p, r, b):
        x, y, z = w.tolist()
        return p*(y-x), x*(r-z)-y, x*y-b*z

    t = np.arange(0, 30, 0.01)
    track1 = odeint(lorenz, (0.0, 1.00, 0.0), t, args=(10.0, 28.0, 3.0))
    track2 = odeint(lorenz, (0.0, 1.01, 0.0), t, args=(10.0, 28.0, 3.0))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(track1[:, 0], track1[:, 1], track1[:, 2])
    ax.plot(track2[:, 0], track2[:, 1], track2[:, 2])
    plt.show()


# 中值滤波
def scipy_signal_noise_filter():
    t = np.arange(0, 20, 0.1)
    x = np.sin(t)
    x[np.random.randint(0, len(t), 20)] += np.random.standard_normal(20) * 0.6
    x2 = signal.medfilt(x, 5)  # 5是窗口值得大小，必须为奇数
    plt.plot(t, x, color="red", lw=3, alpha=0.4, label='with noise')
    plt.plot(t, x2, lw=1, label='filtered')
    plt.legend()
    plt.show()
    # plt.savefig("c:\\figure1.png")


# 图像处理 膨胀&腐蚀
def image_dilation():
    def expand_image(img, value, out=None, size=10):
        if out is None:
            w, h = img.shape
            out = np.zeros((w * size, h * size), dtype=np.uint8)
        tmp = np.repeat(np.repeat(img, size, 0), size, 1)
        out[:, :] = np.where(tmp, value, out)
        out[::size, :] = 0
        out[:, ::size] = 0
        return out

    def show_image(*imgs):
        for idx, img in enumerate(imgs, 1):
            ax = plt.subplot(1, len(imgs), idx)
            plt.imshow(img, cmap='gray')
            ax.set_axis_off()
        # plt.subplots_adjust(0.02, 0, 0.98, 1, 0, 0)
        plt.show()

    def dilation_demo(a, structure=None):
        b = morphology.binary_dilation(a, structure)
        img = expand_image(a, 255)
        return expand_image(np.logical_xor(a, b), 150, out=img)

    def erosion_demo(a, structure=None):
        b = morphology.binary_erosion(a, structure)
        img = expand_image(a, 255)
        return expand_image(np.logical_xor(a, b), 150, out=img)

    a = plt.imread("C:/Users/cacho/Desktop/luotuo.jpg")[:, :, 0].astype(np.uint8)
    img1 = expand_image(a, 255)
    img2_d = dilation_demo(a)  # 默认是4连通 0 1 0 111  010
    img3_d = dilation_demo(a, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 8连通
    img2_e = erosion_demo(a)  # 默认是4连通 0 1 0 111  010
    img3_e = erosion_demo(a, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 8连通
    show_image(a, img1, img2_d, img3_d)
    show_image(a, img1, img2_e, img3_e)


# 图像处理 hit and miss 细线化应用
def image_skeletonization():
    def skeleton(img):
        h1 = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]])
        m1 = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        h2 = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]])
        m2 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        hit_list = []
        miss_list = []
        for k in range(4):
            hit_list.append(np.rot90(h1, k))
            hit_list.append(np.rot90(h2, k))
            miss_list.append(np.rot90(m1, k))
            miss_list.append(np.rot90(m2, k))
        img = img.copy()
        while True:
            last = img
            for hit, miss in zip(hit_list, miss_list):
                hm = binary_hit_or_miss(img, hit, miss)
                # 删除白色点
                img = np.logical_and(img, np.logical_not(hm))
            if np.all(img == last):
                break
        return img
    a = pl.imread('C:/Users/cacho/Desktop/test.png')[:, :, 0].astype(np.uint8)
    b = skeleton(a)
    pl.imshow(b)
    pl.show()


# 连续概率分布
def scipy_stats():
    x_base = stats.norm(loc=1.0, scale=2.0)
    x = x_base.rvs(size=10000)
    np.mean(x)
    np.var(x)
    stats.norm.fit(x)
    t = np.arange(-10, 10, 0.01)
    pl.plot(t, x_base.pdf(t))
    p, t2 = np.histogram(x, bins=100, normed=True)
    t2 = (t2[:-1] + t2[1:])/2
    pl.plot(t2, p)
    pl.plot(t, x_base.cdf(t))
    pl.plot(t2, np.add.accumulate(p)*(t2[1]-t2[0]))
    plt.show()


def binom_poisson():
    _lambda = 10
    pl.figure(figsize=(10, 4))
    for i, time in enumerate([1000, 50000]):
        t = np.random.uniform(0, time, size=_lambda * time)  # 产生_lambda*time个（0,time)的数
        count, time_edges = np.histogram(t, bins=time, range=(0, time))  # 统计时间发生次数
        dist, count_edges = np.histogram(count, bins=20, range=(0, 20), normed=True)  # 统计20秒内的概率分布，normed为True结果和概率密度相等
        x = count_edges[:-1]
        poisson = stats.poisson.pmf(x, _lambda)
        pl.subplot(121 + i)
        pl.plot(x, dist, "-o", lw=2, label=u"统计结果")
        pl.plot(x, poisson, "->", lw=2, label=u"泊松分布", color="red")
        pl.xlabel(u"次数")
        pl.ylabel(u"概率")
        pl.title(u"time = %d" % time)
        pl.legend(loc="lower center")
        plt.show()

# binom_poisson()
# 显示拟合结果
# org_data = plt.scatter(x, y, color="r", marker='o', s=50)
# est_data = plt.plot(x, y_est, color="b", linewidth=3)
#
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Fit funtion with leastseq method")
# plt.legend(["Noise data", "Fited function"]);
# plt.show()