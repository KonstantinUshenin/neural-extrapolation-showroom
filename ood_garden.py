import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import Rbf


def circle(number_of_points, radius=1., examples=(1, -1)):
    a = np.linspace(-1.0, 1.0, number_of_points + 1)[:-1]
    b = np.linspace(-1.0, 1.0, number_of_points + 1)[:-1]

    x = -np.cos(np.pi * a) * radius
    y = np.sin(np.pi * b) * radius

    values = np.array(examples * (number_of_points // len(examples) + 1))[:number_of_points]

    return x, y, values


def flower_0():
    x0, y0, v0 = circle(7, radius=0.3, examples=(1, -1))
    x1, y1, v1 = circle(11, radius=0.5, examples=(-1, 1))
    x2, y2, v2 = circle(21, radius=0.9, examples=(1, -1))
    x3, y3, v3 = circle(7, radius=1.0, examples=(-1, 1))

    x_series = np.concatenate([x0, x1, x2, x3])
    y_series = np.concatenate([y0, y1, y2, y3])
    v_series = np.concatenate([v0, v1, v2, v3])

    return x_series, y_series, v_series


def rbf_analysis(x_series, y_series, v_series, extent=(-1.5, 1.5, -1.5, 1.5), resolution=1000, **argv):
    rbfi = Rbf(x_series, y_series, v_series, **argv)
    x = np.linspace(extent[0], extent[1], 1000)
    y = np.linspace(extent[2], extent[3], 1000)
    xv, yv = np.meshgrid(x, y)
    image = rbfi(xv, yv)
    return image


def draw_flower(x_series, y_series, v_series, image, title='title',
                figsize=(8, 8), extent=(-1.5, 1.5, -1.5, 1.5), cmap='jet', title_fontsize=16,
                vmin=None, vmax=None):
    vmin = vmin if vmin is not None else -np.abs(image).max()
    vmax = vmax if vmax is not None else np.abs(image).max()


    plt.figure(figsize=figsize)
    plt.title(title, fontdict={'fontsize': title_fontsize}, y=1.03)
    plt.imshow(image, extent=extent, origin='lower',
               vmin=vmin, vmax=vmax, cmap=cmap)
    plt.scatter(x_series, y_series, c=v_series, marker='o', edgecolors='k',
                vmin=vmin, vmax=vmax, cmap=cmap)
    cb = plt.colorbar(orientation="horizontal", pad=0.07, shrink=0.8)
    cb.ax.axvline(np.min(v_series), c='k')
    cb.ax.axvline(np.max(v_series), c='k')
    plt.tight_layout()
