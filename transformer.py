import pygame as pg
import colors
import numpy as np
#import time


def f1(x,y):
    return np.sin(y)

def f2(x,y):
    return np.cos(x)


pg.init()
SIZE = 500
MAJOR_DENSITY = 7
MINOR_DENSITY = 3
x_range = (-1.0, 1.0)
extent = (1, 1)    # horizontal, vertical extensions above range of drawing grid, must be positive
y_range = (-1.0, 1.0)
y_size = y_range[1] - y_range[0]
x_size = x_range[1] - x_range[0]

x_ext_size = x_size + 2*extent[0]
y_ext_size = y_size + 2*extent[1]

flags = pg.DOUBLEBUF
gd = pg.display.set_mode((SIZE, SIZE), flags)
FPS = 15
pg.event.set_allowed([pg.QUIT])

back = gd.get_rect()
gd.set_alpha(None)

pg.display.set_caption("Transformer")
clock = pg.time.Clock()

def gen_line_set(init_points, size, start, number, dim):
    offsets = np.expand_dims((start+size*np.arange(number+1).astype(np.float32)/number),1)    # 8 major lines
    lines = np.repeat(np.expand_dims(init_points,0), number+1, axis=0)
    lines[:,:,dim] += offsets
    return lines.reshape(-1,2)


def f(x):
    return np.stack((f1(x[:,0], x[:,1]), f2(x[:,0], x[:,1])), 1)

horz_axis_start = np.array([[(x_range[0]-extent[0])+i*x_ext_size/SIZE, 0] for i in range(SIZE)])
vert_axis_start = np.array([[0, (y_range[0]-extent[1])+i*y_ext_size/SIZE] for i in range(SIZE)])
horz_axis_end = f(horz_axis_start)
vert_axis_end = f(vert_axis_start)

vert_major_start = gen_line_set(vert_axis_start, x_ext_size, x_range[0]-extent[0], MAJOR_DENSITY, 0)
horz_major_start = gen_line_set(horz_axis_start, y_ext_size, y_range[0]-extent[1], MAJOR_DENSITY, 1)
horz_major_end = f(horz_major_start)
vert_major_end = f(vert_major_start)

vert_minor_start = gen_line_set(vert_axis_start, x_ext_size, x_range[0]-extent[0], MINOR_DENSITY*MAJOR_DENSITY, 0)
horz_minor_start = gen_line_set(horz_axis_start, y_ext_size, y_range[0]-extent[1], MINOR_DENSITY*MAJOR_DENSITY, 1)
horz_minor_end = f(horz_minor_start)
vert_minor_end = f(vert_minor_start)

vert_axis_dir  = (vert_axis_end - vert_axis_start)
vert_major_dir = (vert_major_end - vert_major_start)
vert_minor_dir = (vert_minor_end - vert_minor_start)

horz_axis_dir  = (horz_axis_end - horz_axis_start)
horz_major_dir = (horz_major_end - horz_major_start)
horz_minor_dir = (horz_minor_end - horz_minor_start)

def val_to_idx(points):
    p_cpy = np.copy(points)
    p_cpy[:,0] -= x_range[0]
    p_cpy[:,1] -= y_range[0]
    p_cpy[:,0] *= SIZE/x_size

    p_cpy[:,1] *= -SIZE/y_size    # saying p -> scale - p*scale
    p_cpy[:,1] += SIZE
    return p_cpy.astype(np.int32)


def horz_color(width, start_points, direction, t, pixel_array, color_val):   # "horizontal" line drawer
    new_points = t*direction + start_points
    new_indices = val_to_idx(new_points)
    for ind in new_indices:
        if ind[0] >= 0:
            for w in range(-width//2,width//2+1):
                if ind[1]+w >= 0:
                    try:
                        pixel_array[int(ind[0]), int(ind[1]+w)] = color_val
                    except IndexError:
                        pass
    return pixel_array

def vert_color(width, start_points, direction, t, pixel_array, color_val):   # "horizontal" line drawer
    new_points = t*direction + start_points
    new_indices = val_to_idx(new_points)
    for ind in new_indices:
        if ind[1] >= 0:
            for w in range(-width//2,width//2+1):
                if ind[0]+w >= 0:
                    try:
                        pixel_array[int(ind[0]+w), int(ind[1])] = color_val
                    except IndexError:
                        pass
def main():
    running = True
    step = 0.01
    current = 0.0
    factor_x = float(x_ext_size)/(MAJOR_DENSITY*MINOR_DENSITY)
    factor_y = float(y_ext_size)/(MAJOR_DENSITY*MINOR_DENSITY)
    while running:
        for evt in pg.event.get():
            if evt.type == pg.QUIT:
                running = False
        gd.fill(colors.BLACK)

        pixels = pg.PixelArray(gd)     # surface is now locked
        for i in range(MAJOR_DENSITY*MINOR_DENSITY):
            scaled_i_x = int(SIZE*(i*factor_x-extent[0])/x_size)
            scaled_i_y = int(SIZE*(i*factor_y-extent[1])/y_size)
            if 0 < scaled_i_x < SIZE:
                pixels[scaled_i_x, :] = colors.CRIMSON
            if 0 < scaled_i_y < SIZE:
                pixels[:, scaled_i_y] = colors.CRIMSON

        horz_color(1, horz_minor_start, horz_minor_dir, current, pixels, colors.GRAY)
        vert_color(1, vert_minor_start, vert_minor_dir, current, pixels, colors.GRAY)

        horz_color(2, horz_major_start, horz_major_dir, current, pixels, colors.SKY_BLUE)
        vert_color(2, vert_major_start, vert_major_dir, current, pixels, colors.SKY_BLUE)

        horz_color(4, horz_axis_start, horz_axis_dir, current, pixels, colors.WHITE)
        vert_color(4, vert_axis_start, vert_axis_dir, current, pixels, colors.WHITE)
        del pixels                     # surface is unlocked
        current = min(1.0,current+step)
        pg.display.update()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
pg.quit()
