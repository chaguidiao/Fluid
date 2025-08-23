#!/usr/bin/env python

from raylib import *
from Fluid import *

def toner_tu_force(fluid):
    # TODO To merge to Fluid class
    alpha = 2.7
    beta = 1.4
    noise_strength = 0.08
    v = fluid.get_velocity()
    speed_sq = (v[..., 0] ** 2) + (v[..., 1] ** 2)
    force_magnitude = alpha - beta * speed_sq
    noise = np.random.rand(*v.shape) * noise_strength
    v = np.einsum('ij,ijk->ijk', force_magnitude, v) + noise
    fluid.v += (v * fluid.dt)

def apply_upward_force(t, coord):
    def _apply_upward_force(t, coord):
        tdecay = np.maximum((1.0 - (0.05 * t)), 0.)
        upward_force = (
            tdecay
            *
            np.where(
                (
                    (coord[0] > 0.4)
                    &
                    (coord[0] < 0.6)
                    &
                    (coord[1] > 0.1)
                    &
                    (coord[1] < 0.3)
                ),
                np.array([0.0, 1.0]),
                np.array([0.0, 0.0])
            )
        )
        return upward_force

    return np.vectorize(pyfunc=_apply_upward_force, signature='(),(d)->(d)')(t, coord)


def draw_density(dens, window_width, window_height, cell_size):
    z = dens.copy()
    z *= 255.0
    z[z > 255.] = 255
    z = z.astype('uint8')
    x = np.linspace(0, window_width, z.shape[0])
    y = np.linspace(0, window_height, z.shape[1])
    x, y = np.meshgrid(x, y)
    rlBegin(RL_QUADS)
    for i in range(z.shape[0] - 1):
        for j in range(z.shape[1] - 1):
            _z = z[i, j].item()
            rlColor4ub(*([_z]*3) + [255])
            rlVertex2f(x[i, j], y[i, j])
            _z = z[i+1, j].item()
            rlColor4ub(*([_z]*3) + [255])
            rlVertex2f(x[i+1, j], y[i+1, j])
            _z = z[i+1, j+1].item()
            rlColor4ub(*([_z]*3) + [255])
            rlVertex2f(x[i+1, j+1], y[i+1, j+1])
            _z = z[i, j+1].item()
            rlColor4ub(*([_z]*3) + [255])
            rlVertex2f(x[i, j+1], y[i, j+1])
    rlEnd()


def draw_velocity(v, window_width, window_height, cell_size):
    h = 1.0 / cell_size
    x = np.linspace(-0.5, cell_size - 0.5, cell_size + 1) * h
    y = np.linspace(-0.5, cell_size - 0.5, cell_size + 1) * h
    x, y = np.meshgrid(x, y)
    red = ffi.new('struct Color *', [255, 0, 0, 255])[0]
    for i in range(1, v.shape[0] - 2):
        for j in range(1, v.shape[1] - 2):
            xstart = x[i, j] * window_width
            ystart = y[i, j] * window_height
            xend = (x[i, j] + v[i, j, 0]) * window_width
            yend = (y[i, j] + v[i, j, 1]) * window_height
            xstart, ystart, xend, yend = int(xstart), int(ystart), int(xend), int(yend)
            DrawLine(xstart, ystart, xend, yend, red)


def get_source_from_UI(source, window_width, window_height, dens_source, force, omx, omy):
    if IsMouseButtonDown(MOUSE_BUTTON_RIGHT):
        vec = GetMousePosition()
        x = int(vec.x / window_width * source.shape[1]) # Column for width
        y = int(vec.y / window_height * source.shape[0])
        source[y, x, 0] = dens_source
        omx, omy = x, y
    elif IsMouseButtonDown(MOUSE_BUTTON_LEFT):
        vec = GetMousePosition()
        x = int(vec.x / window_width * source.shape[1]) # Column for width
        y = int(vec.y / window_height * source.shape[0])
        source[y, x, 1] = force * (x - omx)
        source[y, x, 2]  = force * (omy - y)
        #print(source[y, x, 1], source[y, x, 2], x - omx, omy - y)
        omx, omy = x, y

    return omx, omy

def forcing_function(time, point):
    time_decay = np.maximum(
        2.0 - 0.5 * time,
        0.0,
    )
    forced_value = (
        time_decay
        *
        np.where(
            (
                (point[0] > 0.4)
                &
                (point[0] < 0.6)
                &
                (point[1] > 0.1)
                &
                (point[1] < 0.3)
            ),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )
    )
    return forced_value

forcing_function_vectorized = np.vectorize(
    pyfunc=forcing_function,
    signature='(),(d)->(d)',
)

def main():
    window_width = 800
    window_height = 600
    N = 41
    show_velocity = True
    show_fps = True
    omx, omy = 0, 0
    force = 10.

    fluid = Fluid(N=N, force=force, dt=0.1, solver='iter')
    # 1st is density, 2nd and 3rd are velocity components
    # respectively
    source = np.zeros((N + 2, N + 2, 3))

    x = np.linspace(0.0, 1.0, N + 2)
    y = np.linspace(0.0, 1.0, N + 2)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coord = np.concatenate(
        (
            X[..., np.newaxis],
            Y[..., np.newaxis],
        ),
        axis=-1,
    )

    InitWindow(window_width, window_height, b"Fluid dynamics the numpy way")
    SetWindowState(FLAG_WINDOW_RESIZABLE)
    SetTargetFPS(30)
    t = 0.
    dt = 0.1
    while not WindowShouldClose():
        t = t + dt

        # Inject upward force for testing
        if IsKeyDown(KEY_W):
            t = 0.
            source[:, :, 1:] = forcing_function_vectorized(t, coord)
            t_decay = np.maximum((1.0 - (0.05 * t)), 0.)
            #source[N//2, N//2, 0] = (10 * np.random.randn() + 20) * t_decay
            source[..., 1:] = apply_upward_force(t, coord) * 20

        # Toner-Tu force for collective motion modeling
        toner_tu_force(fluid)

        # Interactive input
        omx, omy = get_source_from_UI(source, GetScreenWidth(), GetScreenHeight(), fluid.dens_source, fluid.force, omx, omy)

        if IsKeyPressed(KEY_V):
            show_velocity = not show_velocity
        if IsKeyPressed(KEY_C):
            fluid.reset()
        if IsKeyPressed(KEY_F):
            show_fps = not show_fps
        fluid.update_velocity(source)
        fluid.update_density(source)
        BeginDrawing()
        ClearBackground(BLACK)
        if show_velocity:
            draw_velocity(fluid.get_velocity(), GetScreenWidth(), GetScreenHeight(), N)
        else:
            draw_density(fluid.get_density(), GetScreenWidth(), GetScreenHeight(), N)
        if show_fps:
            DrawFPS(1, 1)
        EndDrawing()
        source = np.zeros((N + 2, N + 2, 3))
    CloseWindow()


if __name__ == '__main__':
    main()
