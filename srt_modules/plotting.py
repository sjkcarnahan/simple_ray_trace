'''
Scott Carnahan
simple ray trace - plotting utilities
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def prep_rays_for_plot(ray_history):
    ray_list = []
    num_rays = np.shape(ray_history[-1])[1]
    ray_array = np.array(ray_history)
    for i in range(num_rays):
        xs = ray_array[:, 0, i]
        ys = ray_array[:, 1, i]
        zs = ray_array[:, 2, i]
        ray_list.append([xs, ys, zs])
    return ray_list

def scatter_surfaces(ax, surfaces):
    for i, surf in zip(range(len(surfaces)), surfaces):
        pts = surf.surface_points()
        ax.scatter(pts[0, :], pts[1, :], pts[2, :], label=surf.name, s=1)
    return

def plot_surfaces(ax, surfaces):
    for i, surf in zip(range(len(surfaces)), surfaces):
        X, Y, Z = surf.surface_mesh()
        ax.plot_surface(X, Y, Z, alpha=0.4)
    return

def plot_rays(ax, ray_hist, al=1.0):
    ray_hist = np.array(ray_hist)
    for r in range(np.shape(ray_hist)[2]):
            ray = ray_hist[:, :, r]
            ax.plot(ray[:, 0], ray[:, 1], ray[:, 2], color='black', alpha=al)
    return

def save_3d_plot(surfaces, ray_hist, path='/Users/sqc0815/Not_Backed_Up/hwRepo/ASTR5760', alpha=.015):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    plot_surfaces(ax, surfaces)
    plot_rays(ax, ray_hist, alpha)
    plt.title('3-D View of Ray Trace')
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    ax.set_zlabel('z [m]')
    plt.legend()
    # ax.set_aspect('equal')
    #plt.savefig(path + "/figures/3Dview.png")
    return fig

def plot_ray_starts(xs, ys, angle, path='/Users/sqc0815/Not_Backed_Up/hwRepo/ASTR5760'):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.scatter(-ys, -xs, s=2)
    ax.set_ylim([-0.5, 0.5])
    ax.set_xlim([-0.5, 0.5])
    ax.set_title('ray starts (inverted)')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.savefig(path + '/figures/offAxisRayStart' + str(angle) + '.png')
    return fig

def plot_flat_image(image_plane, L_X, angle, path='/Users/sqc0815/Not_Backed_Up/hwRepo/ASTR5760'):
    img = plt.figure()
    plt.clf()
    ax = img.add_subplot(111)
    pts = image_plane.extract_image(L_X)
    ax.scatter(pts[0, :], pts[1, :], label='image', color='black', s=1)
    max_x = np.max(pts[0, :][~np.isnan(pts[0, :])])
    min_x = np.min(pts[0, :][~np.isnan(pts[0, :])])
    max_y = np.max(pts[1, :][~np.isnan(pts[1, :])])
    min_y = np.min(pts[1, :][~np.isnan(pts[1, :])])
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('flat plane image at ' + str(angle) + "\'\'")
    # ax.set_aspect('equal')
    plt.savefig(path + '/figures/image_' + str(angle) + '.png')
    return img

def plot_flat_image_at_dist(image_plane, L_X, angle, offset, path='/Users/sqc0815/Not_Backed_Up/hwRepo/ASTR5760'):
    img = plt.figure()
    plt.clf()
    ax = img.add_subplot(111)
    pts = image_plane.extract_image(L_X) * 1000
    ax.scatter(pts[0, :], pts[1, :], label='image', color='black', s=1)
    max_x = np.max(pts[0, :][~np.isnan(pts[0, :])])
    min_x = np.min(pts[0, :][~np.isnan(pts[0, :])])
    max_y = np.max(pts[1, :][~np.isnan(pts[1, :])])
    min_y = np.min(pts[1, :][~np.isnan(pts[1, :])])
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('flat plane image at ' + str(angle) + "\'\'_" + str(offset))
    plt.savefig(path + '/figures/image_' + str(angle) + '_' + str(offset) + '.png')
    return img