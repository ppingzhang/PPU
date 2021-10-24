from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_pcds(filename, pcds, titles, use_color=[],color=None, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [ 3 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) *20, 20))
    for i in range(1):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            clr = color[j]
            if color is None or not use_color[j]:
                clr = pcd[:, 0]

            ax = fig.add_subplot(1, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            #print(np.min(np.array(pcd)),np.max(np.array(pcd)))
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], marker='o', zdir=zdir, c=clr, s=size, vmin=-1, vmax=0.5, alpha=0.7)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    #plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

if __name__ == '__main__':

    pwd = './data/test/output/pu_unet/'
    res_all = [2,4,8,16,32,64,128]

    pcds = []
    pcds_res = []
    pname = 'cow'
    for res in res_all:
        p_path = '{pwd}x{res1}/xyz/{pname}_2048_x{res2}.xyz'.format(pwd=pwd, res1=res, pname=pname, res2=res)

        pcds.append(np.loadtxt(p_path))
        pcds_res.append('x'+str(res))


    plot_pcds(pname+'_result.png', pcds, pcds_res, use_color=[0, 0, 0, 0, 0, 0, 0], color=[None, None, None, None, None,  None, None])
