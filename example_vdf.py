import analysator as pt # pt is historically used for analysator
import numpy as np
import matplotlib.pyplot as plt

f = pt.vlsvfile.VlsvReader("data/bulk1.0001612.vlsv")

probept = np.array([-8, 3, 0])*6371e3

cellid = f.get_cellid_with_vdf(probept)

print(cellid)

pt.plot.plot_vdf(vlsvobj=f, cellids=[cellid], xz=1, outputdir="./")

distr, edges = f.read_velocity_distribution_dense(cellid)
ex = edges[0]
ey = edges[1]
ez = edges[2]
# print(distr)
pc = plt.pcolor(ex,ez, np.squeeze(distr[:,60,:]).T, norm="log", vmin=7.0630006e-16, )
plt.colorbar()
pc.axes.set_aspect('equal')
plt.savefig("test.png", dpi=300)