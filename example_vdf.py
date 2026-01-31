import analysator as pt # pt is historically used for analysator
import numpy as np
import matplotlib.pyplot as plt

f = pt.vlsvfile.VlsvReader("data/bulk1.0001612.vlsv")

probept = np.array([-8, 3, 0])*6371e3

cellid = f.get_cellid_with_vdf(probept)

print(cellid)

vth = f.read_variable("proton/vg_thermalvelocity", cellids=cellid)

print("Distribution has thermal velocity of "+str(vth)+" m/s")

pt.plot.plot_vdf(vlsvobj=f, cellids=[cellid], xz=1, outputdir="./", title="Projected VDF")

distr, edges = f.read_velocity_distribution_dense(cellid)

downsample=True

if downsample:
    distr = distr[::2,::2,::2]
    ex = edges[0][::2]
    ey = edges[1][::2]
    ez = edges[2][::2]
else:
    ex = edges[0][::2]
    ey = edges[1][::2]
    ez = edges[2][::2]

pc = plt.pcolor(ex,ez, np.squeeze(distr[:,ey.size//2,:]).T, norm="log", vmin=7.0630006e-16, )
plt.colorbar()
pc.axes.set_aspect('equal')
plt.title("Slice of the dense VDF distribution around $v_y$ midpoint")
plt.xlabel("$v_x / \mathrm{m}\,\mathrm{s}^{-1}$")
plt.ylabel("$v_z / \mathrm{m}\,\mathrm{s}^{-1}$")
plt.savefig("dense_distribution_slice.png", dpi=300)
plt.close()

dxs = (ex[1:] - ex[:-1])
dys = (ey[1:] - ey[:-1])
dzs = (ez[1:] - ez[:-1])

# centers, not used
# xs = ex[:-1]+dxs/2
# ys = ey[:-1]+dys/2
# zs = ez[:-1]+dzs/2

X,Y,Z = np.meshgrid(ex[:-1],ey[:-1],ez[:-1], indexing="ij")
dX,dY,dZ = np.meshgrid(dxs,dys,dzs, indexing="ij")

lX = X.reshape((-1))
lY = Y.reshape((-1))
lZ = Z.reshape((-1))

ldistr = distr.reshape((-1))
plt.hist(distr.reshape((-1)), bins=30, log=True)
plt.yscale("log")
plt.title("Histogram of v-space sample weights")
plt.xlabel("$f(v)$")
plt.savefig("vdf-hist.png")
plt.close()

samples = 1000000

choice = np.random.choice(lX.size, size=samples, p =ldistr/np.sum(ldistr))
particle_vs = np.random.random_sample(size=(samples,3))

ldX = dX.reshape((-1))
ldY = dY.reshape((-1))
ldZ = dZ.reshape((-1))

particle_vs *= np.stack((ldX[choice],ldY[choice],ldZ[choice]),axis=1) # scale by v-cell size
particle_vs += np.stack((lX[choice],lY[choice],lZ[choice]), axis=1) # offset by chosen bin edges

nbins=30
h,_,_,_ = plt.hist2d(particle_vs[:,0], particle_vs[:,2], bins=nbins, norm="log")

plt.clim(vmin=np.median(h)/3)
plt.axis("equal")
plt.colorbar(label="counts")
plt.xlabel("$v_x / \mathrm{m}\,\mathrm{s}^{-1}$")
plt.ylabel("$v_z / \mathrm{m}\,\mathrm{s}^{-1}$")
plt.title(f"Counts in $(v_x,v_z)$ ({samples} samples, {nbins}$^2$ bins)")
plt.savefig("VDF_sampled_histogram.png")

print(distr.shape)

np.savez("VDF_example",allow_pickle=False, dense_array=distr, dense_array_edges_x=ex, dense_array_edges_y=ey, dense_array_edges_z=ez, sampled_particles=particle_vs)

np.savez("VDF_example_particles",allow_pickle=False, sampled_particles=particle_vs)

vitro_norm = 1/50000 # Norm comes straight out of a hat. Not the tophat, this is completely arbirtary.
np.savetxt("/home/mjalho/PO/vitro/src/imca-legacy/vdf_files/VDF_example_particles.txt", particle_vs*vitro_norm, delimiter=' ')
