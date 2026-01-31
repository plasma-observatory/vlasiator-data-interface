import analysator as pt # pt is historically used for analysator
import numpy as np
import matplotlib.pyplot as plt

f = pt.vlsvfile.VlsvReader("/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/restart/restart.0001610.2023-05-14_16-21-41.vlsv")
fb = pt.vlsvfile.VlsvReader("/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.0001610.vlsv")

# these are from the old draft figure, cells along an x-directed line
cellids_BBF_plot = [359048814,
359048815,
359048816,
359048817,
359048818,
359048819,
359048820,
359048821,
359048822,
359048823,
359048824,
359048825,
359048826,
359048827,
359048828,
359048829,
359048830,
359048831,
359048832,
359048833,
359048834,
359048835,
359048836,
359048837,
359048838,
359048839,
359048842,
359048841,
359048843,
359048840]

pts = f.get_cell_coordinates(np.array(cellids_BBF_plot))



# find and expand a bounding box
mins = np.min(pts, axis=0)
maxs = np.max(pts, axis=0)

Dy = 1000e3
Dz = 1000e3

samples = int(1e5)

xmin = mins[0]
xmax = maxs[0]

ymin = mins[1]-Dy
ymax = maxs[1]+Dy

zmin = mins[2]-Dz
zmax = maxs[2]+Dz

cellids_box = np.array(pt.calculations.cut3d(f, xmin, xmax, ymin, ymax, zmin, zmax, "CellID"),dtype=np.int64)

print("Found box with shape of ", cellids_box.shape)
cellids_box = np.unique(cellids_box)
print("Total ", np.size(cellids_box), "unique cells")

coords_out = f.get_cell_coordinates(cellids_box)
vdfsamples_out = []
B_out = fb.read_variable("vg_b_vol", cellids=cellids_box)
E_out = fb.read_variable("vg_e_vol", cellids=cellids_box)
rho_out = fb.read_variable("proton/vg_rho", cellids=cellids_box)


for cellid in cellids_box[:]:
   # cellid = f.get_cellid_with_vdf(probept)

   print(cellid)
   

   pt.plot.plot_vdf(vlsvobj=f, cellids=[cellid], xz=1, outputdir="./", title="Projected VDF", fmin=3e-12, slicethick=0)
   pt.plot.plot_vdf(vlsvobj=f, cellids=[cellid], xy=1, outputdir="./", title="Projected VDF", fmin=3e-12, slicethick=0)

   distr, edges = f.read_velocity_distribution_dense(cellid)
   
   skip = False
   if True:
      downsample=False

      if downsample:
         distr = distr[::2,::2,::2]
         ex = edges[0][::2]
         ey = edges[1][::2]
         ez = edges[2][::2]
      else:
         ex = edges[0][::1]
         ey = edges[1][::1]
         ez = edges[2][::1]


   if not skip:
      EX, EY, EZ = np.meshgrid(ex,ey,ez, indexing="ij")
      pc = plt.pcolormesh(EX[:,ey.size//2,:],EZ[:,ey.size//2,:], np.squeeze(np.sum(distr, axis=1)), norm="log", cmap="hot_desaturated")
      plt.colorbar()
      pc.axes.set_aspect('equal')
      plt.title("Slice of the dense VDF distribution around $v_y$ midpoint")
      plt.xlabel("$v_x / \mathrm{m}\,\mathrm{s}^{-1}$")
      plt.ylabel("$v_z / \mathrm{m}\,\mathrm{s}^{-1}$")
      plt.savefig(str(cellid)+"_dense_distribution_proj.png", dpi=300)
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

   if not skip:
      plt.hist(distr.reshape((-1)), bins=30, log=True)
      plt.yscale("log")
      plt.title("Histogram of v-space sample weights")
      plt.xlabel("$f(v)$")
      plt.savefig(str(cellid)+"_vdf-hist.png")
      plt.close()


   choice = np.random.choice(lX.size, size=samples, p =ldistr/np.sum(ldistr))
   particle_vs = np.random.random_sample(size=(samples,3))

   ldX = dX.reshape((-1))
   ldY = dY.reshape((-1))
   ldZ = dZ.reshape((-1))

   particle_vs *= np.stack((ldX[choice],ldZ[choice],ldY[choice]),axis=1) # scale by v-cell size
   particle_vs += np.stack((lX[choice],lZ[choice],lY[choice]), axis=1) # offset by chosen bin edges

   particle_vs = np.stack((particle_vs[:,0], particle_vs[:,2],particle_vs[:,1]),axis=1)

   vdfsamples_out.append(particle_vs)

   if not skip:

      nbins=int(np.sqrt(samples)//30)
      h,_,_,_ = plt.hist2d(particle_vs[:,0], particle_vs[:,1], bins=nbins, norm="log", cmap="hot_desaturated")

      plt.clim(vmin=np.median(h)/3)
      plt.axis("equal")
      plt.colorbar(label="counts")
      plt.xlabel("$v_x / \mathrm{m}\,\mathrm{s}^{-1}$")
      plt.ylabel("$v_y / \mathrm{m}\,\mathrm{s}^{-1}$")
      plt.title(f"Counts in $(v_x,v_y)$ ({samples} samples, {nbins}$^2$ bins)")
      plt.savefig(str(cellid)+"_vxvy_VDF_sampled_histogram.png")

      plt.close()

      h,_,_,_ = plt.hist2d(particle_vs[:,0], particle_vs[:,2], bins=nbins, norm="log", cmap="hot_desaturated")

      plt.clim(vmin=np.median(h)/3)
      plt.axis("equal")
      plt.colorbar(label="counts")
      plt.xlabel("$v_x / \mathrm{m}\,\mathrm{s}^{-1}$")
      plt.ylabel("$v_z / \mathrm{m}\,\mathrm{s}^{-1}$")
      plt.title(f"Counts in $(v_x,v_z)$ ({samples} samples, {nbins}$^2$ bins)")
      plt.savefig(str(cellid)+"_vxvz_VDF_sampled_histogram.png")



np.savez("boxed_vdf_data_1610_BBF", B=B_out,n=rho_out, E=E_out,coordinates=coords_out,cellids=cellids_box,VDF_samples=np.array(vdfsamples_out))

   #np.savez(str(cellid)+"_VDF_example",allow_pickle=False, dense_array=distr, dense_array_edges_x=ex, dense_array_edges_y=ey, dense_array_edges_z=ez, sampled_particles=particle_vs)

   #np.savez(str(cellid)+"_VDF_example_particles",allow_pickle=False, sampled_particles=particle_vs)

