import numpy as np
import matplotlib.pyplot as plt

data = np.load("boxed_vdf_data_FHA_1610_BBF.npz")

print(*[k for k in data.keys()])
print(len(data["cellids"]), "spatial cells available")

RE=6371e3

xs,ys,zs = data["coordinates"][:,0]/RE,data["coordinates"][:,1]/RE,data["coordinates"][:,2]/RE # m to RE
Bxs,Bys,Bzs = data["B"][:,0],data["B"][:,1],data["B"][:,2] # T
Exs,Eys,Ezs = data["E"][:,0],data["E"][:,1],data["E"][:,2] # V/m

ax = plt.figure(figsize=[8,8]).add_subplot(projection='3d')

pc = ax.scatter(xs, ys, zs, c=np.log10(data['n']), label="$n_\mathrm{p}$")
plt.colorbar(pc,label=r"$\log_{10}(n_\mathrm{p})$")
ax.set_xlabel("x/Re")
ax.set_ylabel("y/Re")
ax.set_zlabel("z/Re")


ax.quiver(xs,ys,zs,Bxs,Bys,Bzs,length=3e6, label="B", color="green")
ax.quiver(xs,ys,zs,Exs,Eys,Ezs,length=3,color="red", label="E")
plt.legend()

plt.savefig("environs.png")
plt.close()

# Let's find the coordinates closest to X = -9RE, Y=-3.8, Z=0.5

query = np.array([-9,-3.8, 0.5])
dists = np.linalg.norm(data["coordinates"]/RE - query[np.newaxis,:], axis=1)
hit = np.argmin(dists)
print("Taking example at index ", hit, ", coordinates=", data["coordinates"][hit,:]/RE)

vsample = data["VDF_samples"][hit]/1e6 #1000km/s

ax = plt.figure(figsize=[8,8]).add_subplot(projection='3d')

vmean = np.mean(vsample,axis=0)


ax.quiver(vmean[0],vmean[1],vmean[2],Bxs[hit],Bys[hit],Bzs[hit], length=1e8, color="green", label="B")
ax.quiver(vmean[0],vmean[1],vmean[2],Exs[hit],Eys[hit],Ezs[hit], length=2e2, color="red", label="E")

ax.scatter(vsample[:,0],vsample[:,1],vsample[:,2], alpha=0.005)
ax.set_xlabel("vx/1000km/s")
ax.set_ylabel("vy/1000km/s")
ax.set_zlabel("vz/1000km/s")
title = "VDF sample at ({:.1f},{:.1f},{:.1f})".format(*data["coordinates"][hit,:]/RE)
title = title+"$R_\mathrm{E}$, $n_\mathrm{p}=$"
title = title+"{:.1f}".format(data["n"][hit])+"$\mathrm{m}^{-3}$"
ax.set_title(title)
plt.legend()

plt.savefig("vdf.png")
plt.close()
