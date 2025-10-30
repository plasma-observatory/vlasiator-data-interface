import analysator as pt # pt is historically used for analysator
import numpy as np
import matplotlib.pyplot as plt

flight = np.loadtxt("trajectories/PO_constellation_flight.txt", delimiter=',')

pts = flight[:,-3:]
t = flight[::7,0]

f = pt.vlsvfile.VlsvReader("data/bulk1.0001612.vlsv")


B = f.read_interpolated_variable("vg_b_vol", pts)

fig, axs = plt.subplots(2)

#plot outer tetra Bz
for po in [0,1,2,3]:
   axs[0].plot(t, B[po::7,2], label="outer:{%d}".format(po))

#plot inner tetra Bz
for po in [0,4,5,6]:
   axs[1].plot(t, B[po::7,2], label="inner:{%d}".format(po))

axs[0].legend()
axs[1].legend()

plt.savefig("test.png",dpi=300)