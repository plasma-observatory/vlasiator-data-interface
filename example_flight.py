import analysator as pt # pt is historically used for analysator
import numpy as np
import matplotlib.pyplot as plt

# Load the trajectory file as given by one of the toolsets
flight = np.loadtxt("trajectories/PO_constellation_flight.txt", delimiter=',')

# trajectory format:
# time_id, spacecraft_id, x, y, z
# handling here leverages that this particular trajectory file has one line per
# time_id, so seven lines per time_id, and that the SC are ordered by id
# -> stride of seven rows will get a timeseries of the positions of a single spacecraft

# fetch all the points to be interpolated out of a snapshot
pts = flight[:,-3:]

# get time indices
t = flight[::7,0]

# Initialize a VLSV reader object that can be queried for e.g. interpolated data
f = pt.vlsvfile.VlsvReader("data/bulk1.0001612.vlsv")

# read in trilinearly-interpolated data on the points; this does interpolate at refinement
# interfaces as well
B = f.read_interpolated_variable("vg_b_vol", pts)

fig, axs = plt.subplots(2)

#plot outer tetra Bz, spacecraft by spacecraft (indexed by po)
for po in [0,1,2,3]:
   axs[0].plot(t, B[po::7,2], label="outer:{%d}".format(po))

#plot inner tetra Bz
for po in [0,4,5,6]:
   axs[1].plot(t, B[po::7,2], label="inner:{%d}".format(po))

axs[0].legend()
axs[1].legend()


plt.savefig("test.png",dpi=300)
