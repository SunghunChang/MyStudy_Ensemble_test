import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import time

day = time.localtime()
day_string = "%04d-%02d-%02d_%02d-%02d-%02d" % (day.tm_year, day.tm_mon, day.tm_mday, day.tm_hour, day.tm_min, day.tm_sec)
vmin_val = -0.6
vmax_val = 0.6

def PlotWeighNbias(weight_layer_1, weight_layer_2, weight_layer_3, bias_layer_1, bias_layer_2, bias_layer_3, k, plot_show):
	#plt.figure(figsize=(8.0, 5.0))

	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
	ax1.set_title('1st Layer Weight [Model #{0:s}]'.format(str(k + 1)))
	#Here vmin and vmax is work main matrix and colorbar both

	im1 = ax1.imshow(weight_layer_1, vmin=vmin_val, vmax=vmax_val, cmap=cm.coolwarm) #'gray cm.coolwarm)  # , aspect='auto') # interpolation='nearest'
	divider1 = make_axes_locatable(ax1)
	cax1 = divider1.append_axes("right", size="5%", pad=0.05)
	# Multiple locator of colorbar : even spacing
	# Ticks list : manually ticked (not force min,max)
	cbar1 = plt.colorbar(im1, cax=cax1, ticks=MultipleLocator(0.2), format="%.1f") # , ticks=[-0.2,0,0.2], format="%.1f")
	#cbar1.set_clim(vmin=vmin_val, vmax=vmax_val) # Color range of main matrix
	#cbar1.ax.set_xlim(-1,1)
	#cbar1.set_ticks([-10,0,-0.2,0.0,10.0])  not works to limit


	ax2.set_title('2st Layer Weight [Model #{0:s}]'.format(str(k + 1)))
	im2 = ax2.imshow(weight_layer_2, vmin=vmin_val, vmax=vmax_val, cmap=cm.coolwarm, aspect='auto')
	divider2 = make_axes_locatable(ax2)
	cax2 = divider2.append_axes("right", size="5%", pad=0.05)
	cbar2 = plt.colorbar(im2, cax=cax2, ticks=MultipleLocator(0.2), format="%.1f")

	ax3.set_title('3rd Layer Weight [Model #{0:s}]'.format(str(k + 1)))
	im3 = ax3.imshow(weight_layer_3, vmin=vmin_val, vmax=vmax_val, cmap=cm.coolwarm, aspect='auto')
	divider3 = make_axes_locatable(ax3)
	cax3 = divider3.append_axes("right", size="5%", pad=0.05)
	cbar3 = plt.colorbar(im3, cax=cax3, ticks=MultipleLocator(0.2), format="%.1f")

	ax4.set_title('1st Layer Bias [Model #{0:s}]'.format(str(k + 1)))
	ax4.set_xlabel("per NODE")
	ax4.set_ylabel("bias value")
	# ax4.axes.get_xaxis().set_ticklabels([])  # It remains grid line
	# ax4.axes.get_xaxis().set_ticklabels(np.arange(1, len(bias_layer_1) + 1, 1))
	ax4.set_ylim(-0.2, 0.2)
	ax4.axes.get_yaxis().set_ticklabels(np.arange(-0.5, 0.6, 0.1))
	ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax4.plot(bias_layer_1, color='b', marker=".", label="Model " + str(k + 1) + " - Layer 1", linestyle="--")
	ax4.legend(loc="best")
	ax4.grid(linestyle=':', linewidth=1)

	ax5.set_title('2nd Layer Bias [Model #{0:s}]'.format(str(k + 1)))
	ax5.set_xlabel("per NODE")
	ax5.set_ylabel("bias value")
	# ax5.axes.get_xaxis().set_ticklabels([])  # It remains grid line
	# ax5.axes.get_xaxis().set_ticklabels(np.arange(1, len(bias_layer_2) + 1, 1))
	# ax5.axes.get_yaxis().set_ticklabels(np.arange(-0.5, 0.6, 0.1))
	ax5.set_ylim(-0.2, 0.2)
	ax5.yaxis.set_ticklabels(np.arange(-0.5, 0.6, 0.1))
	ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax5.plot(bias_layer_2, color='b', marker=".", label="Model " + str(k + 1) + " - Layer 2", linestyle="--")
	ax5.legend(loc="best")
	ax5.grid(linestyle=':', linewidth=1)

	ax6.set_title('3rd Layer Bias [Model #{0:s}]'.format(str(k + 1)))
	ax6.set_xlabel("per NODE")
	ax6.set_ylabel("bias value")
	# ax6.axes.get_xaxis().set_ticklabels([])  # It remains grid line
	# ax6.axes.get_xaxis().set_ticklabels(np.arange(1, len(bias_layer_3) + 1, 1))
	ax6.set_ylim(-0.2, 0.2)
	ax6.axes.get_yaxis().set_ticklabels(np.arange(-0.5, 0.6, 0.1))
	ax6.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax6.plot(bias_layer_3, color='b', marker=".", label="Model " + str(k + 1) + " - Layer 3", linestyle="--")
	ax6.legend(loc="best")
	ax6.grid(linestyle=':', linewidth=1)

	#fig.tight_layout()

#   # Maximize Figure Window
#	manager = plt.get_current_fig_manager()
#	manager.resize(*manager.window.maxsize())
	figure = plt.gcf()
	figure.set_size_inches(24, 16)

	plt.savefig("./01_Run_Weight_n_Bias/Model_{0:03d}_{1}".format(int(k+1),day_string), # figsize=(24, 16),
	            dpi=150, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None,
	            transparent=True, bbox_inches='tight', pad_inches=0.1, frameon=None)
	if plot_show:
		plt.show()