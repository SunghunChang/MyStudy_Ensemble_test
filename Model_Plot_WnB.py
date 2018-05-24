import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def PlotWeighNbias(weight_layer_1, weight_layer_2, weight_layer_3, bias_layer_1, bias_layer_2, bias_layer_3, k):
	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

	ax1.set_title('1st Layer Weight [Model #{0:s}]'.format(str(k + 1)))
	im1 = ax1.imshow(weight_layer_1, cmap='gray')  # , aspect='auto')
	divider1 = make_axes_locatable(ax1)
	cax1 = divider1.append_axes("right", size="5%", pad=0.05)
	cbar1 = plt.colorbar(im1, cax=cax1, ticks=MultipleLocator(0.2), format="%.1f")

	ax2.set_title('2st Layer Weight [Model #{0:s}]'.format(str(k + 1)))
	im2 = ax2.imshow(weight_layer_2, cmap='gray', aspect='auto')
	divider2 = make_axes_locatable(ax2)
	cax2 = divider2.append_axes("right", size="5%", pad=0.05)
	cbar2 = plt.colorbar(im2, cax=cax2, ticks=MultipleLocator(0.2), format="%.1f")

	ax3.set_title('3rd Layer Weight [Model #{0:s}]'.format(str(k + 1)))
	im3 = ax3.imshow(weight_layer_3, cmap='gray', aspect='auto')
	divider3 = make_axes_locatable(ax3)
	cax3 = divider3.append_axes("right", size="5%", pad=0.05)
	cbar3 = plt.colorbar(im3, cax=cax3, ticks=MultipleLocator(0.2), format="%.1f")

	ax4.set_title('1st Layer Bias [Model #{0:s}]'.format(str(k + 1)))
	ax4.set_xlabel("per NODE")
	ax4.set_ylabel("bias value")
	# ax4.axes.get_xaxis().set_ticklabels([])  # It remains grid line
	# ax4.axes.get_xaxis().set_ticklabels(np.arange(1, len(bias_layer_1) + 1, 1))
	ax4.set_ylim(-0.6, 0.6)
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
	ax5.set_ylim(-0.6, 0.6)
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
	ax6.set_ylim(-0.6, 0.6)
	ax6.axes.get_yaxis().set_ticklabels(np.arange(-0.5, 0.6, 0.1))
	ax6.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax6.plot(bias_layer_3, color='b', marker=".", label="Model " + str(k + 1) + " - Layer 3", linestyle="--")
	ax6.legend(loc="best")
	ax6.grid(linestyle=':', linewidth=1)

	#    fig.tight_layout()
	plt.show()