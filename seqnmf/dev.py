import seaborn as sns
from matplotlib import pyplot as plt
import os
from scipy.io import loadmat
from seqnmf import seqnmf, seqnmf_old
from helpers import plot

data = loadmat(os.path.join('data', 'MackeviciusData.mat'))


plt.figure()
sns.heatmap(data['NEURAL'], cmap='gray_r')
plt.show()

W, H, power, loadings, costs = seqnmf(data['NEURAL'])

h = plot(W, H)
h.suptitle('Total Power Explained: {:.3f}'.format(power))
h.show()

W_old, H_old, cost_old, loadings_old, power_old = seqnmf_old(data['NEURAL'])

h = plot(W_old, H_old)
h.suptitle('Total Power Explained: {:.3f}'.format(power_old))
h.show()
