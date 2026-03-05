from .seqnmf import seqnmf
from .helpers import plot

from scipy.io import loadmat
import os
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('seqnmf', 'data/')
example_data = loadmat(os.path.join(DATA_PATH, 'MackeviciusData.mat'))['NEURAL']

del DATA_PATH
del os
del loadmat
del pkg_resources