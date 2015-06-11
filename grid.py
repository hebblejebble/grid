import numpy as np

class Grid(np.ndarray):
	def __new__(cls, arr):
		return np.asarray(arr).view(cls)
	
	def match(self, arr):
		if (self.shape != arr.shape): return False
		return (self == arr).all()
	
	def __getitem__(self, i):
		print i
		return super(Grid, self).__getitem__(i)
	
	def __getslice__(self, i, j):
		return super(Grid, self).__getslice__(i, j)
