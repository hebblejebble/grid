import numpy as np

class Grid(np.ndarray):
	def __new__(cls, arr):
		return np.asarray(arr).view(cls)
	
	def full_match(self, arr):
		if (self.shape != arr.shape): return False
		return (self == arr).all()
