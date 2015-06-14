import numpy as np

class Grid(np.ndarray):
	def __new__(cls, arr, edges=None):
		x = np.asarray(arr).view(cls)
		
		# set edges
		if edges == None:
			x.edges = x.default_edges()
		elif x.validate_edges(edges):
			x.edges = x.cast_edges(edges)
		else:
			raise ValueError("invalid edges argument.")
		
		return x
	
	def match(self, arr):
		if (self.shape != arr.shape): return False
		return (self == arr).all()
	
	def __getitem__(self, i):
		x = super(Grid, self).__getitem__(i)
		return x
	
	# getslice is only called in 1D, integer slices (otherwise tricky getitem call used)
	def __getslice__(self, i, j):
		x = super(Grid, self).__getslice__(i, j)
		return x
	
	# returns the default set of edges for the grid 
	def default_edges(self):
		return [ [ i - 0.5 for i in range(dsize+1) ] for dsize in self.shape ]
	
	# tests that edge set is valid for the grid (monotonic & correct size)
	def validate_edges(self, edges):
		edges = self.cast_edges(edges)
		#dimension test
		if [ len(e)-1 for e in edges ] != list(self.shape):
			return False
		#monotonic test
		for e in edges:
			dx = np.diff(e)
			if not (dx >= 0).all() and not (dx <= 0).all():
				return False
		#tests passed
		return True
	
	# casts edge set into the correct form (list of lists)
	def cast_edges(self, edges):
		try:
			return [ list(edge) for edge in edges ]
		except:
			return [ list(edges) ]
