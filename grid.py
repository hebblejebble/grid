import numpy as np
from bisect import bisect

class Grid(np.ndarray):
	def __new__(cls, arr, edges=None):
		x = np.asarray(arr).view(cls)
		x.set_edges(edges)
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
	
	# return index position for edge set of given dimension
	def __grid_lookup__(self, dim, val):
		edge = self._edges[dim]
		reverse = self._edge_reverse[dim]
		if val < edge[0] or val > edge[-1]:
			# val not in grid
			pass
		index_out = bisect(edge, val) - 1
		if reverse:
			index_out = len(edge) - index_out
		return index_out
	
	# returns the default set of edges for the grid 
	def __default_edges__(self):
		self._edges = [ [ i - 0.5 for i in range(dsize+1) ] for dsize in self.shape ]
		self._edge_reverse = [ False for i in range(self.ndim) ]
		return None
	
	# tests that edge set is valid for the grid (monotonic & correct size)
	def validate_edges(self, edges):
		#type test (uses duck typeing)
		try:
			edges = [ [i for i in e] for e in edges ]
		except:
			return False
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
	
	# sets the edges of the grid, is given None, generates default edges
	def set_edges(self, edges=None):
		if edges == None:
			self.__default_edges__()
		
		elif self.validate_edges(edges):
			self._edges = [ [i for i in e] for e in edges ]
			self._edge_reverse = []
			for edge_set in self._edges:
				if edge_set[0] > edge_set[1]:
					edge_set.reverse()
					self._edge_reverse.append(True)
				else:
					self._edge_reverse.append(False)
					
		elif self.validate_edges([edges]):
			self._edges = [[ i for i in edges ]]
			if edges[0] > edges[1]:
				self._edges[0].reverse()
				self._edge_reverse = [True]
			else:
				self._edge_reverse = [False]
				
		else:
			raise ValueError("invalid edge set")
		
	def get_edges(self):
		out_edges = [ [i for i in e] for e in self._edges ]
		for i in range(len(out_edges)):
			if self._edge_reverse[i]: out_edges[i].reverse()
		return out_edges
