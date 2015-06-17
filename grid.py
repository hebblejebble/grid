import numpy as np
from bisect import bisect

class Grid(np.ndarray):
	def __new__(cls, arr, edges=None):
		x = np.asarray(arr).view(cls)
		x.set_edges(edges)
		return x
	
	# used for test suite. compares every value in self with a ndarray
	def match(self, arr):
		if (self.shape != arr.shape): return False
		return (self == arr).all()
	
	# intercepts ndarray.__getitem__(), floats use grid edges, ints passed through
	# returned grid has sliced edges.
	# NOTE: grid float slice will include last element. int slice will not.
	def __getitem__(self, arg):
		if type(arg) is not tuple:
			arg = [arg]
		else:
			arg = list(arg)
		edge_set = self.get_edges()
		for i in range(len(arg)):
			if type(arg[i]) is slice:
				start = self.__grid_lookup__(i, arg[i].start)
				stop  = self.__grid_lookup__(i, arg[i].stop)
				if type(arg[i].stop) is not int:
					stop += 1
				arg[i] = slice(start, stop, None)
				edge_set[i] = edge_set[i][start:(stop+1)]
			else:
				arg[i] = self.__grid_lookup__(i, arg[i])
				edge_set[i] = None
		while None in edge_set:
			edge_set.remove(None)
		x = super(Grid, self).__getitem__( tuple(arg) )
		if type(x) == Grid:
			x.set_edges(edge_set)
		return x
	
	# getslice is only called in 1D, integer slices (otherwise tricky getitem call used)
	def __getslice__(self, i, j):
		x = super(Grid, self).__getslice__(i, j)
		edge_set = [ e[i:(j+1)] for e in self.get_edges() ]
		x.set_edges(edge_set)
		return x
	
	# return index position for edge set of given dimension
	def __grid_lookup__(self, dim, val):
		edge = self._edges[dim]
		reverse = self._edge_reverse[dim]
		
		# if position is an int, use standard lookup over grid lookup
		if type(val) == int:
			if val < 0:
				return (len(edge)-1) + val
			return val
			
		if val < edge[0] or val > edge[-1]:
			# val not in grid
			raise ValueError(str(val) + " outside of edge set")
		index_out = bisect(edge, val)
		if reverse:
			index_out = (len(edge)-1) - index_out
		else:
			index_out -= 1
		return index_out
	
	# returns the default set of edges for the grid 
	def __default_edges__(self):
		self._edges = [ [ i - 0.5 for i in range(dsize+1) ] for dsize in self.shape ]
		self._edge_reverse = [ False for i in range(self.ndim) ]
		return None
	
	def __insert_edges__(self, edges):
		self._edges = [ [i for i in e] for e in edges ]
		self._edge_reverse = []
		for edge_set in self._edges:
			if edge_set[0] > edge_set[1]:
				edge_set.reverse()
				self._edge_reverse.append(True)
			else:
				self._edge_reverse.append(False)
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
			self.__insert_edges__(edges)
		# lets 1d grids define edges with one list ([...] instead of [[...]])		
		elif self.validate_edges([edges]):
			self.__insert_edges__([edges])
		else:
			raise ValueError("invalid edge set")
		return None
	
	# returns the edge set for the grid
	def get_edges(self):
		out_edges = [ [i for i in e] for e in self._edges ]
		for i in range(len(out_edges)):
			if self._edge_reverse[i]: out_edges[i].reverse()
		return out_edges
