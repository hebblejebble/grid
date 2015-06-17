import unittest
import numpy as np
from grid import Grid

class TestGrid1D(unittest.TestCase):
	def setUp(self):
		self.one_d = np.array( range(10) )
		self.g = Grid( self.one_d.copy(), edges = range(-5,6,1) )
	
	def test_match_true_1d(self):
		self.assertTrue( self.g.match( self.one_d ) )
		
	def test_match_wrong_size_1d(self):
		short_arr = self.one_d[:-1]
		self.assertFalse( self.g.match(short_arr) )
	
	def test_match_wrong_value_1d(self):
		diff_arr = self.one_d.copy()
		diff_arr[0] = -1
		self.assertFalse( self.g.match(diff_arr) )
	
	def test_getitem_pos_int_1d(self):
		x1 = self.one_d[0]
		x2 = self.g[0]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_neg_int_1d(self):
		x1 = self.one_d[-1]
		x2 = self.g[-1]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_float_1d(self):
		x1 = self.one_d[6]
		x2 = self.g[1.5]
		self.assertTrue( x1 == x2 )
	
	def test_getslice_pos_int_1d(self):
		aslice = self.one_d[0:2]
		gslice = self.g[0:2]
		self.assertTrue( gslice.match(aslice) )
		self.assertTrue( gslice.get_edges() == [[-5,-4,-3]] )
	
	def test_getslice_neg_int_1d(self):
		aslice = self.one_d[1:-1]
		gslice = self.g[1:-1]
		self.assertTrue( gslice.match(aslice) )
		self.assertTrue( gslice.get_edges() == [range(-4,5,1)] )
	
	def test_getslice_float_1d(self):
		aslice = self.one_d[4:8]
		gslice = self.g[-0.5:2.5]
		self.assertTrue( gslice.match(aslice) )
		self.assertTrue( gslice.get_edges() == [range(-1,4,1)] )
	
	def test_default_edges_1d(self):
		edges = [[i-.5 for i in range( len(self.g)+1)]]
		grd = Grid( self.one_d.copy() )
		self.assertTrue( grd.get_edges() == edges )
	
	def test_manual_edges_1d(self):
		man_g = Grid(np.array([1,2,3]), edges=[1.0,-1.0,-3.0,-5.0])
		self.assertTrue( man_g.get_edges() == [[1.0,-1.0,-3.0,-5.0]] )
	
	def test_wrong_size_edges_1d(self):
		create_fail = False
		try:
			man_g = Grid(np.array([1,2,3]), edges=[-1.0,1.0,3.0])
		except:
			create_fail = True
		self.assertTrue( create_fail )
	
	def test_not_monotonic_edges_1d(self):
		create_fail = False
		try:
			man_g = Grid(np.array([1,2,3]), edges=[1.0,-1.0,3.0,5.0])
		except:
			create_fail = True
		self.assertTrue( create_fail )


class TestGrid3D(unittest.TestCase):
	def setUp(self):
		edge_set = [range(-6,0,1),range(3,-3,-1),range(2,8,1)]
		self.three_d = np.array( range(125) ).reshape(5,5,5)
		self.g = Grid( self.three_d.copy(), edges=edge_set )
	
	def test_match_true_3d(self):
		self.assertTrue( self.g.match( self.three_d ) )
		
	def test_match_wrong_size_3d(self):
		short_arr = self.three_d[:-1,:-1,:-1]
		self.assertFalse( self.g.match(short_arr) )
	
	def test_match_wrong_value_3d(self):
		diff_arr = self.three_d.copy()
		diff_arr[0,0,0] = -1
		self.assertFalse( self.g.match(diff_arr) )
	
	def test_getitem_pos_int_3d(self):
		x1 = self.three_d[0,0,0]
		x2 = self.g[0,0,0]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_neg_int_3d(self):
		x1 = self.three_d[-1,-1,-1]
		x2 = self.g[-1,-1,-1]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_float_3d(self):
		x1 = self.three_d[1,2,3]
		x2 = self.g[-4.5,0.5,5.5]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_mixed_3d(self):
		x1 = self.three_d[1,4,-1]
		x2 = self.g[1,-1.5,-1]
		self.assertTrue( x1 == x2 )
	
	def test_getslice_pos_int_3d(self):
		edge_set = [[-6,-5,-4],[3,2,1],[2,3,4]]
		aslice = self.three_d[0:2,0:2,0:2]
		gslice = self.g[0:2,0:2,0:2]
		self.assertTrue( gslice.match(aslice) )
		self.assertTrue( gslice.get_edges() == edge_set )
	
	def test_getslice_neg_int_3d(self):
		edge_set = [range(-5,-1,1),range(2,-2,-1),range(3,7,1)]
		aslice = self.three_d[1:-1,1:-1,1:-1]
		gslice = self.g[1:-1,1:-1,1:-1]
		self.assertTrue( gslice.match(aslice) )
		self.assertTrue( gslice.get_edges() == edge_set )
	
	def test_getslice_float_3d(self):
		edge_set = [range(-6,-2,1),range(3,-1,-1),range(2,6,1)]
		aslice = self.three_d[0:3,0:3,0:3]
		gslice = self.g[-5.5:-3.5,2.5:0.5,2.5:4.5]
		self.assertTrue( gslice.match(aslice) )
		self.assertTrue( gslice.get_edges() == edge_set )
	
	def test_getslice_mixed_3d(self):
		edge_set = [range(-5,-2,1),range(2,-1,-1),range(4,7,1)]
		aslice = self.three_d[1:3,1:3,-3:-1]
		gslice = self.g[1:3,1.5:0.5,-3:-1]
		self.assertTrue( gslice.match(aslice) )
		self.assertTrue( gslice.get_edges() == edge_set )
	
	def test_seperate_slice_3d(self):
		gslice = self.g[0][0][0:3]
		self.assertTrue( list(gslice) == [0,1,2] )
	
	def test_default_edges_3d(self):
		grd = Grid( self.three_d.copy() )
		edges = [[i-.5 for i in range(e+1)] for e in self.g.shape]
		self.assertTrue( grd.get_edges() == edges )
	
	def test_manual_edges_3d(self):
		man_edges = [range(6),range(6,0,-1),range(0,-6,-1)]
		man_g = Grid(np.array(self.three_d), edges=man_edges)
		self.assertTrue( man_g.get_edges() == man_edges )
	
	def test_wrong_size_edges_3d(self):
		create_fail = False
		man_edges = [range(6),range(7,0,-1),range(0,-6,-1)]
		try:
			man_g = Grid(np.array(self.three_d), edges=man_edges)
		except:
			create_fail = True
		self.assertTrue( create_fail )
	
	def test_wrong_dim_edges_3d(self):
		create_fail = False
		man_edges = [range(6),range(6,0,-1),range(0,-6,-1),range(6)]
		try:
			man_g = Grid(np.array(self.three_d), edges=man_edges)
		except:
			create_fail = True
		self.assertTrue( create_fail )

if __name__ == '__main__':
	unittest.main()
