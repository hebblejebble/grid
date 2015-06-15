import unittest
import numpy as np
from grid import Grid

class TestGrid1D(unittest.TestCase):
	def setUp(self):
		self.one_d = np.array( range(10) )
		self.g = Grid( self.one_d.copy() )
	
	def test_match_true(self):
		self.assertTrue( self.g.match( self.one_d ) )
		
	def test_match_wrong_size(self):
		short_arr = self.one_d[:-1]
		self.assertFalse( self.g.match(short_arr) )
	
	def test_match_wrong_value(self):
		diff_arr = self.one_d.copy()
		diff_arr[0] = -1
		self.assertFalse( self.g.match(diff_arr) )
	
	def test_getitem_pos_int(self):
		x1 = self.one_d[0]
		x2 = self.g[0]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_neg_int(self):
		x1 = self.one_d[-1]
		x2 = self.g[-1]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_pos_float(self):
		x1 = self.one_d[1]
		x2 = self.g[1.0]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_neg_float(self):
		x1 = self.one_d[-2]
		x2 = self.g[-2.0]
		self.assertTrue( x1 == x2 )
	
	def test_getslice_pos_int(self):
		aslice = self.one_d[0:2]
		gslice = self.g[0:2]
		self.assertTrue( gslice.match(aslice) )
	
	def test_getslice_neg_int(self):
		aslice = self.one_d[1:-1]
		gslice = self.g[1:-1]
		self.assertTrue( gslice.match(aslice) )
	
	def test_getslice_pos_float(self):
		aslice = self.one_d[0:2]
		gslice = self.g[0.0:2.0]
		self.assertTrue( gslice.match(aslice) )
	
	def test_getslice_neg_float(self):
		aslice = self.one_d[1:-1]
		gslice = self.g[1.0:-1.0]
		self.assertTrue( gslice.match(aslice) )
	
	def test_default_edges(self):
		edges = [[i-.5 for i in range( len(self.g)+1)]]
		self.assertTrue( self.g.get_edges() == edges )
	
	def test_manual_edges(self):
		man_g = Grid(np.array([1,2,3]), edges=[1.0,-1.0,-3.0,-5.0])
		self.assertTrue( man_g.get_edges() == [[1.0,-1.0,-3.0,-5.0]] )
	
	def test_wrong_size_edges(self):
		create_fail = False
		try:
			man_g = Grid(np.array([1,2,3]), edges=[-1.0,1.0,3.0])
		except:
			create_fail = True
		self.assertTrue( create_fail )
	
	def test_not_monotonic_edges(self):
		create_fail = False
		try:
			man_g = Grid(np.array([1,2,3]), edges=[1.0,-1.0,3.0,5.0])
		except:
			create_fail = True
		self.assertTrue( create_fail )


class TestGrid3D(unittest.TestCase):
	def setUp(self):
		self.three_d = np.array( range(125) ).reshape(5,5,5)
		self.g = Grid( self.three_d.copy() )
	
	def test_match_true(self):
		self.assertTrue( self.g.match( self.three_d ) )
		
	def test_match_wrong_size(self):
		short_arr = self.three_d[:-1,:-1,:-1]
		self.assertFalse( self.g.match(short_arr) )
	
	def test_match_wrong_value(self):
		diff_arr = self.three_d.copy()
		diff_arr[0,0,0] = -1
		self.assertFalse( self.g.match(diff_arr) )
	
	def test_getitem_pos_int(self):
		x1 = self.three_d[0,0,0]
		x2 = self.g[0,0,0]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_neg_int(self):
		x1 = self.three_d[-1,-1,-1]
		x2 = self.g[-1,-1,-1]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_pos_float(self):
		x1 = self.three_d[0,0,0]
		x2 = self.g[0.0,0.0,0.0]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_neg_float(self):
		x1 = self.three_d[-1,-1,-1]
		x2 = self.g[-1.0,-1.0,-1.0]
		self.assertTrue( x1 == x2 )
	
	def test_getslice_pos_int(self):
		aslice = self.three_d[0:2,0:2,0:2]
		gslice = self.g[0:2,0:2,0:2]
		self.assertTrue( gslice.match(aslice) )
	
	def test_getslice_neg_int(self):
		aslice = self.three_d[1:-1,1:-1,1:-1]
		gslice = self.g[1:-1,1:-1,1:-1]
		self.assertTrue( gslice.match(aslice) )
	
	def test_getslice_pos_float(self):
		aslice = self.three_d[0:2,0:2,0:2]
		gslice = self.g[0.0:2.0,0.0:2.0,0.0:2.0]
		self.assertTrue( gslice.match(aslice) )
	
	def test_getslice_neg_float(self):
		aslice = self.three_d[1:-1,1:-1,1:-1]
		gslice = self.g[1.0:-1.0,1.0:-1.0,1.0:-1.0]
		self.assertTrue( gslice.match(aslice) )
	
	def test_default_edges(self):
		edges = [[i-.5 for i in range(e+1)] for e in self.g.shape]
		self.assertTrue( self.g.get_edges() == edges )
	
	def test_manual_edges(self):
		man_edges = [range(6),range(6,0,-1),range(0,-6,-1)]
		man_g = Grid(np.array(self.three_d), edges=man_edges)
		self.assertTrue( man_g.get_edges() == man_edges )
	
	def test_wrong_size_edges(self):
		create_fail = False
		man_edges = [range(6),range(7,0,-1),range(0,-6,-1)]
		try:
			man_g = Grid(np.array(self.three_d), edges=man_edges)
		except:
			create_fail = True
		self.assertTrue( create_fail )
	
	def test_wrong_dim_edges(self):
		create_fail = False
		man_edges = [range(6),range(6,0,-1),range(0,-6,-1),range(6)]
		try:
			man_g = Grid(np.array(self.three_d), edges=man_edges)
		except:
			create_fail = True
		self.assertTrue( create_fail )

if __name__ == '__main__':
	unittest.main()
