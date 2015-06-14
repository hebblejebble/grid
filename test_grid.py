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
	
	def test_edges(self):
		edges = [[i-.5 for i in range( len(self.g)+1)]]
		self.assertTrue( self.g.edges == edges )


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
	
	def test_edges(self):
		#self.g.shape = (5,5,5)
		edges = [[i-.5 for i in range(e+1)] for e in self.g.shape]
		self.assertTrue( self.g.edges == edges )

if __name__ == '__main__':
	unittest.main()
