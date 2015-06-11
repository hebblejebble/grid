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
	
	def test_getslice_pos_int(self):
		aslice = self.one_d[0:2]
		gslice = self.g[0:2]
		self.assertTrue( gslice.match(aslice) )
	
	def test_getslice_neg_int(self):
		aslice = self.one_d[1:-1]
		gslice = self.g[1:-1]
		self.assertTrue( gslice.match(aslice) )

class TestGrid2D(unittest.TestCase):
	def setUp(self):
		self.two_d = np.array( range(100) ).reshape(10,10)
		self.g = Grid( self.two_d.copy() )
	
	def test_match_true(self):
		self.assertTrue( self.g.match( self.two_d ) )
		
	def test_match_wrong_size(self):
		short_arr = self.two_d[:-1,:-1]
		self.assertFalse( self.g.match(short_arr) )
	
	def test_match_wrong_value(self):
		diff_arr = self.two_d.copy()
		diff_arr[0,0] = -1
		self.assertFalse( self.g.match(diff_arr) )
	
	def test_getitem_pos_int(self):
		x1 = self.two_d[0,0]
		x2 = self.g[0,0]
		self.assertTrue( x1 == x2 )
	
	def test_getitem_neg_int(self):
		x1 = self.two_d[-1,-1]
		x2 = self.g[-1,-1]
		self.assertTrue( x1 == x2 )
	
	def test_getslice_pos_int(self):
		aslice = self.two_d[0:2,0:2]
		gslice = self.g[0:2,0:2]
		self.assertTrue( gslice.match(aslice) )
	
	def test_getslice_neg_int(self):
		aslice = self.two_d[1:-1,1:-1]
		gslice = self.g[1:-1,1:-1]
		self.assertTrue( gslice.match(aslice) )

if __name__ == '__main__':
	unittest.main()
