import unittest
import numpy as np
from grid import Grid

class TestGrid1D(unittest.TestCase):
	def setUp(self):
		self.one_d = np.array( range(10) )
		self.g = Grid( self.one_d.copy() )
	
	def test_full_match_1d_true(self):
		self.assertTrue( self.g.full_match( self.one_d ) )
		
	def test_full_match_1d_wrong_size(self):
		short_arr = self.one_d[:-1]
		self.assertFalse( self.g.full_match(short_arr) )
	
	def test_full_match_1d_wrong_value(self):
		diff_arr = self.one_d.copy()
		diff_arr[0] = -1
		self.assertFalse( self.g.full_match(diff_arr) )

if __name__ == '__main__':
	unittest.main()