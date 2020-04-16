import convert_acc
import unittest


class TestList(unittest.TestCase):

	def test_getAllSensorCodesWithChannels(self):
		def setUp(self):
			self.expected = ['N39x',
				'S39x', 
				'N24x', 
				'S24x', 
				'N12x', 
				'S12x', 
				'B4Fx',
				'N39y', 
				'S39y', 
				'N24y', 
				'S24y', 
				'N12y', 
				'S12y', 
				'B4Fy',
				'N39z', 
				'S39z', 
				'N24z', 
				'S24z', 
				'N12z', 
				'S12z', 
				'B4Fz',
				'FFN',
				'FFW',
				'FFZ'
				]
			self.assertCountEqual(convert_acc.getAllSensorCodesWithChannels(), self.expected)