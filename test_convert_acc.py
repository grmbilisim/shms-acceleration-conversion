import convert_acc
import unittest
import pytest


@pytest.fixture(scope='module')
def unordered_file_list_short():
	return {0: 
			['20190926100000.ALZ.003.B4Fz.m',
			 '20190926100000.ALZ.001.B4Fx.m',
			 '20190926100000.ALZ.002.B4Fy.m',
			]}


@pytest.fixture(scope='module')
def ordered_file_list_short():
	return ['20190926100000.ALZ.001.B4Fx.m',
			 '20190926100000.ALZ.002.B4Fy.m',
			 '20190926100000.ALZ.003.B4Fz.m',
			]


@pytest.fixture(scope='module')
def unordered_file_list_all():
	return ['20190926100000.ALZ.003.B4Fz.m',
			 '20190926100000.ALZ.002.B4Fy.m',
			 '20190926100000.ALZ.003.N39z.m',
			 '20190926100000.ALZ.001.B4Fx.m',
			 '20190926110000.ALZ.002.S24y.m',
			 '20190926110000.ALZ.001.N24x.m',
			 '20190926110000.ALZ.002.FFW.m',
			 '20190926110000.ALZ.003.S24z.m',
			 '20190926100000.ALZ.003.N24z.m',
			 '20190926110000.ALZ.003.FFZ.m',
			 '20190926100000.ALZ.001.N24x.m',
			 '20190926100000.ALZ.002.N12y.m',
			 '20190926110000.ALZ.002.N12y.m',
			 '20190926100000.ALZ.001.FFN.m',
			 '20190926110000.ALZ.002.S39y.m',
			 '20190926110000.ALZ.001.S39x.m',
			 '20190926110000.ALZ.003.N39z.m',
			 '20190926110000.ALZ.003.S12z.m',
			 '20190926100000.ALZ.002.S24y.m',
			 '20190926100000.ALZ.003.S24z.m',
			 '20190926100000.ALZ.002.S39y.m',
			 '20190926110000.ALZ.001.B4Fx.m',
			 '20190926100000.ALZ.002.N24y.m',
			 '20190926110000.ALZ.001.S24x.m',
			 '20190926110000.ALZ.003.N12z.m',
			 '20190926100000.ALZ.001.N39x.m',
			 '20190926110000.ALZ.001.N12x.m',
			 '20190926110000.ALZ.002.S12y.m',
			 '20190926100000.ALZ.002.S12y.m',
			 '20190926100000.ALZ.001.N12x.m',
			 '20190926110000.ALZ.002.N24y.m',
			 '20190926110000.ALZ.003.S39z.m',
			 '20190926100000.ALZ.003.S12z.m',
			 '20190926100000.ALZ.003.N12z.m',
			 '20190926110000.ALZ.001.S12x.m',
			 '20190926110000.ALZ.003.N24z.m',
			 '20190926110000.ALZ.001.N39x.m',
			 '20190926100000.ALZ.002.N39y.m',
			 '20190926110000.ALZ.002.B4Fy.m',
			 '20190926110000.ALZ.003.B4Fz.m',
			 '20190926100000.ALZ.001.S24x.m',
			 '20190926100000.ALZ.003.FFZ.m',
			 '20190926100000.ALZ.002.FFW.m',
			 '20190926100000.ALZ.003.S39z.m',
			 '20190926110000.ALZ.002.N39y.m',
			 '20190926110000.ALZ.001.FFN.m',
			 '20190926100000.ALZ.001.S39x.m',
			 '20190926100000.ALZ.001.S12x.m']


@pytest.fixture(scope='module')
def ordered_file_list_all():
	return ['20190926100000.ALZ.001.N39x.m',
			 '20190926100000.ALZ.002.N39y.m',
			 '20190926100000.ALZ.003.N39z.m',
			 '20190926100000.ALZ.001.S39x.m',
			 '20190926100000.ALZ.002.S39y.m',
			 '20190926100000.ALZ.003.S39z.m',
			 '20190926100000.ALZ.001.N24x.m',
			 '20190926100000.ALZ.002.N24y.m',
			 '20190926100000.ALZ.003.N24z.m',
			 '20190926100000.ALZ.001.S24x.m',
			 '20190926100000.ALZ.002.S24y.m',
			 '20190926100000.ALZ.003.S24z.m',
			 '20190926100000.ALZ.001.N12x.m',
			 '20190926100000.ALZ.002.N12y.m',
			 '20190926100000.ALZ.003.N12z.m',
			 '20190926100000.ALZ.001.S12x.m',
			 '20190926100000.ALZ.002.S12y.m',
			 '20190926100000.ALZ.003.S12z.m',
			 '20190926100000.ALZ.001.B4Fx.m',
			 '20190926100000.ALZ.002.B4Fy.m',
			 '20190926100000.ALZ.003.B4Fz.m',
			 '20190926100000.ALZ.001.FFN.m',
			 '20190926100000.ALZ.002.FFW.m',
			 '20190926100000.ALZ.003.FFZ.m',
			 '20190926110000.ALZ.001.N39x.m',
			 '20190926110000.ALZ.002.N39y.m',
			 '20190926110000.ALZ.003.N39z.m',
			 '20190926110000.ALZ.001.S39x.m',
			 '20190926110000.ALZ.002.S39y.m',
			 '20190926110000.ALZ.003.S39z.m',
			 '20190926110000.ALZ.001.N24x.m',
			 '20190926110000.ALZ.002.N24y.m',
			 '20190926110000.ALZ.003.N24z.m',
			 '20190926110000.ALZ.001.S24x.m',
			 '20190926110000.ALZ.002.S24y.m',
			 '20190926110000.ALZ.003.S24z.m',
			 '20190926110000.ALZ.001.N12x.m',
			 '20190926110000.ALZ.002.N12y.m',
			 '20190926110000.ALZ.003.N12z.m',
			 '20190926110000.ALZ.001.S12x.m',
			 '20190926110000.ALZ.002.S12y.m',
			 '20190926110000.ALZ.003.S12z.m',
			 '20190926110000.ALZ.001.B4Fx.m',
			 '20190926110000.ALZ.002.B4Fy.m',
			 '20190926110000.ALZ.003.B4Fz.m',
			 '20190926110000.ALZ.001.FFN.m',
			 '20190926110000.ALZ.002.FFW.m',
			 '20190926110000.ALZ.003.FFZ.m']


@pytest.mark.parametrize("test_input, expected", [('/home/grm/acc-data-conversion/working/2019-09-26T135930/20190926110000.ALZ.003.S39z.txt', ('S39', 'S39z')),
													('ALZ.003.S39z.txt', ('S39', 'S39z')),
													('/home/grm/AllianzSHMS/working/test-mseed-files_20190926/20190926100000.ALZ.001.B4Fx.m', ('B4F', 'B4Fx')),
													('ALZ.001.B4Fx.m', ('B4F', 'B4Fx')),
														])
def test_getSensorCodeInfo(test_input, expected):
	assert convert_acc.getSensorCodeInfo(test_input) == expected


def test_getFloor():
	assert convert_acc.getFloor('B4Fx') == '4', "Expected '4'"
	assert convert_acc.getFloor('N39x') == '39', "Expected '39'"


def test_getAxis():
	assert convert_acc.getAxis('B4Fx') == 'X', "Expected 'X'"
	assert convert_acc.getAxis('S24z') == 'Z', "Expected 'Z'"
	assert convert_acc.getAxis('N12y') == 'Y', "Expected 'Y'"


@pytest.mark.parametrize("test_input, expected", [('/home/grm/acc-data-conversion/working/2019-09-26T135930/20190926110000.ALZ.003.S39z.txt', ('11', '20190926110000')),
													('/home/grm/AllianzSHMS/working/test-mseed-files_20190926/20190926100000.ALZ.001.B4Fx.m', ('10', '20190926100000')),
														])
def test_getTimeText(test_input, expected):
	assert convert_acc.getTimeText(test_input) == expected


def test_sortFilesBySensorCode(unordered_file_list_short):
	assert convert_acc.sortFilesBySensorCode(unordered_file_list_short[0]) == ['20190926100000.ALZ.001.B4Fx.m',
																				 '20190926100000.ALZ.002.B4Fy.m',
																				 '20190926100000.ALZ.003.B4Fz.m',
																				]



# -------------pytest examples------------------
'''
# ---------basic

def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"

# ---------no parametrization

def test_getSensorCodeInfo():
	assert convert_acc.getSensorCodeInfo('/home/grm/acc-data-conversion/working/2019-09-26T135930/20190926110000.ALZ.003.S39z.txt') == ('S39', 'S39z'), "Expected ('S39', 'S39z')"

	assert convert_acc.getSensorCodeInfo('ALZ.003.S39z.txt') == ('S39', 'S39z'), "Expected ('S39', 'S39z')"

	assert convert_acc.getSensorCodeInfo('/home/grm/AllianzSHMS/working/test-mseed-files_20190926/20190926100000.ALZ.001.B4Fx.m') == ('B4F', 'B4Fx'), "Expected ('B4F', 'B4Fx')"

	assert convert_acc.getSensorCodeInfo('ALZ.001.B4Fx.m') == ('B4F', 'B4Fx'), "Expected ('B4F', 'B4Fx')"


# -----------parametrize example with one failure

@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected


# -----------fixture and parametrize together

@pytest.fixture(scope = 'module')
def global_data():
    return {'presentVal': 0}

@pytest.mark.parametrize('iteration', range(1, 6))
def test_global_scope(global_data, iteration):

    assert global_data['presentVal'] == iteration - 1
    global_data['presentVal'] = iteration
    assert global_data['presentVal'] == iteration

'''


# ---------------------unittest examples-------------------------
'''
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
'''