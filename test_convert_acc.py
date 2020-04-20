import convert_acc
import unittest
import pytest

import pandas as pd

# ----------would prefer to return these dictionaries from fixtures but not yet sure how to access them in parametrize decorators-----------
# may not be possible: https://medium.com/opsops/deepdive-into-pytest-parametrization-cb21665c05b9

files_sensor_codes = {'/home/grm/acc-data-conversion/working/2019-09-26T135930/20190926110000.ALZ.003.S39z.txt': ('S39', 'S39z'),
          'ALZ.003.S39z.txt': ('S39', 'S39z'), 
          '/home/grm/AllianzSHMS/working/test-mseed-files_20190926/20190926100000.ALZ.001.B4Fx.m': ('B4F', 'B4Fx'),
            'ALZ.001.B4Fx.m': ('B4F', 'B4Fx')}

sensor_codes_floors = {'B4Fx': '4', 'N39x': '39'}

floors_axes = {'B4Fx': 'X', 'S24z': 'Z', 'N12y': 'Y'}

files_timestamps = {'/home/grm/acc-data-conversion/working/2019-09-26T135930/20190926110000.ALZ.003.S39z.txt': ('11', '20190926110000'),
					'/home/grm/AllianzSHMS/working/test-mseed-files_20190926/20190926100000.ALZ.001.B4Fx.m': ('10', '20190926100000')}

# --------------------------------------------------------------


# ------------------unit tests for functions------------------

@pytest.mark.parametrize("test_input, expected", [(k, v) for k, v in files_sensor_codes.items()])
def test_getSensorCodeInfo(test_input, expected):
	assert convert_acc.getSensorCodeInfo(test_input) == expected


@pytest.mark.parametrize("test_input, expected", [(k, v) for k, v in sensor_codes_floors.items()])
def test_getFloor(test_input, expected):
	assert convert_acc.getFloor(test_input) == expected


@pytest.mark.parametrize("test_input, expected", [(k, v) for k, v in floors_axes.items()])
def test_getAxis(test_input, expected):
	assert convert_acc.getAxis(test_input) == expected


@pytest.mark.parametrize("test_input, expected", [(k, v) for k, v in files_timestamps.items()])
def test_getTimeText(test_input, expected):
	assert convert_acc.getTimeText(test_input) == expected


def test_sortFilesBySensorCode(unordered_file_list_short, ordered_file_list_short):
	assert convert_acc.sortFilesBySensorCode(unordered_file_list_short[0]) == ordered_file_list_short[0]


def test_sortFiles(unordered_file_list_all, ordered_file_list_all):
	assert convert_acc.sortFiles(unordered_file_list_all[0]) == ordered_file_list_all[0]


# ----------unit tests for methods------------

# unit tests for ProcessedFromTxtFile methods

def test_setHeaderList(pObject):
	assert pObject.headerList == ['TIMESERIES AT_ALZ__B4F_D',' 360000 samples', ' 100 sps', ' 2019-09-26T10:00:00.010000', ' SLIST',
       ' INTEGER', ' Counts']


def test_getTimestamp(pObject):
	assert pObject.getTimestamp() == pd.Timestamp('2019-09-26 10:00:00.010000')


def test_getCountColumnHeader(pObject):
	""""assert that column containing count data has data at first index"""
	assert pObject.df['count'][0] is not None


def test_getTimestampSeries(pObject):
	"""assert correct first and last values of series and correct interval between first two value"""
	timestampSeries = pObject.getTimestampSeries()
	assert timestampSeries[0] == pd.Timestamp('2019-09-26 10:00:00.010')
	assert timestampSeries[len(timestampSeries) - 1] == pd.Timestamp('2019-09-26 11:00:00')
	assert timestampSeries[1] - timestampSeries[0] == pd.Timedelta('0 days 00:00:00.01000')


# unit tests for Conversion methods
'''
def test_truncateDfFirst(cObject):
	assert cObject.df.iloc[0]['timestamp'] == pd.Timestamp('2019-09-26 10:58:30')

def test_truncateDfLast(cObject):
	assert cObject.df.iloc[-1]['timestamp'] == pd.Timestamp('2019-09-26 11:05:10')
	#assert len(cObject.df) == 401
'''


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