import convert_acc
import unittest
import pytest

import pandas as pd
import numpy as np

# dictionaries holding test inputs as keys and expected values as values
# ----------would prefer to return these dictionaries from fixtures but not yet sure how to access them in parametrize decorators-----------
# may not be possible: https://medium.com/opsops/deepdive-into-pytest-parametrization-cb21665c05b9

fileSensorCode = {'/home/grm/acc-data-conversion/working/2019-09-26T135930/20190926110000.ALZ.003.S39z.txt': ('S39', 'S39z'),
          'ALZ.003.S39z.txt': ('S39', 'S39z'), 
          '/home/grm/AllianzSHMS/working/test-mseed-files_20190926/20190926100000.ALZ.001.B4Fx.m': ('B4F', 'B4Fx'),
            'ALZ.001.B4Fx.m': ('B4F', 'B4Fx')}

sensorCodeFloor = {'B4Fx': '4', 'N39x': '39'}

floorAxis = {'B4Fx': 'X', 'S24z': 'Z', 'N12y': 'Y'}

fileTimestamp = {'/home/grm/acc-data-conversion/working/2019-09-26T135930/20190926110000.ALZ.003.S39z.txt': ('11', '20190926110000'),
					'/home/grm/AllianzSHMS/working/test-mseed-files_20190926/20190926100000.ALZ.001.B4Fx.m': ('10', '20190926100000')}

#countG = {-39527: -0.009423971176147461, 8250: 0.0019669532775878906}

#sensorCodeWithChannelSensitivity = {'B4Fx': 1.25, 'FFN': 1.25, 'N39z': 0.625, 'S12y': 0.625}

# --------------------------------------------------------------


# ------------------unit tests for functions------------------

@pytest.mark.parametrize("testInput, expected", [(k, v) for k, v in fileSensorCode.items()])
def test_getSensorCodeInfo(testInput, expected):
	assert convert_acc.getSensorCodeInfo(testInput) == expected


@pytest.mark.parametrize("testInput, expected", [(k, v) for k, v in sensorCodeFloor.items()])
def test_getFloor(testInput, expected):
	assert convert_acc.getFloor(testInput) == expected


@pytest.mark.parametrize("testInput, expected", [(k, v) for k, v in floorAxis.items()])
def test_getAxis(testInput, expected):
	assert convert_acc.getAxis(testInput) == expected


@pytest.mark.parametrize("testInput, expected", [(k, v) for k, v in fileTimestamp.items()])
def test_getTimeText(testInput, expected):
	assert convert_acc.getTimeText(testInput) == expected


def test_sortFilesBySensorCode(unorderedFileListShort, orderedFileListShort):
	assert convert_acc.sortFilesBySensorCode(unorderedFileListShort[0]) == orderedFileListShort[0]


def test_sortFiles(unorderedFileListAll, orderedFileListAll):
	assert convert_acc.sortFiles(unorderedFileListAll[0]) == orderedFileListAll[0]


# ----------unit tests for methods------------

# unit tests for ProcessedFromTxtFile methods
# using ProcessedFromTxtFile object from 'test_data/20190926100000.ALZ.001.B4Fx.txt'

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
# using Conversion object with B4Fx for event starting at 2019-09-26T135930 local time
# may have been better to mock the dataframe but keeping test data for now


# cannot do it this way - need an instance or mock instance of the class
'''
@pytest.mark.parametrize("testInput, expected", [(k, v) for k, v in sensorCodeWithChannelSensitivity.items()])
def test_setSensitivity(testInput, expected):
	assert convert_acc.Conversion.setSensitivity(testInput) == expected


@pytest.mark.parametrize("testInput, expected", [(k, v) for k, v in countG.items()])
def test_convertCountToG(testInput, expected):
	assert convert_acc.Conversion.convertCountToG(testInput) == expected
'''


def test_getTruncateIndexes(cObject):
	"""assert that indexes used for truncating dataframe are correct"""
	assert cObject.getTruncateIndexes() == (350999, 390999)


def test_truncateDfFirstRow(cObject):
	"""assert that first row of timestamp column of dataframe is correct"""
	netIgnoredSamples = cObject.ignoredSamples - cObject.zeroPadLength
	ignoredSeconds = netIgnoredSamples / 100
	ignoredTimedelta = pd.Timedelta('{0} seconds'.format(ignoredSeconds))
	assert cObject.df.iloc[0]['timestamp'] == pd.Timestamp('2019-09-26 10:58:30') + ignoredTimedelta


def test_truncateDfLastLow(cObject):
	"""assert that last row of timestamp column of dataframe is correct"""
	assert cObject.df.iloc[-1]['timestamp'] == pd.Timestamp('2019-09-26 11:05:10')


def test_setSensitivity(cObject):
	assert cObject.setSensitivity('B4Fx') == 1.25
	assert cObject.setSensitivity('FFN') == 1.25
	assert cObject.setSensitivity('N39y') == 0.625
	assert cObject.setSensitivity('S12z') == 0.625


def test_convertCountToG(cObject):
	assert cObject.convertCountToG(-39527) == -0.009423971176147461
	assert cObject.convertCountToG(8250) == 0.0019669532775878906


def test_getZeroPaddedDf(cObject):
	"""
	assert that dataframes returned by getZeroPaddedDf contain zero pads of 
	correct lengths 
	"""
	paddedDf = cObject.getZeroPaddedDf(cObject.inputDf, ['timestamp', 'count'])
	zeros = np.zeros(shape=(cObject.zeroPadLength))
	zeroPad = pd.Series(zeros)
	nullList = [None] * cObject.zeroPadLength
	timestampHead = list(paddedDf.iloc[:cObject.zeroPadLength]['timestamp'])
	timestampTail = list(paddedDf.iloc[-cObject.zeroPadLength:]['timestamp'])
	countHead = list(paddedDf.iloc[:cObject.zeroPadLength]['count'])
	countTail = list(paddedDf.iloc[-cObject.zeroPadLength:]['count'])
	assert timestampHead == nullList
	assert timestampTail == nullList
	assert countHead == list(zeroPad)
	assert countTail == list(zeroPad)


def test_convertGToMetric(cObject):
	assert cObject.convertGToMetric(0.07916) == 0.7762944139999999


def test_butterPass(cObject):
	"""assert that the returned arrays are equal up to 8 decimal places"""
	bandB, bandA = cObject.butterPass('band')
	highB, highA = cObject.butterPass('high')
	np.testing.assert_almost_equal(bandB, [0.63748352, 0., -1.27496704, 0., 0.63748352], 8)
	np.testing.assert_almost_equal(bandA, [1., -0.85279266, -0.87204231, 0.31384784, 0.41101232], 8)
	np.testing.assert_almost_equal(highB, [0.99778102, -1.99556205, 0.99778102], 8)
	np.testing.assert_almost_equal(highA, [1., -1.99555712, 0.99556697], 8)


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

@pytest.mark.parametrize("testInput,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def test_eval(testInput, expected):
    assert eval(testInput) == expected


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