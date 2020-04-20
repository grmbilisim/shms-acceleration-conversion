import pytest
import convert_acc
import pandas as pd
import numpy as np


# not using now - wanted to access this from parametrize decorator
@pytest.fixture(scope='module')
def files_sensor_codes():
	return {'/home/grm/acc-data-conversion/working/2019-09-26T135930/20190926110000.ALZ.003.S39z.txt': ('S39', 'S39z'),
          'ALZ.003.S39z.txt': ('S39', 'S39z'), 
          '/home/grm/AllianzSHMS/working/test-mseed-files_20190926/20190926100000.ALZ.001.B4Fx.m': ('B4F', 'B4Fx'),
            'ALZ.001.B4Fx.m': ('B4F', 'B4Fx')}


# fixtures for testing functions

@pytest.fixture(scope='module')
def unordered_file_list_short():
	return {0: 
			['20190926100000.ALZ.003.B4Fz.m',
			 '20190926100000.ALZ.001.B4Fx.m',
			 '20190926100000.ALZ.002.B4Fy.m',
			]}



@pytest.fixture(scope='module')
def ordered_file_list_short():
	return {0:
			['20190926100000.ALZ.001.B4Fx.m',
			 '20190926100000.ALZ.002.B4Fy.m',
			 '20190926100000.ALZ.003.B4Fz.m',
			]}


@pytest.fixture(scope='module')
def unordered_file_list_all():
	return {0:
			['20190926100000.ALZ.003.B4Fz.m',
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
			 '20190926100000.ALZ.001.S12x.m'
			 ]}


@pytest.fixture(scope='module')
def ordered_file_list_all():
	return {0:
			['20190926100000.ALZ.001.N39x.m',
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
			 '20190926110000.ALZ.003.FFZ.m'
			 ]}



# fixtures for testing methods
@pytest.fixture(scope='module')
def pObject():
	"""return instance of ProcessedFromTxtFile class"""
	p = convert_acc.ProcessedFromTxtFile(r'test_data/20190926100000.ALZ.001.B4Fx.txt')
	return p


@pytest.fixture(scope='module')
def cObject():
	"""return instance of Conversion class: B4Fx for event starting at 2019-09-26T135930 local time"""
	csv_file = r'test_data/processedFromTxtFile_20190926_B4Fx.csv'
	raw_df = pd.read_csv(csv_file, header=0)
	df = raw_df[['count', 'timestamp']]
	# cast timestamp values to pandas Timestamps to match those of actual df
	df['timestamp'] = df['timestamp'].apply(lambda x: pd.Timestamp(x))

	c = convert_acc.Conversion(df, 'B4F', 'B4Fx', '2019-09-26T135930')
	print(c.df.head())
	print(c.df.tail())
	return c

