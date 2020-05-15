
import os
from obspy import read
import pandas as pd


# get clean df from single miniseed file
class DfFromMseed:
    def __init__(self, mseedFilePath):
        self.mseedFilePath = mseedFilePath
        self.workingDir = r'/home/grm/acc-data-conversion/obspy_test'       

        self.txtFilePath = self.mseedToTxt()
        self.headerList = self.getOriginalHeader()
        self.rawDf = self.convertTxtToDf()
        self.df = self.convertRawDf()
        self.printDfStats(self.df)

    def mseedToTxt(self):
    	"""
    	convert miniseed file to txt file
    	return: string holding path of new txt file
    	"""
    	basename = self.mseedFilePath.rsplit(".m")[0]
    	txtFilename = basename + '.txt'
    	txtFilePath = os.path.join(basename, txtFilename)
    	stream = read(self.mseedFilePath)
    	try:
    		stream.write(txtFilePath, format='SLIST')
    	except Exception as e:
    		print('error converting miniseed to text file: {0}'.format(e))
    	return txtFilePath

    def getOriginalHeader(self):
    	"""return header (first line of txt file)"""
    	tempDf = pd.read_csv(self.txtFilePath, nrows=0)
    	return [item.strip() for item in list(tempDf.columns.values)]

    def convertTxtToDf(self):
    	"""return pandas dataframe converted from text file with no processing"""
    	df = pd.read_csv(self.txtFilePath, header=0, sep='\t')
    	return df

    def convertRawDf(self):
    	"""
    	return usable df (count data in single column) from raw df (with count data in multiple columns)
    	"""
    	rowGenerator = self.rawDf.iterrows()
    	rawRows = []
    	for i in rowGenerator:
    		rawRows.append(i[0])
    	flatRawRows = [item for sublist in rawRows for item in sublist]
    	df = pd.DataFrame(columns=self.headerList)
    	df['count'] = pd.Series(flatRawRows)
    	return df

    def truncateDf(self):
    	pass

    def addTimestamp(self):
    	pass

    def printDfStats(self, df):
    	print('df head: {0}'.format(df.head()))
    	print('df tail: {0}'.format(df.tail()))
    	print('length of df: {0}'.format(len(df)))
    	print('dtypes: {0}'.format(df.dtypes))

testMseed = r'/home/grm/AllianzSHMS/working/test-mseed-files_20190926/20190926100000.ALZ.001.B4Fx.m'

df = DfFromMseed(testMseed)

'''
stream = read(r'/home/grm/AllianzSHMS/working/test-mseed-files_20190926/20190926100000.ALZ.001.B4Fx.m')
stream.write(r'/home/grm/AllianzSHMS/working/obspy_test_B4Fx.txt', format='SLIST')

# use both tabs and spaces as delimiter
df = pd.read_csv(r'/home/grm/AllianzSHMS/working/obspy_test_B4Fx.txt', header=0, sep='\t|,', engine='python')

g = df.iterrows()
s = []
for i in g:
	s.append(i[0])



print('original df head:')
print(df.head())


print(s)

'''

# drop all data from df (maintain header)
#df.drop(df.index, inplace=True)

# attempt to put new data back in old df returns ValueError

# ValueError: Buffer dtype mismatch, expected 'Python object' but got 'long'

#df['test'] = pd.Series(c)

# so...

'''
c = [item for sublist in s for item in sublist]

count = pd.Series(c)
df1 = pd.DataFrame({'Count': count})

print('new df head:')
df1.head()
'''