
import os
from obspy import read
import pandas as pd


# get clean df from single miniseed file
class DfFromMseed:
    def __init__(self, mseedFilePath, eventTimestamp):
        self.mseedFilePath = mseedFilePath
        self.eventTimestamp = eventTimestamp
        self.workingDir = r'/home/grm/acc-data-conversion/obspy_test'       

        self.txtFilePath = self.mseedToTxt()
        self.headerList = self.getOriginalHeader()
        self.rawDf = self.convertTxtToDf()
        self.countDf = self.convertRawDf()

        print('COUNT DF STATS:\n')
        self.printPdStats(self.countDf)

        self.hourDf = self.getCleanDf(self.countDf)
        self.getTruncateIndexes(self.hourDf)

        print('HOUR DF STATS:\n')
        self.printPdStats(self.hourDf)

    def mseedToTxt(self):
        """
        convert miniseed file to txt file
        return: string holding path of new txt file
        """
        mseedFilename = self.mseedFilePath.split('/')[-1]
        txtFilename = mseedFilename.replace('.m', '.txt')
        txtFilePath = os.path.join(self.workingDir, txtFilename)
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
        # without reset index here, count data (sep by \t) will get set as index
    	df = pd.read_csv(self.txtFilePath, header=0, sep='\t').reset_index()
    	return df

    def convertRawDf(self):
        """
        return usable df (count data in single column) from raw df (with count data in multiple columns)
        """
        rowGenerator = self.rawDf.iterrows()
        # genLen = sum(1 for item in rowGenerator)
        #print('\nLENGTH OF GENERATOR: {0}\n'.format(genLen))
        #rowGenerator = self.rawDf.iterrows()
        counts = []
        for i in rowGenerator:
            for x in range(6):
                counts.append(i[1][x])
        '''
        print('\nLENGTH OF RAW ROWS LIST: {0}\n'.format(len(rawRows)))
        print('\nLENGTH OF SINGLE RAW ROW: {0}\n'.format(len(rawRows[0])))
        print('\nFIRST RAW ROW: {0}\n'.format(rawRows[0]))
        flatRawRows = [item for sublist in rawRows for item in sublist]
        print('\nLENGTH OF COUNT LIST: {0}\n'.format(len(flatRawRows)))
        '''
        # NOW UNNECESSARY - CAN RETURN LIST OF COUNTS HERE INSTEAD
        df = pd.DataFrame(columns=self.headerList)
        
        df['count'] = pd.Series(counts)
        return df

    def getTimestamp(self):
        """
        get pandas timestamp from among column headers
        return: pandas Timestamp object or None
        """
        timestamp = None
        for item in self.headerList:
            try:
                pd.Timestamp(item)
                return pd.Timestamp(item)
            except:
                pass

    def getTimestampSeries(self):
        """get pandas series holding all necessary timestamps for given hour"""
        startTime = self.getTimestamp()
        # time delta between entries: 0.01 s
        timeDelta = pd.Timedelta('0 days 00:00:00.01000')
        # time period covered by entire file after first entry - 1 hour minus 0.01 s
        filePeriod = pd.Timedelta('0 days 00:59:59.990000')
        endTime = startTime + filePeriod
        # create timestamp series
        timestampSeries = pd.Series(pd.date_range(start=startTime, end=endTime, freq=timeDelta))
        print('TIMESTAMP SERIES STATS:\n')
        self.printPdStats(timestampSeries)

        return timestampSeries

    def getCleanDf(self, templateDf):
        """
        return dataframe with only count and timestamp columns
        templateDf: dataframe with count values in single column titled 'count'
        """
        df = pd.DataFrame(columns=['timestamp', 'count'])
        df['count'] = templateDf['count']
        df['timestamp'] = self.getTimestampSeries()
        return df

    def getTruncateIndexes(self, df):
        """
        return tuple holding start and end indexes to be used to truncate dataframe
        df: dataframe from which indexes will be taken
        """
        trTimestamp = pd.Timestamp(self.eventTimestamp)
        # convert string timestamp to pandas Timestamp instance
        utcTimestamp = trTimestamp - pd.Timedelta('3 hours')
        startTime = utcTimestamp - pd.Timedelta('1 minute')
        endTime = startTime + pd.Timedelta('400 seconds')
        startIndex = df.index[df['timestamp'] == startTime]
        endIndex = df.index[df['timestamp'] == endTime]
        print(startIndex, endIndex)
        # FINISH
        #return self.df.loc[(self.df['timestamp'])]

    def printPdStats(self, pdObject):
        """
        print head, tail, length, dtypes of series or dataframe
        pdObject: pandas series or dataframe
        """
        print('df head: {0}'.format(pdObject.head()))
        print('df tail: {0}'.format(pdObject.tail()))
        print('length: {0}'.format(len(pdObject)))
        print('dtypes: {0}\n'.format(pdObject.dtypes))

testMseed = r'/home/grm/AllianzSHMS/working/test-mseed-files_20190926/20190926100000.ALZ.001.B4Fx.m'
testTimestamp = '2019-09-26T133930'

df = DfFromMseed(testMseed, testTimestamp)

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