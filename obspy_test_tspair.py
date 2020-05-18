
import os
from obspy import read
import pandas as pd
import time

startTime = time.time()
print(time.strftime("start time after imports: %a, %d %b %Y %H:%M:%S", time.localtime()))

# get clean df from single miniseed file
class DfFromMseed:
    def __init__(self, mseedFilePath, eventTimestamp):
        self.mseedFilePath = mseedFilePath
        self.eventTimestamp = eventTimestamp
        self.workingDir = r'/home/grm/acc-data-conversion/obspy_test'       

        self.txtFilePath = self.mseedToTxt()
        self.headerList = self.getOriginalHeader()
        self.rawDf = self.convertTxtToDf()
        self.hourDf = self.getCleanDf()
        self.getTruncateIndexes(self.hourDf)

    def mseedToTxt(self):
        """
        convert miniseed file to txt file - will not be processed but text file will contain
        timestamp and count values as string in single column
        return: string holding path of new txt file
        """
        tspairStartTime = time.time()
        mseedFilename = self.mseedFilePath.split('/')[-1]
        txtFilename = mseedFilename.replace('.m', '.txt')
        txtFilePath = os.path.join(self.workingDir, txtFilename)
        stream = read(self.mseedFilePath)
        try:
            stream.write(txtFilePath, format='TSPAIR')
        except Exception as e:
            print('error converting miniseed to text file: {0}'.format(e))
        tspairTime = time.time() - tspairStartTime
        print('tspair file written in {0} seconds'.format(tspairTime))
        return txtFilePath

    def getOriginalHeader(self):
        """return header (first line of txt file)"""
        tempDf = pd.read_csv(self.txtFilePath, nrows=0)
        return [item.strip() for item in list(tempDf.columns.values)]

    def convertTxtToDf(self):
        """return pandas dataframe converted from text file with no processing"""
        df = pd.read_csv(self.txtFilePath, header=0)
        return df

    def getSeparateSeries(self):
        """get separate timestamp and count series"""
        # get series holding timestamps and counts as single string
        singleColumnDf = self.rawDf.iloc[:, [0]]
        header = singleColumnDf.columns[0]
        # get separate series with string as dtype
        timestampSeries = pd.Series()
        countSeries = pd.Series()
        timestampSeries, countSeries = singleColumnDf[header].str.split(' ', 1).str
        # cast series to respective datatypes
        #timestampSeries.astype(pd.Timestamp)
        countSeries.astype('int32')
        return timestampSeries, countSeries

    def getCleanDf(self):
        """
        return dataframe with only count and timestamp columns
        templateDf: dataframe with count values in single column titled 'count'
        """
        df = pd.DataFrame(columns=['timestamp', 'count'])
        df['timestamp'], df['count'] = self.getSeparateSeries()
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

seconds = time.time() - startTime
print("--- {} seconds ---".format(seconds))