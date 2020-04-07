# Filename: convert_acc.py

"""Visualization and conversion of accelerometer data to velocity and displacement using Python and PyQt5."""

"""Successfully clipping df (removing first 6000 sample after bandpass filter) in Conversion class for plotting and stats"""

import sys
import os
import subprocess
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, detrend
import math
import logging


from matplotlib.figure import Figure
import matplotlib.pyplot as plt


__version__ = '0.1'
__author__ = 'Nick LiBassi'

ERROR_MSG = 'ERROR'

# set logging to 'DEBUG' to print all statements to console, 'INFO' to print less
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

"""
General Notes:
use camelCase throughout as Qt uses it
remove unused imports leftover from calculator app

fix inconsistent use of '_' for private variables
fix inconsisent use of sensor code naming (prefixes, sensorCode, sensorCodeWithChannel)
"""

SENSOR_CODES = ('N39', 'S39', 'N24', 'S24', 'N12', 'S12', 'B4F', 'FF')


def getAllSensorCodesWithChannels():
    """
    Return list of 24 strings holding sensor codes with channels in form 'N39x'
    """
    buildingSensorCodes = [c for c in SENSOR_CODES if c != 'FF']
    axes = ['x', 'y', 'z']
    farFieldCodes = ['FFW', 'FFN', 'FFZ']
    allSensorCodesWithChannels = [] + farFieldCodes
    for p in buildingSensorCodes:
        for a in axes:
            allSensorCodesWithChannels.append(p + a)
    return allSensorCodesWithChannels


def getSensorCodeInfo(inputFile):
    """
    Return strings holding sensor code only (ex. 'B4F') and sensor code with channel (ex. 'B4Fx')
    In far field cases, these will be identical ('FFW', 'FFN', 'FFZ')
    args:
    inputFile: string holding either filename or path of .txt or .m file
    """
    # get string holding file extension - either 'txt' or 'm' (without dot)
    inputFileExt = inputFile.split('.')[-1]
    inputFileBase = inputFile.split('.' + inputFileExt)[0]
    # info in filenames are separated by dots
    sensorCodeWithChannel = inputFileBase.rsplit('.')[-1]

    lastLetter = sensorCodeWithChannel[-1]
    if lastLetter.islower():
        sensorCode = sensorCodeWithChannel[:-1]
    else:
        sensorCode = sensorCodeWithChannel

    return sensorCode, sensorCodeWithChannel


def getTimeText(inputFile):
    """
    Get hour (int) and full timestamp (string) from given miniseed or text file name.
    args:
    inputFile: string holding either filename or path of .txt or .m file
    return:
    tuple in form: (int holding hour between 0-23, string holding full timestamp)
    ex. (10, '20190926100000')
    """
    filename = inputFile.split('/')[-1]
    timestampText = filename.split('.')[0]
    UTCHour = timestampText[8:10]
    return (UTCHour, timestampText)


def sortFilesBySensorCode(inputFileList):
    """
    Sort given list of files by sensor code.
    inputFileList: list of strings holding filenames (or paths?) of miniseed or text files.
    return: list of files sorted by sensor code (according to order of SENSOR_CODES).
    """
    sortedList = []
    for code in SENSOR_CODES:
        fileGroup = [f for f in inputFileList if code in f]
        sortedFileGroup = sorted(fileGroup)
        sortedList += sortedFileGroup
    return sortedList


def sortFiles(inputFileList):
    """
    Sort given list of files according to Safe report.
    inputFileList: list of strings holding filenames or full paths of miniseed or text files.
    return: sorted list of input text files
    """
    uniqueTimeTuples = list(set([getTimeText(file) for file in inputFileList]))
    numHours = len(uniqueTimeTuples)
    if numHours == 1:
        return sortFilesBySensorCode(inputFileList)
    elif numHours == 2:
        if uniqueTimeTuples[0][0] < uniqueTimeTuples[1][0]:
            firstTime = uniqueTimeTuples[0][1]
        else:
            firstTime = uniqueTimeTuples[1][1]
        firstFileList = sortFilesBySensorCode([f for f in inputFileList if firstTime in f])
        secondFileList = sortFilesBySensorCode([f for f in inputFileList if firstTime not in f])
        return firstFileList + secondFileList


# get info on miniseed files, convert to to text files, get info on text files
class InputProcessing:
    def __init__(self, eventTimestamp, miniseedDirPath):
        """
        initialize InputProcessing object
        eventTimestamp: string holding approximate time of event in form '2020-01-11T163736'
        mseedDirPath: string holding path of directory holding (24 or 48) miniseed files 
        """
        self.eventTimestamp = eventTimestamp
        self.miniseedDirPath = miniseedDirPath
        self.miniseedFileList = None
        self.miniseedFileCount = None
        
        self.workingBaseDir = "/home/grm/acc-data-conversion/working/no_ui"
        self.workingDirPath = None
        self.txtFileList = None
        self.txtFileCount = None
        self.pairedTxtFileList = []
        
        self.setMiniseedFileInfo()
        self.setWorkingDir()
        self.convertMiniseedToAscii()
        self.setTxtFileInfo()
        # if input file count is 48, self.pairedTxtFileList will not be empty 
        # after self.pairDeviceTxtFiles() runs - 
        # will hold list of list of two-item tuples in format (filename, sensorCode)
        self.pairDeviceTxtFiles()


    def setMiniseedFileInfo(self):
        """Get number of input miniseed files (must be either 24 or 48)"""
        self.miniseedFileList = [f for f in os.listdir(self.miniseedDirPath) if f.endswith(".m")]
        self.miniseedFileCount = len(self.miniseedFileList)
        logging.debug('miniseed file count: {}'.format(self.miniseedFileCount))
        if self.miniseedFileCount not in [24, 48]:
            raise ValueError('number of input miniseed files must be 24 or 48 - check directory holding miniseed files')         
            

    def setWorkingDir(self):
        """
        If not yet created, create working dir using event id entered
        by user.
        """
        
        self.workingDirPath = os.path.join(self.workingBaseDir, self.eventTimestamp)

        if not os.path.isdir(self.workingDirPath):
            os.mkdir(self.workingDirPath)


    def convertMiniseedToAscii(self):
        """
        Convert all miniseed files in miniseed directory to ascii files 
        """
        for f in self.miniseedFileList:
            basename = f.rsplit(".m")[0]
            filename = basename + ".txt"
            outPath = os.path.join(self.workingDirPath, filename)
            os.chdir(self.miniseedDirPath)
            subprocess.run(["./mseed2ascii", f, "-o", outPath])


    def setTxtFileInfo(self):
        """Set list of sorted input text file paths and number of text files"""
        self.txtFileList = [os.path.join(self.workingDirPath, f) for f in os.listdir(self.workingDirPath)]
        self.txtFileList = sortFiles(self.txtFileList)
        # may not need self._inputTxtFileCount
        self.txtFileCount = len(self.txtFileList)
        for path in self.txtFileList:
            logging.debug(path)


    def pairDeviceTxtFiles(self):
        """
        If seismic event spans two hours, create list of 24 two-item lists holding path pairs of miniseed files to be processed.
        Assume number of input miniseed files to be 24 (if event falls in single hour) or 48 (if event spans two hours)
        """
        if self.txtFileCount == 48:
            txtFileSensorCodeList = []
            for file in self.txtFileList:
                sensorCodeWithChannel = getSensorCodeInfo(file)[1]
                txtFileSensorCodeList.append((file, sensorCodeWithChannel))

            for t in txtFileSensorCodeList:
                logging.debug('txtFileSensorCodeList tuple: {}'.format(t))

            allSensorCodesWithChannels = getAllSensorCodesWithChannels()
            for c in allSensorCodesWithChannels:
                pairedList = [f for f in txtFileSensorCodeList if c in f]
                self.pairedTxtFileList.append(pairedList)



# get clean df from single text file
class ProcessedFromTxtFile:
    def __init__(self, txtFilePath):
        self.txtFilePath = txtFilePath
        self.df = None
        self.headerList = None
        self.sensorCode, self.sensorCodeWithChannel = getSensorCodeInfo(self.txtFilePath)
        
        self.convertTxtToDf()
        self.setHeaderList()
        self.getDfWithTimestampedCounts()


    def convertTxtToDf(self):
        """convert given text file to pandas dataframe"""
        self.df = pd.read_csv(self.txtFilePath, header=0)


    def setHeaderList(self):
        """set header list for object from dataframe columns"""
        self.headerList = [item for item in list(self.df.columns.values)]


    def getFirstTimestamp(self):
        """
        get earliest pandas timestamp from among column headers 
        (which will contain either one or two timestamps)
        return: pandas Timestamp object
        """
        # create empty timestamp list
        tsList = []
        for item in self.headerList:
            try:
                ts = pd.Timestamp(item)
                tsList.append(ts)
            except:
                pass
        earliestTs = tsList[0]
        if len(tsList) == 1:
            return earliestTs
        elif len(tsList) == 2:
            if tsList[1] < tsList[0]:
                earliestTs = tsList[1]
            return earliestTs


    def getCountColumnHeader(self):
        """
        get column header that contains count values
        return: string holding name of column holding count values 
        """
        countHeader = None
        for header in self.headerList:
            firstVal = self.df[header][0]
            logging.debug('{0}:{1}'.format(header, firstVal))
            if firstVal is not None:
                logging.debug(firstVal)
                countHeader = header
                logging.debug(countHeader)
                return countHeader


    def getDfWithTimestampedCounts(self):
        """arrange self.df to contain only timestamp and count columns"""
        startTime = self.getFirstTimestamp()
        # time delta between entries: 0.01 s
        timeDelta = pd.Timedelta('0 days 00:00:00.01000')
        # time delta of entire file - 1 hour minus  0.01 s
        fileTimeDelta = pd.Timedelta('0 days 00:59:59.990000')
        endTime = startTime + fileTimeDelta
        # create timestamp series
        timestampSeries = pd.Series(pd.date_range(start=startTime, end=endTime, freq=timeDelta))
        # add new columns to dataframe
        self.df['count'] = self.df[self.getCountColumnHeader()]
        self.df['timestamp'] = timestampSeries
        requiredColumns = ['timestamp', 'count']
        extraneousColumns = [header for header in self.headerList if header not in requiredColumns]
        for c in extraneousColumns:
            self.df.drop(c, axis=1, inplace=True)
        logging.info(self.df.head())


# class used to convert data from count to acceleration, velocity, and displacement
class Conversion:
    def __init__(self, df, sensorCode, sensorCodeWithChannel, eventTimestamp):
        self.df = df
        self.sensorCode = sensorCode
        self.sensorCodeWithChannel = sensorCodeWithChannel
        self.eventTimestamp = eventTimestamp
        self.sensitivity = None

        self.plotDir = "/home/grm/acc-data-conversion/working/no_ui/plots"

        self.accRawStats = None
        self.accOffsetStats = None
        self.accBandpassedStats = None
        self.velStats = None
        self.dispStats = None

        # int holding number of samples (at head of dataset) to ignore when plotting and getting stats
        self.ignoredSamples = 7000

        # these may be taken from user in future
        # low cutoff frequency for bandpass and highpass filters
        self.lowcut = 0.07
        # high cutoff frequency for bandpass filter
        self.highcut = 40
        # sampling frequency
        self.fs = 100
        # order of filters
        self.order = 2
        # time between samples in seconds
        self.dt = 1/float(self.fs)
        
        self.truncateDf()
        self.setSensitivity()
        self.convertCountToG()
        self.addZeroPad()
        self.convertGToOffsetG()
        self.convertGToMetric()

        self.butterBandpassFilter('offset_g', 'bandpassed_g')
        self.butterBandpassFilter('acc_ms2', 'bandpassed_ms2')

        self.integrateDfColumn('bandpassed_ms2', 'velocity_ms')

        self.clipDf()

        self.detrendData('velocity_ms', 'detrended_velocity_ms')

        self.convertMToCm('detrended_velocity_ms', 'detrended_velocity_cms')
        
        self.integrateDfColumn('detrended_velocity_ms', 'displacement_m')

        self.detrendData('displacement_m', 'detrended_displacement_m')

        self.butterHighpassFilter('detrended_displacement_m', 'highpassed_displacement_m')

        self.convertMToCm('highpassed_displacement_m', 'highpassed_displacement_cm')

        self.accRawStats = self.getStats('g')
        self.accOffsetStats = self.getStats('offset_g')
        self.accBandpassedStats = self.getStats('bandpassed_g')
        self.velStats = self.getStats('detrended_velocity_cms')
        self.dispStats = self.getStats('highpassed_displacement_cm')
        
        
    def logHeadTail(self):
        """print head and tail of self.df to console"""
        logging.debug(self.df.head())
        logging.debug(self.df.tail())
    
    
    def truncateDf(self):
        """
        truncate self.df based on timestamp for known time event
        args: 
        eventTimestamp: string holding timestamp (same as event dir name) 
        in format: '2020-01-13T163712'
        """
        # convert string timestamp to pandas Timestamp instance
        eventTimestamp = pd.Timestamp(self.eventTimestamp)
        # convert timestamp from local Turkish time to UTC
        eventTimestamp = eventTimestamp - pd.Timedelta('3 hours')
        startTime = eventTimestamp - pd.Timedelta('1 minute')
        endTime = startTime + pd.Timedelta('400 seconds')
        
        # see caveats in pandas docs on this!!
        self.df = self.df.loc[(self.df['timestamp'] > startTime) & (self.df['timestamp'] <= endTime)]
        logging.info('start time: {}'.format(startTime))
        logging.info('end time: {}'.format(endTime))
        self.logHeadTail()
        
        
    def setSensitivity(self):
        """
        return float holding sensitivity in V/g based on sensorCode
        """
        # CHECK AUTOPRO REPORT ON THIS - FF SHOULD GET SAME AS GROUND FLOOR
        groundFloorSensorCodes = [c for c in SENSOR_CODES if c.endswith('F')]
        upperFloorSensorCodes = [c for c in SENSOR_CODES if not c.endswith('F')]
        for code in groundFloorSensorCodes:
            if self.sensorCode.startswith(code):
                self.sensitivity = 1.25
        for code in upperFloorSensorCodes:
            if self.sensorCode.startswith(code):
                self.sensitivity = 0.625
        if self.sensitivity:
            return self.sensitivity
        else:
            raise ValueError('Non-null value must be assigned to sensitivity.')
        
            
    def convertCountToG(self):
        """
        add new column to df to hold acceleration in g (converted from 
        raw counts) per formula in Autopro report
        """
        self.df['g'] = self.df['count'] * (2.5/8388608) * (1/self.sensitivity)
        self.logHeadTail()
        
        
    def addZeroPad(self, padLength=500, location='both'):
        """
        add zeropad of given length at given location to necessary columns: timestamp and g
        """
        zeros = np.zeros(shape=(padLength))
        zeroPad = pd.Series(zeros)
        nullList = [None] * padLength
        nullSeries = pd.Series(nullList)
        if location == 'both':
            paddedG = pd.concat([zeroPad, self.df['g'], zeroPad])
            paddedTimestamp = pd.concat([nullSeries, self.df['timestamp'], nullSeries])
        elif location == 'tail':
            paddedG = pd.concat([self.df['g'], zeroPad])
            paddedTimestamp = pd.concat([self.df['timestamp'], nullSeries])
        paddedTimestamp.reset_index(drop=True, inplace=True)
        paddedG.reset_index(drop=True, inplace=True)
        paddedData = {'timestamp': paddedTimestamp, 'g': paddedG}
        self.df = pd.DataFrame(paddedData, columns=['timestamp','g'])
        
        
    def convertGToOffsetG(self):
        """add new column to df where mean of g has been removed"""
        #gMean = self.df['g'].mean(axis=0)
        #self.df['offset_g'] = self.df['g'] - gMean
        #print('mean of g: {0} subtracted from g for {1}'.format(gMean, self.sensorCodeWithChannel))
        
        # gives same result as above
        self.df['offset_g'] = detrend(self.df['g'], type='constant')
    
    
    def convertGToMetric(self):
        """add new column to df showing acceleration in m/s^2"""
        self.df['acc_ms2'] = self.df['offset_g'] * 9.80665
        self.logHeadTail()
    
    
    def butterBandpass(self):
        """
        return bandpass filter coefficients...
        from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
        return:
        b: numerator of filter
        a: denominator of filter
        """
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        #b, a = butter(self.order, low, btype='highpass')
        logging.info('butterworth coefficients - b: {0}, a: {1}'.format(b, a))
        return b, a
    
    
    def butterBandpassFilter(self, inputColumn, outputColumn):
        """
        create columns in df holding bandpassed acceleration data
        (apply bandpass filter to data using filter coefficients b and a)
        from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
        """
        b, a = self.butterBandpass()
        self.df[outputColumn] = lfilter(b, a, self.df[inputColumn])

    
    def integrateDfColumn(self, inputColumn, outputColumn):
        """integrate given dataframe column (given as string)"""
        inputList = list(self.df[inputColumn])
        inputListFromIndex1 = inputList[1:]
        zipped = zip(inputList, inputListFromIndex1)
        # values will be appended to integratedList
        integratedList = [0]
        for z in zipped:
            integrated = self.dt * (z[0] + z[1])/2. + integratedList[-1]
            integratedList.append(integrated)
        self.df[outputColumn] = integratedList


    def clipDf(self):
        self.df = self.df.iloc[self.ignoredSamples:40500].reset_index()
        
        
    def detrendData(self, inputColumn, outputColumn):
        """
        detrend data by removing mean
        (will get called after data is clipped so as not to include extreme values
        that would skew the mean)
        """
        #mean = df.iloc[self.ignoredSamples:40500].mean()
        #df[outputColumn] = df[inputColumn] - mean
        self.df[outputColumn] = detrend(self.df[inputColumn], type='constant')
    
    
    def butterHighpass(self):
        """
        return coefficients for highpass filter
        modeled after Butterworth bandpass code in scipy cookbook
        """
        nyq = 0.5 * self.fs
        cutoff = self.lowcut / nyq
        b, a = butter(self.order, cutoff, btype='high')
        return b, a
    
    
    def butterHighpassFilter(self, inputColumn, outputColumn):
        """
        modeled after Butterworth bandpass code in scipy cookbook
        """
        b, a = self.butterHighpass()
        self.df[outputColumn] = lfilter(b, a, self.df[inputColumn])
        
        
    def convertMToCm(self, inputColumn, outputColumn):
        """
        convert values in meters to values in centimeters
        """
        self.df[outputColumn] = self.df[inputColumn] * 100


    def getStats(self, columnName):
        """
        get min, max, mean and peak value of self.df
        (to be called after df has been clipped)
        columnName: string holding name of column in self.df
        return: tuple holding:
            1. string holding max and min (to be printed to console or on plots)
            2. float holding peak value (greater absolute value of min and max)
        """

        minVal = round(self.df[columnName].min(), 5)
        maxVal = round(self.df[columnName].max(), 5)
        meanVal = round(self.df[columnName].mean(), 5)
        peakVal = max(abs(minVal), abs(maxVal))

        logging.info('stats for {0}:{1}\n'.format(self.sensorCodeWithChannel, columnName))
        stats = 'min: {0}\nmax: {1}\nmean: {2}\n'.format(minVal, maxVal, meanVal)
        logging.info(stats)
        return stats, peakVal


    def plotGraph(self, column, titleSuffix, yLimit, padded=False):
        """
        plot graph of data and save graph
        column: string holding name of column used as y data
        plotTitle: string holding title of plot
        yLimit: float holding value to be used to set range of plot along y-axis (from negative yLimit to yLimit)
        padded: boolean - True if input data has pad (of 500 zeros on both ends)
        """
        plotTitle = self.sensorCodeWithChannel + ' ' + titleSuffix
        #if padded:
            # was originally set to display [500:40500]
        #plot = df.iloc[self.ignoredSamples:40500].reset_index().plot(kind='line', linewidth=1.0, x='index', y=column, color='red', title=plotTitle)
        #else:
        plot = self.df.reset_index().plot(kind='line', x='index', y=column, color='red', title=plotTitle, linewidth=1.0)
        fig = plot.get_figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(-yLimit, yLimit)
        # only text portion of stats will be shown on plot
        stats = self.getStats(column)[0]
        ax.annotate(stats, xy=(0.6, 0.65), xycoords='figure fraction')
        plotFilename = '_'.join([self.eventTimestamp, self.sensorCodeWithChannel, column]) + '.png'
        plotOutpath = os.path.join(self.plotDir, plotFilename)
        fig.savefig(plotOutpath)


# table holding peak values for acceleration, velocity, and displacement for each
# channel of each device (modeled after Safe Report)
class StatsTable:
    def __init__(self):
        self.columnHeaders = ['ID', 'acc_g_offset', 'acc_g_bandpassed', 'vel_cm_s', 'disp_cm']
        self.df = pd.DataFrame(columns=self.columnHeaders)
        self.df['ID'] = getAllSensorCodesWithChannels()


    def updateStatsDf(self, sensorCodeWithChannel, statsColumnName, value):
        """
        update stats dataframe where 'ID' equals sensorCodeWithChannel
        sensorCodeWithChannel: string holding sensor code with channel (from Conversion object)
        statsColumnName: string holding name of stats table dataframe column to be updated
        value: value to be set in statsDf (will come from stats of Conversion object)
        """
        #statsColumnName = self.columnDict[conversionColumnName]
        self.df.loc[self.df['ID'] == sensorCodeWithChannel, statsColumnName] = value


    def getColumnMax(self, statsColumnName):
        """
        get maximum value in given column
        statsColumnName: string holding name of stats table dataframe column from which to find maximum value
        """
        return self.df[statsColumnName].max()
        

# Client code
def main():
    """Main function."""

    ip = InputProcessing('2019-09-26T135930', '/home/grm/AllianzSHMS/working/test-mseed-files_20190926')
    #ip = InputProcessing('2020-01-11T163736', '/home/grm/AllianzSHMS/working/test-mseed-files_20200111')

    # if two files exist for each channel of each device, convert both files to df's and combine df's before conversion
    
    st = StatsTable()

    # maintain order of following three lists - will be zipped together to create plot args
    conversionColumnNames = ['offset_g', 'bandpassed_g', 'detrended_velocity_cms', 'highpassed_displacement_cm']
    statsColumnNames = ['acc_g_offset', 'acc_g_bandpassed', 'vel_cm_s', 'disp_cm']
    titleSuffixes = ['offset acceleration (g)', 'bandpassed acceleration (g)', 'detrended velocity (cm/s)', 'highpassed displacement (cm)']


    # replace c's and st's with conversionObject and statsTableObject given as args
    def updateStatsTable():
        """update row of stats table with max values at each of the (currently four) parameters"""
        accRawPeakVal = c.accRawStats[1]
        accOffsetPeakVal = c.accOffsetStats[1]
        accBandpassedPeakVal = c.accBandpassedStats[1]
        velPeakVal = c.velStats[1]
        dispPeakVal = c.dispStats[1]
        #st.updateStatsDf(c.sensorCodeWithChannel, 'acc_g_raw', accRawPeakVal)
        st.updateStatsDf(c.sensorCodeWithChannel, 'acc_g_offset', accOffsetPeakVal)
        st.updateStatsDf(c.sensorCodeWithChannel, 'acc_g_bandpassed', accBandpassedPeakVal)
        st.updateStatsDf(c.sensorCodeWithChannel, 'vel_cm_s', velPeakVal)
        st.updateStatsDf(c.sensorCodeWithChannel, 'disp_cm', dispPeakVal)


    def getStatsMaxValues(statsTableObject):
        """
        get max values from each column of the stats table (called only once after final stats table is completely populated)
        (ensure that items of statsColumnNames and conversionColumnNames are in the same order)
        statsTableObject: instance of StatsTable class
        return: list holding max values of each column in stats table
        """
        columnMaxValues = []
        for column in statsColumnNames:
            columnMaxValues.append(statsTableObject.getColumnMax(column))
        print('max values of {0}:\n{1}'.format(statsColumnNames, columnMaxValues))
        #statsDict = dict(zip(conversionColumnNames, columnMaxValues))
        return columnMaxValues


    def getPlotArgs(columnMaxValues):
        """
        columnMaxValues: list holding max values of each column in stats table
        return a list of tuples holding:
             string holding conversion column names
             string holding title suffix to be used for plot
             max value from each STATS column (which will be used to define range of ylimits of all plots for given parameter)
        """
        plotArgs = zip(conversionColumnNames, titleSuffixes, columnMaxValues)
        return plotArgs


    def drawPlots(conversionObject, columnMaxValues):
        """
        draw all (currently four) plots associated with given conversionObject 
        conversionObject: instance of the Conversion class
        """
        plotArgs = getPlotArgs(columnMaxValues)
        for item in plotArgs:
            columnName, titleSuffix, maxVal = item
            # add 1% of maxVal to maxVal to get range of plots along y-axis
            yLimit = maxVal + maxVal * 0.01
            # will probably need to create multiple df's in Conversion class for this
            conversionObject.plotGraph(columnName, titleSuffix, yLimit)


    def getConversionObjectFromTwoTxtFiles(txtFilePair):
        """
        txtFilePair: list of tuples containing text file name (and sensor code channel)
        return: conversion object made from dataframe from combined text files
        """
        txtFilePair = sorted([item[0] for item in txtFilePair])
        p1 = ProcessedFromTxtFile(txtFilePair[0])
        df1 = p1.df
        p2 = ProcessedFromTxtFile(txtFilePair[1])
        df2 = p2.df
        df = pd.concat([df1, df2])
        return Conversion(df, p1.sensorCode, p1.sensorCodeWithChannel, ip.eventTimestamp)


    def getConversionObjectFromOneTxtFile(txtFile):
        """
        txtFile: string holding name of text file produced from miniseed file
        return: conversion object made from dataframe produced from text file
        """
        p = ProcessedFromTxtFile(txtFile)
        df = p.df
        return Conversion(df, p.sensorCode, p.sensorCodeWithChannel, ip.eventTimestamp)


    # create Conversion objects twice: 
        # once to compile statistics into stats table
        # second time to create plots (using peak values from stats table as plot 
        # ranges so that each parameter uses same scale)
    if ip.pairedTxtFileList:
        for pair in ip.pairedTxtFileList:
            c = getConversionObjectFromTwoTxtFiles(pair)
            updateStatsTable()

        statsColumnMaxValues = getStatsMaxValues(st)

        for pair in ip.pairedTxtFileList:
            c = getConversionObjectFromTwoTxtFiles(pair)
            drawPlots(c, statsColumnMaxValues)
        
    else:
        for txtFile in ip.txtFileList:
            c = getConversionObjectFromOneTxtFile(txtFile)
            updateStatsTable()

        statsColumnMaxValues = getStatsMaxValues(st)

        for txtFile in ip.txtFileList:
            c = getConversionObjectFromOneTxtFile(txtFile)
            drawPlots(c, statsColumnMaxValues)

    

    print(st.df)
    statsDfCsvPath = '/home/grm/acc-data-conversion/working/no_ui/' + ip.eventTimestamp + '_stats.csv'
    st.df.to_csv(statsDfCsvPath)

    # Create an instance of QApplication
    #convertacc = QApplication(sys.argv)

    #model = AccConvertModel()
    #results = ResultsCanvas(model)
    # Show the application's GUI

    #view = AccConvertUi(model, results)
    
    #view.show()

    # Create the instances of the model and the controller
    #model = AccConvertModel()
    #AccConvertCtrl(model=model, view=view)

    #results._printPDF()
    # Execute the calculator's main loop
    #sys.exit(convertacc.exec_())

    #results._printPDF()



if __name__ == '__main__':
    main()