# Filename: convert_acc.py

"""Visualization and conversion of accelerometer data to velocity and displacement using Python and PyQt5."""

import sys
import os
import subprocess
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, detrend
import math
import logging

# Import QApplication and the required widgets from PyQt5.QtWidgets

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QFileInfo
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QErrorMessage

from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QButtonGroup
from PyQt5.QtWidgets import QRadioButton
from PyQt5.QtWidgets import QScrollArea

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


__version__ = '0.1'
__author__ = 'Nick LiBassi'

ERROR_MSG = 'ERROR'

# set logging to 'DEBUG' to print debug statements to console
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

"""
General Notes:
use camelCase throughout as Qt uses it
remove unused imports leftover from calculator app

moved functionality from Controller class to PrimaryUi class

can handle either 24 or 48 input miniseed files

need to fix B4 plots on ComparisonCanvases
"""

# move all helper functions into a public class?

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


def getFloor(sensorCodeText):
    """
    Return string holding only floor portion (only digits) of sensor code
    sensorCodeText: string holding either sensor code or sensor code with channel
    """
    floorChars = [c for c in sensorCodeText if c.isdigit()]
    floor = ''.join(floorChars)
    return floor


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


# class used to convert data (from single dataframe) from 
# count to acceleration, velocity, and displacement
class Conversion:
    def __init__(self, df, sensorCode, sensorCodeWithChannel, eventTimestamp):
        #self.ui = ui
        self.df = df
        self.sensorCode = sensorCode
        self.sensorCodeWithChannel = sensorCodeWithChannel
        self.floor = getFloor(self.sensorCode)
        self.eventTimestamp = eventTimestamp

        self.sensitivity = None
        self.accRawStats = None
        self.accOffsetStats = None
        self.accBandpassedStats = None
        self.velStats = None
        self.dispStats = None

        self.workingBaseDir = "/home/grm/acc-data-conversion/working"
        self.workingDirPath = os.path.join(self.workingBaseDir, self.eventTimestamp)
        #self.plotDir = "/home/grm/acc-data-conversion/working/no_ui/plots"

        # these may be taken from user in future
        # low cutoff frequency for bandpass and highpass filters
        self.lowcut = 0.05
        # high cutoff frequency for bandpass filter
        self.highcut = 40
        # sampling frequency
        self.fs = 100
        # order of filters
        self.order = 2
        # time between samples in seconds
        self.dt = 1/float(self.fs)

        self.ignoredSamples = 6000

        self.resultsSubplotDict = {'B4Fx': 19,
                             'B4Fy': 20,
                             'B4Fz': 21,
                             'FFN': 22,
                             'FFW': 23,
                             'FFZ': 24,
                             'N12x': 13,
                             'N12y': 14,
                             'N12z': 15,
                             'N24x': 7,
                             'N24y': 8,
                             'N24z': 9,
                             'N39x': 1,
                             'N39y': 2,
                             'N39z': 3,
                             'S12x': 16,
                             'S12y': 17,
                             'S12z': 18,
                             'S24x': 10,
                             'S24y': 11,
                             'S24z': 12,
                             'S39x': 4,
                             'S39y': 5,
                             'S39z': 6}

        # key '4' refers to floor 'B4'
        self.comparisonSubplotDict = {'39': 1, '24': 2, '12': 3, '4': 4}

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


    def plotResultsGraph(self, canvasObject, column, titleSuffix, yLimit):
        """
        plot graph of data to given ResultsCanvas object (one of 24 graphs on one page)
        canvasObject: instance of ResultsCanvas class
        column: string holding name of column used as y data
        plotTitle: string holding title of plot
        yLimit: float holding value to be used to set range of plot along y-axis (from negative yLimit to yLimit)
        """
        plotTitle = self.sensorCodeWithChannel + ' ' + titleSuffix

        subplotPos = self.resultsSubplotDict[self.sensorCodeWithChannel]

        ax = canvasObject.figure.add_subplot(8, 3, subplotPos)

        plot = self.df.reset_index().plot(kind='line', x='index', y=column, color='red', title=plotTitle, linewidth=1.0, ax=ax)

        ax.set_ylim(-yLimit, yLimit)
        ax.xaxis.set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))

        # only text portion of stats will be shown on plot
        #stats = self.getStats(column)[0]
        #ax.annotate(stats, xy=(0.6, 0.65), xycoords='figure fraction')

        canvasObject.figure.tight_layout()
        

    def plotComparisonGraph(self, canvasObject, yLimit):
        """
        plot displacement (for now) on a figure comparing displacements at different levels
        canvasObject: instance of ComparisonCanvas class
        yLimit: float holding value to be used to set range of plot along y-axis (from negative yLimit to yLimit)
        """
        subplotPos = self.comparisonSubplotDict[self.floor]
        ax = canvasObject.figure.add_subplot(4, 1, subplotPos)
        plot = self.df.reset_index().plot(kind='line', x='index', y='highpassed_displacement_cm', color='red', linewidth=1.0, ax=ax)
        ax.set_ylim(-yLimit, yLimit)
        ax.xaxis.set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        if self.floor == '39':
            ax.set_title(canvasObject.windowTitle)
        ax.set_ylabel(self.sensorCodeWithChannel, rotation=0)

        # only text portion of stats will be shown on plot
        #stats = self.getStats(column)[0]
        #ax.annotate(stats, xy=(0.6, 0.65), xycoords='figure fraction')

        canvasObject.figure.tight_layout()


# Create a subclass of QMainWindow to set up the portion of GUI 
# to take information from user and serve as controller of program
class PrimaryUi(QMainWindow):
    """AccConvert's initial view for taking input from user."""
    def __init__(self):
        """View initializer."""
        super().__init__()

        self.eventTimestamp = None
        self.miniseedDirPath = None
        self.miniseedFileList = None
        self.miniseedFileCount = None
        
        self.workingBaseDir = "/home/grm/acc-data-conversion/working"
        self.workingDirPath = None

        self.txtFileList = None
        self.txtFileCount = None

        self.pairedTxtFileList = []

        # ************moved from old Controller class
        self.statsTable = StatsTable()
        self.offsetGResultsCanvas = ResultsCanvas('Acceleration (g)')
        self.bandpassedGResultsCanvas = ResultsCanvas('Bandpassed Acceleration (g)')
        self.velResultsCanvas = ResultsCanvas('Velocity (cm/s)')
        self.dispResultsCanvas = ResultsCanvas('Displacement (cm)')
        # order of following four lists must remain as is
        # get stats column names except for 'ID'
        self.statsColumnNames = self.statsTable.columnHeaders[1:]
        self.conversionColumnNames = ['offset_g', 'bandpassed_g', 'detrended_velocity_cms', 'highpassed_displacement_cm']
        self.titleSuffixes = ['offset acceleration (g)', 'bandpassed acceleration (g)', 'detrended velocity (cm/s)', 'highpassed displacement (cm)']
        self.resultsCanvases = [self.offsetGResultsCanvas, self.bandpassedGResultsCanvas, self.velResultsCanvas, self.dispResultsCanvas]
        # ************
        # self.statsColumnMaxValues will hold peak values in same order as self.conversionColumnNames 
        self.statsColumnMaxValues = None

        self.NXcomparisonCanvas = ComparisonCanvas('N. Corner X-Dir (cm)')
        self.NYcomparisonCanvas = ComparisonCanvas('N. Corner Y-Dir (cm)')
        self.SXcomparisonCanvas = ComparisonCanvas('S. Corner X-Dir (cm)')
        self.SYcomparisonCanvas = ComparisonCanvas('S. Corner Y-Dir (cm)')

        self.allCanvases = self.resultsCanvases + [self.NXcomparisonCanvas, self.NYcomparisonCanvas, self.SXcomparisonCanvas, self.SYcomparisonCanvas]

        # Set some of main window's properties
        self.setWindowTitle('Accelerometer Data Conversion')
        self.setFixedSize(500, 300)
        # Set the central widget and the general layout
        self.generalLayout = QVBoxLayout()
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.centralWidget.setLayout(self.generalLayout)
        
        # Create the display and the buttons
        self.createTextInputFields()
        self.createRadioButtons()
        self.createSubmitButton()


    def getEventTimestamp(self):
        """Get user input (string) for event id field"""
        logging.debug(self.eventField.text())
        self.eventTimestamp = self.eventField.text()


    def getMiniseedDirPath(self):
        """Get user input (string) for miniseed directory path"""
        self.miniseedDirPath = self.miniseedDirField.text()


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


    def createTextInputFields(self):
        """Create text input fields"""
        self.eventLabel = QLabel(self)
        self.eventLabel.setText('Create event id ex. "2020-01-11T163736"')
        self.miniseedDirLabel = QLabel(self)
        self.miniseedDirLabel.setText('Path of directory holding miniseed files')
        self.eventField = QLineEdit(self)
        self.miniseedDirField = QLineEdit(self)
        self.generalLayout.addWidget(self.eventLabel)
        self.generalLayout.addWidget(self.eventField)
        self.generalLayout.addWidget(self.miniseedDirLabel)
        self.generalLayout.addWidget(self.miniseedDirField)


    def createRadioButtons(self):
        self.approachGroup = QButtonGroup(self.centralWidget)
        self.timeAutopro = QRadioButton('Time-domain per Autopro report')
        self.timeAutopro.setChecked(True)
        self.timeBaseline = QRadioButton('Time-domain with baseline correction')
        self.freqCorrection = QRadioButton('Frequency-domain with correction filter')
        self.freqPlain = QRadioButton('Frequency-domain without correction filter')

        self.approachGroup.addButton(self.timeAutopro)
        self.approachGroup.addButton(self.timeBaseline)
        self.approachGroup.addButton(self.freqCorrection)
        self.approachGroup.addButton(self.freqPlain)

        self.generalLayout.addWidget(self.timeAutopro)
        self.generalLayout.addWidget(self.timeBaseline)
        self.generalLayout.addWidget(self.freqCorrection)
        self.generalLayout.addWidget(self.freqPlain)


    def setTxtFileInfo(self):
        """Set list of sorted input text file paths and number of text files"""
        self.txtFileList = [os.path.join(self.workingDirPath, f) for f in os.listdir(self.workingDirPath)]
        self.txtFileList = sortFiles(self.txtFileList)
        self.txtFileCount = len(self.txtFileList)
        for path in self.txtFileList:
            logging.debug('txt file path: {0}'.format(path))


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


    #*************************
    def updateStatsTable(self, conversionObject):
        """update row of stats table with max values at each of the (currently four) parameters"""
        accRawPeakVal = conversionObject.accRawStats[1]
        accOffsetPeakVal = conversionObject.accOffsetStats[1]
        accBandpassedPeakVal = conversionObject.accBandpassedStats[1]
        velPeakVal = conversionObject.velStats[1]
        dispPeakVal = conversionObject.dispStats[1]

        self.statsTable.updateStatsDf(conversionObject.sensorCodeWithChannel, 'acc_g_offset', accOffsetPeakVal)
        self.statsTable.updateStatsDf(conversionObject.sensorCodeWithChannel, 'acc_g_bandpassed', accBandpassedPeakVal)
        self.statsTable.updateStatsDf(conversionObject.sensorCodeWithChannel, 'vel_cm_s', velPeakVal)
        self.statsTable.updateStatsDf(conversionObject.sensorCodeWithChannel, 'disp_cm', dispPeakVal)


    def getStatsMaxValues(self):
        """
        get max values from each column of the stats table (called only once after final stats table is completely populated)
        (ensure that items of statsColumnNames and conversionColumnNames are in the same order)
        return: list holding max values of each column in stats table
        """
        columnMaxValues = []
        for column in self.statsColumnNames:
            columnMaxValues.append(self.statsTable.getColumnMax(column))
        print('max values of {0}:\n{1}'.format(self.statsColumnNames, columnMaxValues))
        #statsDict = dict(zip(conversionColumnNames, columnMaxValues))
        return columnMaxValues


    def getPlotArgs(self):
        """
        return a list of tuples holding:
             string holding conversion column names
             string holding title suffix to be used for plot
             max value from each STATS column (which will be used to define range of ylimits of all plots for given parameter)
        """
        plotArgs = zip(self.resultsCanvases, self.conversionColumnNames, self.titleSuffixes, self.statsColumnMaxValues)
        return plotArgs


    def drawResultsPlots(self, conversionObject):
        """
        draw all (currently four) plots associated with given conversionObject 
        conversionObject: instance of the Conversion class
        """
        plotArgs = self.getPlotArgs()
        for item in plotArgs:
            canvas, columnName, titleSuffix, maxVal = item
            # add 1% of maxVal to maxVal to get range of plots along y-axis
            yLimit = maxVal + maxVal * 0.01
            # will probably need to create multiple df's in Conversion class for this (??)
            conversionObject.plotResultsGraph(canvas, columnName, titleSuffix, yLimit)


    def drawComparisonPlot(self, conversionObject):
        """
        add subplot of displacement to ComparisonCanvas if self.sensorCodeWithChannel meets criteria
        """
        displacementMaxVal = self.statsColumnMaxValues[-1]
        if all(i in conversionObject.sensorCodeWithChannel for i in ['N', 'x']):
            conversionObject.plotComparisonGraph(self.NXcomparisonCanvas, displacementMaxVal)
        elif all(i in conversionObject.sensorCodeWithChannel for i in ['S', 'x']):
            conversionObject.plotComparisonGraph(self.SXcomparisonCanvas, displacementMaxVal)
        elif all(i in conversionObject.sensorCodeWithChannel for i in ['N', 'y']):
            conversionObject.plotComparisonGraph(self.NYcomparisonCanvas, displacementMaxVal)
        elif all(i in conversionObject.sensorCodeWithChannel for i in ['S', 'y']):
            conversionObject.plotComparisonGraph(self.SYcomparisonCanvas, displacementMaxVal)

        # modify this so these appear on the x or y canvases no matter what (will be repeated)
        elif all(i in conversionObject.sensorCodeWithChannel for i in ['B', 'x']):
            conversionObject.plotComparisonGraph(self.SYcomparisonCanvas, displacementMaxVal)
        elif all(i in conversionObject.sensorCodeWithChannel for i in ['B', 'y']):
            conversionObject.plotComparisonGraph(self.SYcomparisonCanvas, displacementMaxVal)


    def getConversionObjectFromTwoTxtFiles(self, txtFilePair):
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
        return Conversion(df, p1.sensorCode, p1.sensorCodeWithChannel, self.eventTimestamp)            


    def getConversionObjectFromOneTxtFile(self, txtFile):
        """
        txtFile: string holding name of text file produced from miniseed file
        return: conversion object made from dataframe produced from text file
        """
        p = ProcessedFromTxtFile(txtFile)
        df = p.df
        return Conversion(df, p.sensorCode, p.sensorCodeWithChannel, self.eventTimestamp)


    def showCanvases(self):
        for canvas in self.allCanvases:
            canvas.show()


    def getResults(self):
        """
        perform conversions for all datasets (24 datasets from 24 or 48 text files)
        call drawResultsPlots() to plot acceleration, velocity, and displacement for all datasets 
        """
        if self.pairedTxtFileList:
            for pair in self.pairedTxtFileList:
                c = self.getConversionObjectFromTwoTxtFiles(pair)

                # export df to csv if necessary
                #outCsv = os.path.join(c.workingDirPath, c.sensorCodeWithChannel + '.csv')
                #c.df.to_csv(outCsv)

                self.updateStatsTable(c)
            self.statsColumnMaxValues = self.getStatsMaxValues()

            for pair in self.pairedTxtFileList:
                c = self.getConversionObjectFromTwoTxtFiles(pair)
                self.drawResultsPlots(c)
                self.drawComparisonPlot(c)

            self.showCanvases()

        else:
            for txtFile in self.txtFileList:
                c = self.getConversionObjectFromOneTxtFile(txtFile)
                self.updateStatsTable(c)
            self.statsColumnMaxValues = self.getStatsMaxValues()

            for txtFile in self.txtFileList:
                c = self.getConversionObjectFromOneTxtFile(txtFile)
                self.drawResultsPlots(c)
                self.drawComparisonPlot(c)

            self.showCanvases()


    #****************************

    def processUserInput(self):
        self.getEventTimestamp()
        self.getMiniseedDirPath()
        self.setMiniseedFileInfo()
        self.setWorkingDir()
        self.convertMiniseedToAscii()
        self.setTxtFileInfo()
        self.pairDeviceTxtFiles()
        self.getResults()


    def createSubmitButton(self):
        """Create single submit button"""
        self.submitBtn = QPushButton(self)
        self.submitBtn.setText("Submit")
        # using only Autopro approach now - add others later
        self.submitBtn.clicked.connect(self.processUserInput)
        self.generalLayout.addWidget(self.submitBtn)


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



# Comparison Figure objects were going to be used to display four plots but ...
class ComparisonFigure(Figure):
    def __init__(self, title, width=14, height=20, dpi=100):
        Figure.__init__(self)
        self.title = title
        self.figsize = (width, height)
        self.dpi = dpi


class BaseCanvas(QMainWindow):
    def __init__(self, windowTitle, width=14, height=20, dpi=100):
        QMainWindow.__init__(self)
        self.windowTitle = windowTitle
        self.setWindowTitle(self.windowTitle)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.vbox = QVBoxLayout()
        self.widget.setLayout(self.vbox)
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.canvas = FigureCanvas(self.figure)

        self.scroll = QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)    


# ComparisonCanvas objects used to display four plots 
# wanted to show four ComparisonFigure objects (with four plots each) but seems that
# each FigureCanvas can only have one Figure object
class ComparisonCanvas(BaseCanvas):
    def __init__(self, windowTitle):
        BaseCanvas.__init__(self, windowTitle, width=5, height=10)
        


# ResultsCanvas objects used to display plots for all 24 devices/channels
# Class has one instance each of:
#    Figure (which can hold multiple subplots)
#    FigureCanvas (which holds the figure) 
class ResultsCanvas(BaseCanvas):
    def __init__(self, windowTitle):
        BaseCanvas.__init__(self, windowTitle)



# Client code
def main():
    """Main function."""
    # Create an instance of QApplication
    convertacc = QApplication(sys.argv)

    # Show the application's GUI
    view = PrimaryUi()
    view.show()

    # Execute the program's main loop
    sys.exit(convertacc.exec_())

    #results._printPDF()


if __name__ == '__main__':
    main()