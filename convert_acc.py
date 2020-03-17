# Filename: convert_acc_qt.py

"""Conversion of accelerometer data to velocity and displacement using Python and PyQt5."""

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
General Notes:
use camelCase throughout as Qt uses it
remove unused imports leftover from calculator app
"""

# move all helper functions into a public class?

def getAllSensorCodesWithChannels():
    sensorCodePrefixes = ['N39', 'S39', 'N24', 'S24', 'N12', 'S12', 'B4F', 'FF']
    axes = ['x', 'y', 'z']
    allSensorCodesWithChannels = []
    for p in sensorCodePrefixes:
        for a in axes:
            allSensorCodesWithChannels.append(p + a)
    return allSensorCodesWithChannels


def sortTxtFiles(inputTxtFileList):
    """
    Sort given text files according to Safe report.
    (may need to be modified later for events that span two hours)
    """
    sortedList = []
    sensorCodePrefixes = ['N39', 'S39', 'N24', 'S24', 'N12', 'S12', 'B4F', 'FF']
    for prefix in sensorCodePrefixes:
        fileGroup = [f for f in inputTxtFileList if prefix in f]
        sortedFileGroup = sorted(fileGroup)
        sortedList += sortedFileGroup
    return sortedList


# not to be confused with _getSensorCode()
def getSensorCodeWithChannel(inputTxtFile):
    """
    Return string holding both sensor code and channel e.g. 'B4Fx'
    args:
    inputTxtFile: string holding either text filename or path 
    """
    inputTxtFileBase = inputTxtFile.split('.txt')[0]
    # info in filenames are separated by dots
    sensorCodeWithChannel = inputTxtFileBase.rsplit('.')[-1]
    return sensorCodeWithChannel


def getStats(df, columnName):
    minVal = round(df[columnName].min(), 5)
    maxVal = round(df[columnName].max(), 5)
    meanVal = round(df[columnName].mean(), 5)
    #stats = 'min: {0}\nmax: {1}\nmean: {2}\n'.format(minVal, maxVal, meanVal)
    stats = 'min: {0}\nmax: {1}'.format(minVal, maxVal)
    return stats


def printWidget(widget, filename):
    """
    print widget function modified from:
    https://stackoverflow.com/questions/57286334/export-widget-in-pdf-file
    """
    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(filename)
    painter = QPainter(printer)

    # start scale
    xscale = printer.pageRect().width() * 1.0 / widget.width()
    yscale = printer.pageRect().height() * 1.0 / widget.height()
    scale = min(xscale, yscale)
    painter.translate(printer.paperRect().center())
    painter.scale(scale, scale)
    painter.translate(-widget.width() / 2, -widget.height() / 2)
    # end scale

    widget.render(painter)
    painter.end()


class OrganizedDfFromTxtFile:
    def __init__(self, txtFilePath):
        self.txtFilePath = txtFilePath
        #self.df = None
        #self.headerList = None
        #self.sensorCode = None

        self._convertTxtToDf()
        self._getHeaderList()
        self._getSensorCode()
        self._getDfWithTimestampedCounts()


    def _convertTxtToDf(self):
        """Convert given text file to pandas dataframe"""
        self.df = pd.read_csv(inputTxtFilePath, header=0)


    def _getHeaderList(self):
        self.headerList = [item for item in list(self.df.columns.values)]


    def _getSensorCode(self):
        """
        Return string holding sensor code for given df
        (single sensor code exists for each input text file)
        """
        locationCode = [header for header in self.headerList if '__' in header][0]
        self.sensorCode = locationCode.split('__')[1]


    def _getEarliestTimestamp(self):
        """
        get earliest pandas timestamp from among column headers 
        (which will contain either one or two timestamps)
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


    def _getCountColumnHeader(self):
        """
        get column header that contains count values 
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


    def _getDfWithTimestampedCounts(self):
        """return clean dataframe with only timestamp and count columns"""
        startTime = self.getEarliestTimestamp()
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



# Create model used to access and convert data
class AccConvertModel:
    def __init__(self):
        
        self.sensitivity = None

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


    def truncateDf(self, eventTimestamp):
        """
        return truncated dataframe based timestamp for known time event
        args: 
        eventTimestamp: string holding timestamp (same as event dir name) 
        in format: '2020-01-13T163712'
        ret:
        self.df
        """
        # convert string timestamp to pandas Timestamp instance
        eventTimestamp = pd.Timestamp(eventTimestamp)
        # convert timestamp from local Turkish time to UTC-3
        eventTimestamp = eventTimestamp - pd.Timedelta('3 hours')
        startTime = eventTimestamp - pd.Timedelta('1 minute')
        endTime = startTime + pd.Timedelta('400 seconds')

        # see caveats in pandas docs on this!!
        self.df = self.df.loc[(self.df['timestamp'] > startTime) & (self.df['timestamp'] <= endTime)]
        logging.debug(startTime)
        logging.debug(endTime)
        return self.df


    def getSensitivity(self):
        """
        return float holding sensitivity in V/g based
        on first few letters of given sensorCode
        """
        groundFloorSensorCodes = ['B4F', 'FF']
        upperFloorSensorCodes = ['S12', 'N12', 'S24', 'N24', 'S39', 'N39']
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
        # making a copy may or may not be helpful here 
        # (no need to access original so may not matter anyway)
        # but got rid of pandas 'caveats' message
        self.df = self.df.copy()
        self.df['g'] = self.df['count'] * (2.5/8388608) * (1/self.sensitivity)
        return self.df


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
        return self.df


    def convertGToOffsetG(self):
        """add new column to df where mean of g has been removed"""
        gMean = self.df['g'].mean(axis=0)
        self.df['offset_g'] = self.df['g'] - gMean
        return self.df


    def convertGToMetric(self):
        """add new column to df showing acceleration in m/s^2"""
        self.df['acc_ms2'] = self.df['offset_g'] * 9.80665
        return self.df


    def removeZeropad(self, padLength=500, location='both'):
        """
        preserve only data between pad lengths added to both sides 
        i.e. with pad_length=500, preserve only rows between first 500 and last 500 rows
        args:
        padLength: int holding size of pad added before processing
        location: string holding 'tail' or 'both'
        """
        endRow = len(self.df) - padLength
        self.df = self.df.copy()
        if location == 'both':
            startRow = padLength
            self.df = pd.DataFrame(self.df.iloc[startRow:endRow])
        elif location == 'tail':
            self.df = pd.DataFrame(self.df.iloc[:endRow])
        return self.df


    def _butterBandpass(self):
        """
        from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
        """
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return b, a


    def butterBandpassFilter(self):
        """
        from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
        """
        b, a = self._butterBandpass()
        self.df['bandpassed_g'] = lfilter(b, a, self.df['offset_g'])
        self.df['bandpassed_ms2'] = lfilter(b, a, self.df['acc_ms2'])
        return self.df


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
        return self.df


    # MAY need to detrend data without the zero padding
    # type can be 'linear' (linear regression) or 'constant' (mean subtracted)
    def detrendData(self, inputColumn, outputColumn):
        self.df[outputColumn] = detrend(self.df[inputColumn], type='linear')


    def _butterHighpass(self):
        """
        modeled after Butterworth bandpass code in scipy cookbook
        """
        nyq = 0.5 * self.fs
        cutoff = self.lowcut / nyq
        b, a = butter(self.order, cutoff, btype='high')
        return b, a


    def butterHighpassFilter(self):
        """
        modeled after Butterworth bandpass code in scipy cookbook
        """
        pass
        #b, a = self._butterHighpass()
        #self.df['name of displacement column']


# Create a subclass of QMainWindow to set up the portion of GUI 
# to take information from user
class AccConvertUi(QMainWindow):
    """AccConvert's initial view for taking input from user."""
    def __init__(self, model, results):
        """View initializer."""
        super().__init__()

        self._model = model
        # results will be shown in second window
        self._results = results
        
        # Set some of main window's properties
        self.setWindowTitle('Accelerometer Data Conversion')
        self.setFixedSize(500, 300)
        # Set the central widget and the general layout
        self.generalLayout = QVBoxLayout()
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        
        # Create the display and the buttons
        self._createTextInputFields()
        self._createRadioButtons()
        self._createSubmitButton()

        # Call other initial methods
        self._getEventId()
        self._getMiniseedDirPath()
        self._getMiniseedFileInfo()
        self._getDeviceDatasetCount()

        self._getWorkingDir()
        self._convertMiniseedToAscii()
        self._setInputTxtFileInfo()


    def _getEventId(self):
        """Get user input (string) for event id field"""
        logging.debug(self.eventField.text())
        self.eventId = self.eventField.text()


    def _getMiniseedDirPath(self):
        """Get user input (string) for miniseed directory path"""
        self.miniseedDirPath = self.miniseedDirField.text()


    def _getMiniseedFileInfo(self):
        """Get number of input miniseed files"""
        self.miniseedFileList = [f for f in os.listdir(self.miniseedDirPath) if f.endswith(".m")]
        self.miniseedFileCount = len(self.miniseedFileList)


    def _pairDeviceMiniseedFiles(self):
        """
        If seismic event spans two hours, create list of 24 two-item lists holding path pairs of miniseed files to be processed.
        Assume number of input miniseed files to be 24 (if event falls in single hour) or 48 (if event spans two hours)
        """
        if self.miniseedFileCount == 48:
            miniseedSensorCodeList = []
            for file in self.miniseedFileList:
                sensorCodeWithChannel = getSensorCodeWithChannel(file)
                miniseedSensorCodeList.append((file, sensorCodeWithChannel))

            self.pairedMiniseedList = []
            allSensorCodesWithChannels = getAllSensorCodesWithChannels()
            for c in allSensorCodesWithChannels:
                pairedList = [f for f in miniseedSensorCodeList if c in f]
                self.pairedMiniseedList.append(pairedList)

        elif self.miniseedFileCount != 24:
            inputCountErrorDialog = QErrorMessage()
            inputCountErrorDialog.showMessage('Number of input files must be 24 or 48.')



    # transferred from Model
    def _convertMiniseedToAscii(self):
        """
        Convert all miniseed files in given dir to ascii files 
        args:
        mseed_dir_path_entry: instance of tkinter Entry class holding path of dir holding miniseed files entered by user
        working_dir_entry: string holding name of working directory for event (must be in timestamp form)
        ret: None
        """
        
        #self.miniseedDirPath = self._getMiniseedDirPath()
        self.workingDirPath = self._getWorkingDir()
        
        for f in self.miniseedFileList:
            self.basename = f.rsplit(".m")[0]
            self.filename = self.basename + ".txt"
            self.outPath = os.path.join(self.workingDirPath, self.filename)
            # path below hard-coded only for now - will need to be modified - must have both mseed files and mseed2ascii executable
            #os.chdir("/home/grm/AllianzSHMS/working/test-mseed-files")
            os.chdir(self.miniseedDirPath)
            subprocess.run(["./mseed2ascii", f, "-o", self.outPath])


    # transferred from Model
    def _getWorkingDir(self):
        """
        If not yet created, create working dir using event id entered
        by user. return string holding path to working dir for event
        """
        
        #self.eventId = ""
        #self.eventId = self._getEventId()
        # hard-coded for now
        self.workingBaseDir = "/home/grm/acc-data-conversion/working"
        self.workingDirPath = os.path.join(self.workingBaseDir, self.eventId)

        if not os.path.isdir(self.workingDirPath):
            os.mkdir(self.workingDirPath)
        
        return self.workingDirPath

        
    def _createTextInputFields(self):
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


    def _createRadioButtons(self):
        self.approachGroup = QButtonGroup(self._centralWidget)
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


    def _setInputTxtFileInfo(self):
        """Set list of input text file paths and number of text files"""
        self._inputTxtFilePaths = [os.path.join(self.workingDirPath, f) for f in os.listdir(self.workingDirPath)]
        # may not need self._inputTxtFileCount
        self._inputTxtFileCount = len(self._inputTxtFilePaths)


    def _setPlotDict(self, sortedTxtFilePaths):
        """
        Set dictionary with the following:
        keys: positions (int) of plots
        values: tuple in form: (path, sensor codes with channels (used as subplot titles))
        """
        #txtFileCount = len(sortedTxtFilePaths)
        #txtFilePositions = list(range(1, self._inputTxtFileCount + 1))
        # hardcoding number of plots to 24 for now
        txtFilePositions = list(range(1, 25))
        figureTitles = [getSensorCodeWithChannel(path) for path in sortedTxtFilePaths]
        zippedPathsTitles = zip(sortedTxtFilePaths, figureTitles)
        self.plotDict = dict(zip(txtFilePositions, zippedPathsTitles))


    # manipulate data per Autopro report
    # move code shared with other approaches to separate method later
    def _mainAutopro(self):
        logging.debug("mainAutopro run")

        self._inputTxtFilePaths = sortTxtFiles(self._inputTxtFilePaths)
        
        self._setPlotDict(inputTxtFilePaths)


        for position, pathTitlePair in self.plotDict.items():

            # make this attribute of view class?
            #figureTitle = getSensorCodeWithChannel(file)

            # set model attributes
            self._model.df = convertTxtToDf(pathTitlePair[0])
            self._model.getHeaderList()

            logging.debug(self._model.df.head())
            logging.debug(self._model.headerList)

            testSensorCode = self._model.getSensorCode()
            logging.debug(testSensorCode)

            self._model.getDfWithTimestampedCounts()
            logging.debug(self._model.df.head())

            self._model.truncateDf(self.eventId)
            logging.debug(self._model.df.head())

            self._model.sensitivity = self._model.getSensitivity()
            logging.debug(self._model.sensitivity)

            logging.debug(len(self._model.df))

            self._model.convertCountToG()

            # accept default values to add 500 zeros to each side of g data
            self._model.addZeroPad()

            self._model.convertGToOffsetG()

            self._model.convertGToMetric()

            logging.debug(self._model.df.head())
            logging.debug(self._model.df.tail())

            logging.debug(self._model.getHeaderList())

            #stats = self._model.getStats('g')

            # plot acceleration
            self._results.createSubplot('offset_g', 0.008, position, pathTitlePair[1])

            # get 'bandpassed_g' and 'bandpassed_ms2'
            self._model.butterBandpassFilter()

            self._model.integrateDfColumn('bandpassed_ms2', 'velocity_ms')

            self._model.detrendData('velocity_ms', 'detrended_velocity_ms')

            #self._model.removeZeropad()

            # plot velocity
            self._results.createSubplot('detrended_velocity_ms', 0.005, position + 24, pathTitlePair[1])
            
            
            # add displacement
            self._model.integrateDfColumn('detrended_velocity_ms', 'displacement_m')

            self._model.detrendData('displacement_m', 'detrended_displacement_m')

            self._model.df['detrended_displacement_cm'] = self._model.df['detrended_displacement_m'] * 100

            # plot displacement
            self._results.createSubplot('detrended_displacement_cm', 0.001, position + 48, pathTitlePair[1])
            
            self._results.show()


    def _createSubmitButton(self):
        """Create single submit button"""
        self.submitBtn = QPushButton(self)
        self.submitBtn.setText("Submit")
        # using only Autopro approach now - add others later
        self.submitBtn.clicked.connect(self._mainAutopro)
        self.generalLayout.addWidget(self.submitBtn)



# Create canvas to display results by subclassing FigureCanvasQTAgg
class ResultsCanvas(QMainWindow):
    def __init__(self, model, width=14, height=20, dpi=100):
        
        #figure = Figure(figsize=(width, height), dpi=dpi, tight_layout={'w_pad': 0.25, 'h_pad': 0.25})
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        #self.figure.tight_layout()
        self._model = model
        #self.axes = figure.add_subplot(111)
        #self.axes.xaxis.set_major_locator(plt.MaxNLocator(4))

        QMainWindow.__init__(self)
        #FigureCanvas.__init__(self, self.figure)
        #self.createSubplot()

        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.vbox = QVBoxLayout()
        self.widget.setLayout(self.vbox)
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.canvas = FigureCanvas(self.figure)
        self.canvas.draw()

        self.scroll = QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        #self.draw()

        #self._printPDF()


    def createSubplot(self, yData, yLimit, subplotPos, figTitle):
        """
        Plot figure
        args:
        yData: string holding name of dataframe column to be plotted
        subplotPos: integer holding position of plot
        """
        ax = self.figure.add_subplot(24, 3, subplotPos)
        # y_lim values will be set dynamically per event
        ax.set_ylim(-yLimit, yLimit)
        ax.xaxis.set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        dfCopy = self._model.df.copy()
        stats = getStats(dfCopy, yData)
        logging.debug('stats for {0}: {1}'.format(figTitle, stats))
        # text location will be set dynamically per event
        ax.text(20000, -0.005, stats)
        # plot all data except for zero pads - edit later if necessary
        self._model.df.iloc[500:40500].reset_index().plot(kind='line', x='index', y=yData, title=figTitle, ax=ax)
        self.figure.tight_layout()
        del dfCopy
        #self.draw()


    # might not be necessary
    def _printPDF(self):
        """Print results to pdf"""
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export PDF", None, "PDF files (.pdf);;All Files()"
        )
        if fn:
            if QFileInfo(fn).suffix() == "":
                fn += ".pdf"

            printWidget(self, fn)



# Client code
def main():
    """Main function."""
    # Create an instance of QApplication
    convertacc = QApplication(sys.argv)

    model = AccConvertModel()
    results = ResultsCanvas(model)
    # Show the application's GUI

    view = AccConvertUi(model, results)
    
    view.show()

    # Create the instances of the model and the controller
    #model = AccConvertModel()
    #AccConvertCtrl(model=model, view=view)

    #results._printPDF()
    # Execute the calculator's main loop
    sys.exit(convertacc.exec_())

    #results._printPDF()



if __name__ == '__main__':
    main()