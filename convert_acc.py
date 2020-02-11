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

from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QButtonGroup
from PyQt5.QtWidgets import QRadioButton

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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


# not to be confused with getSensorCode() in model class
def getSensorCodeWithChannel(inputTxtFile):
    """Return string holding both sensor code and channel e.g. 'B4Fx'"""
    inputTxtFileBase = inputTxtFile.split('.txt')[0]
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


# Create model used to access and convert data
class AccConvertModel:
    def __init__(self):
        self.df = None
        self.headerList = None
        self.sensorCode = None
        self.sensitivity = None


    def getHeaderList(self):
        self.headerList = [item for item in list(self.df.columns.values)]
        return self.headerList


    def getSensorCode(self):
        """
        return string holding sensor code for given df
        (single sensor code exists for each input text file)
        """
        locationCode = [header for header in self.headerList if '__' in header][0]
        self.sensorCode = locationCode.split('__')[1]
        return self.sensorCode


    def getEarliestTimestamp(self):
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


    def getCountColumnHeader(self):
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


    def getDfWithTimestampedCounts(self):
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
        return self.df


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

        #self._sensorCodeWithChannel = None


    def _getEventId(self):
        """Get user input (string) for event id field"""
        print(self.eventField.text())
        return self.eventField.text()


    def _getMiniseedDirPath(self):
        """Get user input (string) for miniseed directory path"""
        return self.miniseedDirField.text()


    # was in Model
    def _convertMseedToAscii(self):
        """
        convert all miniseed files in given dir to ascii files 
        args:
        mseed_dir_path_entry: instance of tkinter Entry class holding path of dir holding miniseed files entered by user
        working_dir_entry: string holding name of working directory for event (must be in timestamp form)
        ret: None
        """
        self.miniseedDirPath = ""
        self.miniseedDirPath = self._getMiniseedDirPath()
        self.mseedFileList = [f for f in os.listdir(self.miniseedDirPath) if f.endswith(".m")]
        self.workingDirPath = self._getWorkingDir()
        
        for f in self.mseedFileList:
            self.basename = f.rsplit(".m")[0]
            self.filename = self.basename + ".txt"
            self.outPath = os.path.join(self.workingDirPath, self.filename)
            # path below hard-coded only for now - will need to be modified - must have both mseed files and mseed2ascii executable
            #os.chdir("/home/grm/AllianzSHMS/working/test-mseed-files")
            os.chdir(self.miniseedDirPath)
            subprocess.run(["./mseed2ascii", f, "-o", self.outPath])


    # was in Model
    def _getWorkingDir(self):
        """
        If not yet created, create working dir using event id entered
        by user. return string holding path to working dir for event
        """
        
        self.eventId = ""
        self.eventId = self._getEventId()
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


    def _convertTxtToDf(self, inputTxtFile):
        """convert given text file to pandas dataframe"""
        self.rawDf = pd.read_csv(inputTxtFile, header=0)
        return self.rawDf


    def _getInputTxtFilePaths(self):
        """return list of input text file paths"""
        return [os.path.join(self.workingDirPath, f) for f in os.listdir(self.workingDirPath)]





    # manipulate data per Autopro report
    # move code shared with other approaches to separate method later
    def _mainAutopro(self):
        logging.debug("mainAutopro run")
        self._getWorkingDir()
        self._convertMseedToAscii()
        
        inputTxtFilePaths = self._getInputTxtFilePaths()
        inputTxtFilePaths = sortTxtFiles(inputTxtFilePaths)

        # move to separate function later?
        txtFileCount = len(inputTxtFilePaths)
        txtFilePositions = list(range(1, txtFileCount + 1))
        txtFileDict = dict(zip(txtFilePositions, inputTxtFilePaths))

        for position, file in txtFileDict.items():

            # make this attribute of view class?
            figureTitle = getSensorCodeWithChannel(file)

            # set model attributes
            self._model.df = self._convertTxtToDf(file)
            self._model.headerList = self._model.getHeaderList()

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

            self._model.removeZeropad()

            #stats = self._model.getStats('g')

            self._results.plotFig('offset_g', position, figureTitle)

            self._results.show()


    def _createSubmitButton(self):
        """Create single submit button"""
        self.submitBtn = QPushButton(self)
        self.submitBtn.setText("Submit")
        # using only Autopro approach now - add others later
        self.submitBtn.clicked.connect(self._mainAutopro)
        self.generalLayout.addWidget(self.submitBtn)





# Create canvas to display results by subclassing FigureCanvasQTAgg
class ResultsCanvas(FigureCanvas):
    def __init__(self, model, width=14, height=20, dpi=100):
        
        #figure = Figure(figsize=(width, height), dpi=dpi, tight_layout={'w_pad': 0.25, 'h_pad': 0.25})
        figure = Figure(figsize=(width, height), dpi=dpi)
        self._model = model
        #self.axes = figure.add_subplot(111)
        #self.axes.xaxis.set_major_locator(plt.MaxNLocator(4))

        FigureCanvas.__init__(self, figure)
        #self.plotFig()

        self.draw()

        #self._printPDF()


    def plotFig(self, yData, subplotPos, figTitle):
        """
        Plot figure
        args:
        yData: string holding name of dataframe column to be plotted
        subplotPos: integer holding position of plot
        """
        ax = self.figure.add_subplot(8, 3, subplotPos)
        # y_lim values will be set dynamically per event
        ax.set_ylim(-0.008, 0.008)
        ax.xaxis.set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        dfCopy = self._model.df.copy()
        stats = getStats(dfCopy, yData)
        logging.debug('stats for {0}: {1}'.format(figTitle, stats))
        # text location will be set dynamically per event
        ax.text(30000, -0.005, stats)
        self._model.df.reset_index().plot(kind='line', x='index', y=yData, title=figTitle, ax=ax)
        self.figure.tight_layout()
        del dfCopy
        #self.draw()


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