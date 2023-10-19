# -*- coding: utf-8 -*-
"""

Postural Sway Assesment Tool GUI.

"""

try:
    from PySide import QtGui, QtCore

    Signal = QtCore.Signal
    Slot = QtCore.Slot

    onDesktop = False

except ImportError as exp:
    print(exp)
    print("Looking for PyQT")

    from PyQt5 import QtGui, QtCore, QtWidgets

    Signal = QtCore.pyqtSignal
    Slot = QtCore.pyqtSlot

    onDesktop = True

import calendar
import collections
import ctypes
import glob
import multiprocessing
import os
import pdb
import subprocess
import sys
import threading
import time

# import gc
import cv2
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import ir_marker
import posturalCam_master_NETWORK as backend
from file_system import FileOrderingSystem
from shh_client import shh_client

# Switch to the directory that the file is in
os.chdir(os.path.dirname(sys.argv[0]))
sys.stderr = open("PSAT_Error.txt", 'w')
sys.stdout = open("PSAT_out.txt", 'w')


# import posturalCam_master_NETWORK as pCam


def vector_magnitude(v):
    """Calculate the magnitude of a vector"""

    # Check that v is a vector
    if v.ndim != 1:
        raise TypeError("Input is not a vector")
    return np.sqrt(np.sum(np.square(v)))


def load_demo_images():
    """For demo purposes we can load 2 images"""
    frame1 = cv2.imread('calib_img_0.tiff')
    frame2 = cv2.imread('calib_img_1.tiff')
    return frame1, frame2


class cameraSystemDialog(QtWidgets.QDialog):

    def __init__(self, parent):

        super(cameraSystemDialog, self).__init__(parent)

        self.setObjectName("CameraSystemDialog")
        self.setWindowTitle("PSAT")
        self.setModal(True)  # Lock focus on this widget

        #        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        self.masterLayout = QtWidgets.QVBoxLayout()

        main_label = QtWidgets.QLabel("Welcome to PSAT")
        self.masterLayout.addWidget(main_label)

        self.masterLayout.addSpacing(20)

        sub_label = QtWidgets.QLabel(
            "Please wait a moment for PSAT to start communicating with the cameras")
        sub_label.setStyleSheet("""font-size: 15px""")
        self.masterLayout.addWidget(sub_label)

        self.masterLayout.addSpacing(20)
        sub_label2 = QtWidgets.QLabel(
            "If you want to use PSAT without the cameras click cancel")
        sub_label2.setStyleSheet("""font-size: 15px""")
        self.masterLayout.addWidget(sub_label2)

        self.masterLayout.addSpacing(20)
        self.masterLayout.addLayout(self.create_buttons())

        self.masterLayout.setAlignment(main_label, QtCore.Qt.AlignCenter)
        self.masterLayout.setAlignment(sub_label, QtCore.Qt.AlignCenter)
        self.masterLayout.setAlignment(sub_label2, QtCore.Qt.AlignCenter)

        self.setLayout(self.masterLayout)

        self.show()

    def create_buttons(self):

        buttonLayout = QtWidgets.QHBoxLayout()

        self.CloseButton = QtWidgets.QPushButton("Cancel")
        self.CloseButton.clicked.connect(self.close)

        buttonLayout.addWidget(self.CloseButton)

        return buttonLayout

    @Slot()
    def camerasLive(self):
        """Recivies a signal when the cameras go live and then starts a close down sequence"""
        print("SHUTTING CAMERA DIALOG")
        self.changeButtonText()

        QtWidgets.QApplication.processEvents()  # Update the button
        t0 = time.time()

        while True:

            t1 = time.time() - t0

            if t1 > 1.5:
                self.close()
                break

    def changeButtonText(self):

        self.CloseButton.setEnabled(False)
        self.CloseButton.setText("Cameras are live")
        self.CloseButton.setStyleSheet("""background: rgb(16, 159, 221)""")


class myDateLineEdit(QtWidgets.QLineEdit):
    """A QLineEdit for displaying the DOB. On click it should load a window to select a DOB"""

    clicked = Signal()

    def __init__(self):
        super(myDateLineEdit, self).__init__()
        self.setReadOnly(True)
        self.setText("DD/MM/YYYY")

    def mousePressEvent(self, e):
        #        print("DOB Pressed")
        #        self.setStyleSheet("""background-color: rgb(255,105,); """)
        self.clicked.emit()
        super(myDateLineEdit, self).mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        #        self.setStyleSheet("""background-color: rgb(52,50,52);""")
        super(myDateLineEdit, self).mouseReleaseEvent(e)

    def reset_date(self):
        self.setText("DD/MM/YYYY")


class myDateDialog(QtWidgets.QDialog):
    """A diaglog widget to get DOB information"""

    def __init__(self, parent_window=None):

        super(myDateDialog, self).__init__()

        self.setObjectName("MyDateDialog")
        self.setWindowTitle("Select DOB")

        #        self.setGeometry(100, 100, 10, 10)

        #        self.setFixedSize(600, 360) #Do not allow the window to resize
        #        self.move(10,10)
        self.setModal(True)  # Window will be locked in focus

        self.decade = 2000  # Decade the calendar will start on

        self.selectMonth()

        self.cal = calendar.Calendar(0)  # Create a calendar object

        self.masterWidget = QtWidgets.QWidget()
        self.masterWidget.setObjectName("MyDOB")
        self.masterLayout = QtWidgets.QGridLayout()

        self.selectMonth()  # Create Month window
        self.masterLayout.addWidget(self.monthView, 0, 0, 1, 1)

        self.selectYear()
        self.masterLayout.addWidget(self.yearView, 0, 0, 1, 1)
        self.yearView.hide()

        self.setLayout(self.masterLayout)

        self.setStyleSheet("""  QDialog {background: rgb(52,50,52)}
                                QLabel {border: 0px; color: rgb(221, 89, 2); font-family: embrima; font-size: 22px; font-weight: 500;}
                                QPushButton {background: rgb(52,50,52); border: 1px solid rgb(34, 85, 96); color: rgb(255, 255, 255);
                                             font-family: embrima; font-size: 20px; font-weight: 500; padding: 8px;} 
                                QToolButton {background: rgb(52,50,52); border: 1px solid rgb(34, 85, 96); color: rgb(255, 255, 255); padding: 8px;}
                           """)

    def selectMonth(self):

        self.monthView = QtWidgets.QWidget()

        masterLayout = QtWidgets.QVBoxLayout()

        myLayout = QtWidgets.QGridLayout()

        instruct1 = QtWidgets.QLabel("Select a month")
        instruct1.setAlignment(QtCore.Qt.AlignCenter)
        masterLayout.addWidget(instruct1)

        self.months = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                       'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'Decemeber': 12}
        self.months = collections.OrderedDict(
            sorted(self.months.items(), key=lambda t: t[1]))

        month_labels = [key for key in self.months]
        #        print(month_labels)
        self.buttons = []

        i = 0
        for row in range(3):
            for col in range(4):
                button = QtWidgets.QPushButton(month_labels[i])
                button.clicked.connect(self.setMonth)

                myLayout.addWidget(button, row, col, 1, 1)
                self.buttons.append(button)
                i += 1

        masterLayout.addLayout(myLayout)
        self.monthView.setLayout(masterLayout)

    def setMonth(self):

        sender = self.sender()

        self.month_value = self.months[sender.text()]

        #        print(sender.text(), self.months[sender.text()])

        self.monthView.hide()
        self.yearView.show()

    def selectYear(self):

        self.yearView = QtWidgets.QWidget()

        masterLayout = QtWidgets.QVBoxLayout()
        myLayout = QtWidgets.QGridLayout()

        instruct2 = QtWidgets.QLabel("Select a year")
        instruct2.setAlignment(QtCore.Qt.AlignCenter)
        masterLayout.addWidget(instruct2)

        display_years = range(self.decade, self.decade + 10)

        self.yearButtons = []
        i = 0
        for row in range(2):
            for col in range(5):
                button = QtWidgets.QPushButton(str(display_years[i]))
                button.clicked.connect(self.setYear)
                self.yearButtons.append(button)
                myLayout.addWidget(button, row, col, 1, 1)
                i += 1

        prev_button = QtWidgets.QToolButton()
        prev_button.setArrowType(QtCore.Qt.LeftArrow)
        prev_button.setObjectName("prev_decade")
        prev_button.clicked.connect(self.selectYearChangeDecade)

        next_button = QtWidgets.QToolButton()
        next_button.setArrowType(QtCore.Qt.RightArrow)
        next_button.setObjectName("next_decade")
        next_button.clicked.connect(self.selectYearChangeDecade)

        next_prevLayout = QtWidgets.QHBoxLayout()
        next_prevLayout.addWidget(prev_button)
        next_prevLayout.addWidget(next_button)

        masterLayout.addLayout(myLayout)
        masterLayout.addSpacing(8)
        masterLayout.addLayout(next_prevLayout)
        self.yearView.setLayout(masterLayout)

    def selectYearChangeDecade(self):

        sender = self.sender().objectName()

        #        print(sender)

        if sender == 'next_decade':

            self.decade = self.decade + 10

            self.updateYearButtons()

        elif sender == "prev_decade":

            self.decade = self.decade - 10
            self.updateYearButtons()

    def updateYearButtons(self):

        display_years = range(self.decade, self.decade + 10)

        i = 0
        for row in range(2):
            for col in range(5):
                self.yearButtons[i].setText(str(display_years[i]))
                i += 1

    def setYear(self):

        sender = self.sender()

        self.year_value = sender.text()

        self.yearView.hide()

        self.selectDay()
        self.masterLayout.addWidget(self.dayView, 0, 0, 1, 1)
        self.dayView.show()

    def selectDay(self):

        self.dayView = QtWidgets.QWidget()

        myLayout = QtWidgets.QGridLayout()

        masterLayout = QtWidgets.QVBoxLayout()

        instruct3 = QtWidgets.QLabel("Select a day")
        instruct3.setAlignment(QtCore.Qt.AlignCenter)
        masterLayout.addWidget(instruct3)

        cal = calendar.Calendar(0)
        days = [i for i in cal.itermonthdays(int(self.year_value), int(
            self.month_value)) if i != 0]  # Get all the days in the month

        row = 0
        col = 0

        for d in range(len(days)):

            button = QtWidgets.QPushButton(str(days[d]))
            button.clicked.connect(self.setDay)

            myLayout.addWidget(button, row, col, 1, 1)

            col += 1

            if col > 4:
                row += 1
                col = 0

        masterLayout.addLayout(myLayout)
        self.dayView.setLayout(masterLayout)

    def setDay(self):

        sender = self.sender()
        self.day_value = sender.text()

        self.close()

    def save(self):
        """Save the result"""

        self.DOB = '{}/{}/{}'.format(self.day_value,
                                     self.month_value, self.year_value)

        return self.DOB

    def showEvent(self, event):

        geom = self.frameGeometry()
        geom.moveCenter(QtGui.QCursor.pos())

        self.setGeometry(geom)
        super(myDateDialog, self).showEvent(event)


#        print("SHOWING")


class recordingLocLabel(QtWidgets.QLabel):
    """A scrolling label to show where the data is being stored"""

    def __init__(self, speed):

        super(recordingLocLabel, self).__init__()

        self.speed = speed

        self.setMinimumSize(300, 25)

        self.textPosOffset = 0

        self.qp = QtGui.QPainter()

        self.font_metrics = QtGui.QFontMetrics(self.qp.font())

        self.standard_message = "No save directory has been set: Insert a USB and set the save directory"
        self.message = self.standard_message

    def paintEvent(self, e):

        self.qp.begin(self)
        self.drawWidget(self.qp)
        self.qp.end()

    def drawWidget(self, qp):

        #        font = QtWidgets.QFont('Serif', 7, QtWidgets.QFont.Light)
        #        qp.setFont(font)

        size = self.size()
        w = size.width()
        h = size.height()

        pen = QtGui.QPen(QtGui.QColor(52, 50, 52), 1)
        qp.setPen(pen)
        qp.setBrush(QtGui.QColor(52, 50, 52))
        qp.drawRect(0, 0, w, h)
        #        qp.drawText()
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255), 1)
        qp.setPen(pen)
        qp.drawText(int(w - self.textPosOffset), h - 5, self.message)

    def scrollText(self):

        pixelsWide = 2 * self.font_metrics.width(self.message)

        self.textPosOffset += 1 * self.speed

        if self.textPosOffset > (pixelsWide):
            self.textPosOffset = 0
        self.repaint()

    def setMessage(self, message):
        """Set a new message"""

        if message == '':
            self.message = self.standard_message
        else:

            self.message = "Recording directory: {}".format(message)
            self.textPosOffset = 0


class positionLabel(QtWidgets.QLabel):
    """Label for the distance from the center of the cameras"""

    def __init__(self, text):
        super(positionLabel, self).__init__(text)

        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(sizePolicy)

    def sizeHint(self):
        return QtCore.QSize(100, 100)

    def heightForWidth(self, width):
        return width


class positionWidget(QtWidgets.QWidget):

    def __init__(self, minPos, maxPos, idealPos, error=250):

        super(positionWidget, self).__init__()

        self.minPos = minPos
        self.maxPos = maxPos
        self.idealPos = idealPos
        self.error = error
        self.initUI()

    def initUI(self):

        self.setMinimumSize(1, 25)

        self.value = 1000

    def paintEvent(self, e):

        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()

    def transform_pos(self, pos, min_pos, step):
        """Transform a value to a position on the display"""

        return (pos - min_pos) * step

    def drawWidget(self, qp):

        #        font = QtWidgets.QFont('Serif', 7, QtWidgets.QFont.Light)
        #        qp.setFont(font)

        size = self.size()
        w = size.width()
        h = size.height()

        length = self.maxPos - self.minPos
        step = w / length

        pos = self.transform_pos(self.value, self.minPos, step)

        pen = QtGui.QPen(QtGui.QColor(16, 159, 221), 5)
        qp.setPen(pen)
        qp.setBrush(QtGui.QColor(52, 50, 52))
        qp.drawRect(0, 0, w, h)

        #        pen = QtGui.QPen(QtGui.QColor(255,255,255), 10)
        #        qp.setPen(pen)
        #        qp.setBrush(QtGui.QColor(255,255,255))
        #        qp.drawRect(self.transform_pos(self.idealPos - self.error, self.minPos, step), 0, self.error * 2 * step, h)

        # Draw the line
        if (self.idealPos - self.error) < self.value < (self.idealPos + self.error):

            pen = QtGui.QPen(QtGui.QColor(255, 255, 255),
                             8, QtCore.Qt.SolidLine)

        else:

            pen = QtGui.QPen(QtGui.QColor(255, 255, 255),
                             8, QtCore.Qt.SolidLine)

        qp.setPen(pen)
        qp.drawLine(pos, 1, pos, h)

    def setValue(self, value):
        """If the value has changed repaint the widget"""
        if value != self.value:
            self.value = value
            self.repaint()


class virtualKeyboard(QtWidgets.QDialog):
    """"A virtual Keyboard"""

    def __init__(self, parent=None):

        super(virtualKeyboard, self).__init__(parent)

        self.create_keys()

    def create_keys(self):

        self.lower_key_labels = [['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
                                 ['q', 'w', 'e', 'r', 't',
                                  'y', 'u', 'i', 'o', 'p'],
                                 ['a', 's', 'd', 'f', 'g',
                                  'h', 'j', 'k', 'l', ''],
                                 ['CAP', 'z', 'x', 'c', 'v',
                                  'b', 'n', 'm', '<-', ''],
                                 ['', 'Spacebar', '']]

        self.upper_key_labels = [['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
                                 ['Q', 'W', 'E', 'R', 'T',
                                  'Y', 'U', 'I', 'O', 'P'],
                                 ['A', 'S', 'D', 'F', 'G',
                                  'H', 'J', 'K', 'L', ''],
                                 ['CAP', 'Z', 'X', 'C', 'V',
                                  'B', 'N', 'M', '<-', ''],
                                 ['', 'Spacebar', '']]

        self.alt_key_labels = [['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
                               ['@', '£', '&', '_',
                                '(', ')', ':', ';', '"', ''],
                               ['', '!', '#', '=', '*', '/', '+', '-', '*', ''],
                               ['CAP', ',', '.', '', '', '', '', '', '<-', ''],
                               ['', 'Spacebar', '']]

        CAP_f = 'shift_key.png'
        alt_f = 'alt_keys.png'

        self.buttons = [[{'lower': '1', 'upper': '1', 'alt': '1', 'type': 'text'},
                         {'lower': '2', 'upper': '2',
                          'alt': '2', 'type': 'text'},
                         {'lower': '3', 'upper': '3',
                          'alt': '3', 'type': 'text'},
                         {'lower': '4', 'upper': '4',
                          'alt': '4', 'type': 'text'},
                         {'lower': '5', 'upper': '5',
                          'alt': '5', 'type': 'text'},
                         {'lower': '6', 'upper': '6',
                          'alt': '6', 'type': 'text'},
                         {'lower': '7', 'upper': '7',
                          'alt': '7', 'type': 'text'},
                         {'lower': '8', 'upper': '8',
                          'alt': '8', 'type': 'text'},
                         {'lower': '9', 'upper': '9',
                          'alt': '9', 'type': 'text'},
                         {'lower': '0', 'upper': '0', 'alt': '0', 'type': 'text'}],

                        [{'lower': 'q', 'upper': 'Q', 'alt': '@', 'type': 'text'},
                         {'lower': 'w', 'upper': 'W',
                          'alt': '£', 'type': 'text'},
                         {'lower': 'e', 'upper': 'E',
                          'alt': '&', 'type': 'text'},
                         {'lower': 'r', 'upper': 'R',
                          'alt': '_', 'type': 'text'},
                         {'lower': 't', 'upper': 'T',
                          'alt': '(', 'type': 'text'},
                         {'lower': 'y', 'upper': 'Y',
                          'alt': ')', 'type': 'text'},
                         {'lower': 'u', 'upper': 'U',
                          'alt': ':', 'type': 'text'},
                         {'lower': 'i', 'upper': 'I',
                          'alt': ';', 'type': 'text'},
                         {'lower': 'o', 'upper': 'O',
                          'alt': '"', 'type': 'text'},
                         {'lower': 'p', 'upper': 'P', 'alt': '', 'type': 'text'}],

                        [{'lower': 'a', 'upper': 'A', 'alt': '', 'type': 'text'},
                         {'lower': 's', 'upper': 'S',
                          'alt': '!', 'type': 'text'},
                         {'lower': 'd', 'upper': 'D',
                          'alt': '#', 'type': 'text'},
                         {'lower': 'f', 'upper': 'F',
                          'alt': '=', 'type': 'text'},
                         {'lower': 'g', 'upper': 'G',
                          'alt': '*', 'type': 'text'},
                         {'lower': 'h', 'upper': 'H',
                          'alt': '/', 'type': 'text'},
                         {'lower': 'j', 'upper': 'J',
                          'alt': '\\', 'type': 'text'},
                         {'lower': 'k', 'upper': 'K',
                          'alt': '+', 'type': 'text'},
                         {'lower': 'l', 'upper': 'L',
                          'alt': '-', 'type': 'text'},
                         {'lower': '', 'upper': '', 'alt': '*', 'type': 'text'}],

                        [{'lower': CAP_f, 'upper': CAP_f, 'alt': CAP_f, 'type': 'icon', 'name': 'CAP'},
                         {'lower': 'z', 'upper': 'Z',
                          'alt': '.', 'type': 'text'},
                         {'lower': 'x', 'upper': 'X',
                          'alt': ',', 'type': 'text'},
                         {'lower': 'c', 'upper': 'C',
                          'alt': '', 'type': 'text'},
                         {'lower': 'v', 'upper': 'V',
                          'alt': '', 'type': 'text'},
                         {'lower': 'b', 'upper': 'B',
                          'alt': '', 'type': 'text'},
                         {'lower': 'n', 'upper': 'N',
                          'alt': '', 'type': 'text'},
                         {'lower': 'm', 'upper': 'M',
                          'alt': '', 'type': 'text'},
                         {'lower': '<-', 'upper': '<-',
                          'alt': '<-', 'type': 'text'},
                         {'lower': '', 'upper': '', 'alt': '', 'type': 'text'}],

                        [{'lower': alt_f, 'upper': alt_f, 'alt': alt_f, 'type': 'icon', 'name': 'theta'},
                         {'lower': 'SPACEBAR', 'upper': 'SPACEBAR',
                          'alt': 'SPACEBAR', 'type': 'text'},
                         {'lower': '', 'upper': '', 'alt': '', 'type': 'text'}]]

        self.button_case = 'upper'  # Start the buttons in lower case. Used to toggle the case
        self.alt_keys = False  # Start not on alt keys

        #        self.key_buttons = [[Text_Button(k) for k in row] for row in self.lower_key_labels] #Create all the buttons

        for row in self.buttons:
            for k in row:

                if k['type'] == 'text':
                    k['button'] = Text_Button(k[self.button_case])
                elif k['type'] == 'icon':
                    k['button'] = Icon_Button(k[self.button_case])

        self.connect_to_misc_buttons()  # Connect the shift and alt key

        master_grid = QtWidgets.QGridLayout()  # Master layout

        row_grids = [QtWidgets.QGridLayout()
                     for r in range(len(self.buttons))]  # Row Layouts

        # Layout each row

        for key in range(len(self.buttons[0])):
            # Add the first row of buttons to the first grid
            row_grids[0].addWidget(self.buttons[0][key]['button'], 0, key)

        for key in range(len(self.buttons[1])):
            # Add the second row of buttons to the first grid
            row_grids[1].addWidget(self.buttons[1][key]['button'], 0, key)

        for key in range(len(self.buttons[2])):
            # Add the third row of buttons to the first grid
            row_grids[2].addWidget(self.buttons[2][key]['button'], 0, key)

        for key in range(len(self.buttons[3])):
            # Add the forth row of buttons to the first grid
            row_grids[3].addWidget(self.buttons[3][key]['button'], 0, key)

        # Add Spacebar
        row_grids[4].addWidget(self.buttons[4][0]['button'], 0, 0, 1, 3)
        row_grids[4].addWidget(self.buttons[4][1]['button'], 0, 3, 1, 5)
        row_grids[4].addWidget(self.buttons[4][2]['button'], 0, 8, 1, 4)

        # Add all to the master grid
        master_grid.addLayout(row_grids[0], 0, 0)
        master_grid.addLayout(row_grids[1], 1, 0)
        master_grid.addLayout(row_grids[2], 2, 0)
        master_grid.addLayout(row_grids[3], 3, 0)
        master_grid.addLayout(row_grids[4], 4, 0)

        self.setLayout(master_grid)

    #        self.set_case('lower')

    #        self.setStyleSheet("background-color: rgba(0, 0, 0, 100%);")

    #        self.hide() #Make the keyboard invisible

    def connect_to_buttons(self, obj):
        """connect all the buttons to a suitable Qt object"""

        for row in self.buttons:
            for k in row:
                if k['type'] == 'text':
                    k['button'].pressed.connect(obj)

    def connect_to_misc_buttons(self):
        """connect the keyboard to the shiftkey"""
        for row in self.buttons:
            for k in row:
                if (k['type'] == 'icon') and (k['name'] == 'CAP'):
                    k['button'].pressed.connect(self.set_case)

                if (k['type'] == 'icon') and (k['name'] == 'theta'):
                    k['button'].pressed.connect(self.set_alt_keys)

    @Slot()
    def set_alt_keys(self):

        print("RECEIVED SIGNAL ALT")
        if not self.alt_keys:
            print("HI")
            self.alt_keys = True

            for row in self.buttons:
                for k in row:
                    if k['type'] == 'text':
                        k['button'].change_text(k['alt'])

        else:
            # When changing back to text change back to whatever the button case was previously
            self.alt_keys = False

            for row in self.buttons:
                for k in row:
                    if k['type'] == 'text':
                        k['button'].change_text(k[self.button_case])

    @Slot()
    def set_case(self):
        """Toggle the letter case"""

        print("RECIEVED SIGNAL")

        if self.button_case == 'lower':

            self.button_case = 'upper'

            for row in self.buttons:
                for k in row:
                    if k['type'] == 'text':
                        k['button'].change_text(k['upper'])

        elif self.button_case == 'upper':

            self.button_case = 'lower'
            for row in self.buttons:
                for k in row:
                    if k['type'] == 'text':
                        k['button'].change_text(k['lower'])

    @Slot(int)
    def cap_zero_len(self, text_len):

        print("RECIEVED LENGTH: {}".format(text_len))

        if text_len == 0:

            if self.button_case == 'lower':
                self.button_case = 'upper'

                for row in self.buttons:
                    for k in row:
                        if k['type'] == 'text':
                            k['button'].change_text(k['upper'])

        elif text_len == 1:
            if self.button_case == 'upper':
                self.button_case = 'lower'

                for row in self.buttons:
                    for k in row:
                        if k['type'] == 'text':
                            k['button'].change_text(k['lower'])


class Icon_Button(QtWidgets.QToolButton):
    """A Icon button class for the virtual keyboard"""

    pressed = Signal()

    def __init__(self, icon_f, parent=None):
        """Pass the file location of the icon"""
        super(Icon_Button, self).__init__(parent)

        self.icon_f = icon_f
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Preferred)

        rMyIcon = QtGui.QPixmap(self.icon_f)
        self.setIcon(QtGui.QIcon(rMyIcon))

        self.setStyleSheet("""background-color: rgb(52,50,52); color: rgb(255,255,255); border: 1px solid rgb(34, 85, 96); 
                                     border-radius: 0px; font-family: embrima; font-size: 17px; font-weight: 700 """)

    def sizeHint(self):
        size = super(Icon_Button, self).sizeHint()
        size.setHeight(size.height() + 20)
        size.setWidth(max(size.width(), size.height()))
        return size

    def mousePressEvent(self, e):
        self.pressed.emit()

        self.setStyleSheet("""background-color: rgb(255, 105, 0); 
                            border: 0px;
                            color: rgb(255,255,255);""")

    def mouseReleaseEvent(self, e):
        if self.text != '':
            self.setStyleSheet("""background-color: rgb(52,50,52); color: rgb(255,255,255); border: 1px solid rgb(34, 85, 96); 
                                     border-radius: 0px; font-family: embrima; font-size: 17px; font-weight: 700 """)


class Text_Button(QtWidgets.QToolButton):
    """A text button class for the keyboard"""

    pressed = Signal(str)

    def __init__(self, text, parent=None):
        super(Text_Button, self).__init__(parent)

        self.text = text
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Preferred)
        self.setText(text)

        self.setStyleSheet("""background-color: rgb(52,50,52); color: rgb(255,255,255); border: 1px solid rgb(34, 85, 96); 
                                     border-radius: 0px; font-family: embrima; font-size: 17px; font-weight: 700 """)

    def sizeHint(self):
        size = super(Text_Button, self).sizeHint()
        size.setHeight(size.height() + 20)
        size.setWidth(max(size.width(), size.height()))
        return size

    def mousePressEvent(self, e):

        if self.text != '':
            self.pressed.emit(self.text)
            print("Key: {}".format(self.text))

            self.setStyleSheet("""background-color: rgb(255,105,0);                         
                                        color: rgb(255,255,255);
                                        font-weight: 800""")

    def mouseReleaseEvent(self, e):

        if self.text != '':
            self.setStyleSheet("""background-color: rgb(52,50,52); color: rgb(255,255,255); border: 1px solid rgb(34, 85, 96); 
                                     border-radius: 0px; font-family: embrima; font-size: 17px; font-weight: 700 """)

    def change_text(self, text):
        self.text = text
        self.setText(text)


class MyTimeLineEdit(QtWidgets.QLineEdit):
    """Widget for entering the recording time"""

    pressed = Signal()  # Signal to emit when mousepress focused
    end_focus = Signal()
    text_length = Signal(int)  # A integer showing how long the text is

    def __init__(self, parent=None):
        super(MyTimeLineEdit, self).__init__(parent)
        self.setObjectName("MyTimeLineEdit")
        self.text2 = ''

        self.in_focus = None

        self.enabled = True

    def mousePressEvent(self, e):

        self.pressed.emit()  # Emit a signal when the key is pressed
        print("Key Pressed")

    def focusInEvent(self, e):

        QtWidgets.QLineEdit.focusInEvent(self, QtGui.QFocusEvent(
            QtCore.QEvent.FocusIn))  # Call the default In focus event
        self.text_length.emit(len(self.text2))
        self.in_focus = True

        print("IN")

    def focusOutEvent(self, e):

        QtWidgets.QLineEdit.focusOutEvent(self, QtGui.QFocusEvent(
            QtCore.QEvent.FocusOut))  # Call the default Outfocus event

        self.in_focus = False
        self.end_focus.emit()  # Emit signal that focus was lost
        print("OUT")

    @Slot(str)
    def recieve_input(self, inp):

        if self.in_focus and self.enabled:

            print("Recieved key {}".format(inp))

            if inp == '<-':

                self.text2 = self.text2[:-1]

            elif inp == 'SPACEBAR':

                self.text2 += ''

            else:

                if self.test_int(inp):
                    self.text2 += inp

            self.setText(self.text2)

            self.text_length.emit(len(self.text2))

    def test_int(self, text_):
        """Test whether a string can be cast to an integer"""

        is_int = True

        try:

            int(text_)

        except ValueError:

            is_int = False

        if not is_int:

            print("Not a valid integer")

            return False

        else:

            return True

    def toggle_enable(self):

        if self.enabled:
            self.enabled = False
        else:
            self.enabled = True

    def reset_text(self):

        self.text2 = ''
        self.setText('')
        self.text_length.emit(len(self.text2))

    def set_text(self, text_):
        """Override the text"""
        self.text2 = text_
        self.setText(text_)
        self.text_length.emit(len(self.text2))

    def value(self):
        """Only for use with the Record time edit
        Return the value entered into the record time as an int"""

        try:
            value = int(self.text())
        except ValueError as e:
            print("{}: The record value is not a valid integer".format(e))

        return value


class MyLineEdit(QtWidgets.QLineEdit):
    """A line edit that can take input from the virtual keyboard"""

    pressed = Signal()  # Signal to emit when mousepress focused
    end_focus = Signal()
    text_length = Signal(int)  # A integer showing how long the text is

    def __init__(self, parent=None):
        super(MyLineEdit, self).__init__(parent)

        self.text2 = ''

        self.in_focus = None

        self.enabled = True

    def mousePressEvent(self, e):

        self.pressed.emit()  # Emit a signal when the key is pressed
        print("Key Pressed")

    def focusInEvent(self, e):

        QtWidgets.QLineEdit.focusInEvent(self, QtGui.QFocusEvent(
            QtCore.QEvent.FocusIn))  # Call the default In focus event
        self.text_length.emit(len(self.text2))
        self.in_focus = True

        print("IN")

    def focusOutEvent(self, e):

        QtWidgets.QLineEdit.focusOutEvent(self, QtGui.QFocusEvent(
            QtCore.QEvent.FocusOut))  # Call the default Outfocus event

        self.in_focus = False
        self.end_focus.emit()  # Emit signal that focus was lost
        print("OUT")

    @Slot(str)
    def recieve_input(self, inp):

        if self.in_focus and self.enabled:

            print("Recieved key {}".format(inp))

            if inp == 'SPACEBAR':
                self.text2 += ' '
            elif inp == '<-':
                self.text2 = self.text2[:-1]
            else:
                self.text2 += inp

            self.setText(self.text2)

            self.text_length.emit(len(self.text2))

    def toggle_enable(self):

        if self.enabled:
            self.enabled = False
        else:
            self.enabled = True

    def reset_text(self):

        self.text2 = ''
        self.setText('')
        self.text_length.emit(len(self.text2))

    def set_text(self, text):
        """Override the text"""
        self.text2 = text
        self.setText(text)
        self.text_length.emit(len(self.text2))

    def value(self):
        """Only for use with the Record time edit
        Return the value entered into the record time as an int"""

        try:
            value = int(self.text2)
        except ValueError as e:
            print("{}: The record value is not a valid integer".format(e))

        return value


class MyDateEdit(QtWidgets.QDateEdit):
    """
    DEPRECIATED: Now replaced by a special pop up widget
    A child of the QDateEdit that emits a signal when it is pressed"""

    pressed = Signal()  # Signal to emit when mousepress focused
    end_focus = Signal()

    def __init__(self, parent=None):
        super(MyDateEdit, self).__init__(parent)

        self.text = ''

        self.in_focus = None

        self.setCalendarPopup(True)
        self.calendar = self.calendarWidget()

        self.setDate(QtCore.QDate(2000, 1, 1))

    #    def mousePressEvent(self, e):
    #
    #        self.pressed.emit() #Emit a signal when the key is pressed
    #        print("Key Pressed")

    def focusInEvent(self, e):
        QtWidgets.QDateEdit.focusInEvent(self, QtGui.QFocusEvent(
            QtCore.QEvent.FocusIn))  # Call the default In focus event

        self.in_focus = True

        print("IN")

    def focusOutEvent(self, e):
        QtWidgets.QDateEdit.focusOutEvent(self, QtGui.QFocusEvent(
            QtCore.QEvent.FocusOut))  # Call the default Outfocus event

        self.in_focus = False
        self.end_focus.emit()  # Emit signal that focus was lost
        print("OUT")

    @Slot(str)
    def recieve_input(self, inp):
        if self.in_focus:
            self.text = self.sectionText(self.currentSection())
            print(self.text, self.currentSection())
            print(self.date())

    def reset_date(self):
        self.setDate(QtCore.QDate(2000, 1, 1))


class Console(QtWidgets.QTextEdit):

    def __init__(self, parent=None):

        super(Console, self).__init__(parent)

    def addText(self, text):

        if self.toPlainText() == '':

            self.setText(text)

        else:

            self.setText(self.toPlainText() + '\n' + text)

        QtWidgets.QApplication.processEvents()

    def newLine(self):

        self.setText(self.toPlainText() + '\n')
        QtWidgets.QApplication.processEvents()


class QLED(QtWidgets.QWidget):
    """Not currently used or doing anything. Probably doesn't work"""

    def __init__(self, parent=None):
        super(QLED, self).__init__(parent)

        palette = QtCore.QPalette(self.palette())
        palette.setColor(palette.Background, QtCore.Qt.transparent)
        self.setPalette(palette)


class MainWindow(QtWidgets.QMainWindow):
    """
    Main UI

    """
    camerasConnected = Signal()

    def __init__(self):
        super(MainWindow, self).__init__(None)
        self.initUI()

        self.participant_list_loaded = False

    def initUI(self):

        self.setGeometry(50, 50, 800, 480)
        self.setFixedSize(800, 480)  # Do not allow the window to resize
        self.setWindowTitle('Postural Sway Assessment Tool')

        QtWidgets.QApplication.setStyle(
            QtWidgets.QStyleFactory.create('Cleanlooks'))

        self.timer = QtCore.QTimer()  # A timer for triggering events
        self.create_widgets()

        self.add_menubar()  # Add the menubar
        self.show()

        # Set a storage_loc. If none it will save to the main directory
        self.storage_loc = ''

        # Just a work around, but the onDesktop variable should be depreciated in future
        onDesktop = False
        if not onDesktop:
            # self.launch_server_pi()
            self.camera_backend_live = False  # Marker that backend has been imported
            self.create_camera_backend()

    def add_menubar(self):
        """Add a menubar to the main UI"""

        # drop down menu
        exitAction = QtWidgets.QAction(
            QtGui.QIcon('exit.png'), '&Shutdown', self)
        exitAction.setStatusTip('Shutdown PSAT')
        exitAction.triggered.connect(self.shutdown_event)

        # New participant button
        new_partMenu = QtWidgets.QAction(
            QtGui.QIcon('exit.png'), '&New Participant', self)
        new_partMenu.triggered.connect(self.new_participant)

        # Choose storage location
        data_locMenu = QtWidgets.QAction(QtGui.QIcon(
            'exit.png'), '&Set Save Directory', self)
        data_locMenu.triggered.connect(self.get_storage_location)

        # Load a participant list
        self.part_listMenu = QtWidgets.QAction(
            QtGui.QIcon(''), '&Load Participant List', self)
        self.part_listMenu.triggered.connect(self.toggle_load_part_button)

        # Select different windows

        self.processDataMenu = QtWidgets.QAction(
            QtGui.QIcon(''), '&Process Data', self)
        self.processDataMenu.triggered.connect(self.processing_window)

        # Add the menubar and the menu buttons
        menubar = self.menuBar()

        #        menubar.setFixedHeight(25)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        participantMenu = menubar.addMenu('&Data Control')
        participantMenu.addAction(data_locMenu)
        participantMenu.addAction(self.part_listMenu)
        participantMenu.addAction(new_partMenu)

        processMenu = menubar.addMenu("&Data Processing")
        processMenu.addAction(self.processDataMenu)

        self.rec_scroll = recordingLocLabel(2.3)  # Takes the speed as an input

        menubar.setCornerWidget(self.rec_scroll, QtCore.Qt.TopRightCorner)

        #        menubar.setCornerWidget(test_lab, QtCore.Qt.TopRightCorner)

        self.timer.timeout.connect(self.rec_scroll.scrollText)
        self.timer.start(33)

    def processing_window(self):
        """Go to the processing window if a storage location has been set"""

        if not self.storage_loc == '':

            self.show_window_3()

        else:

            error_msgBox = QtWidgets.QMessageBox(self)
            error_msgBox.setText(
                """A storage location has not been set.\n\nSelect a storage location by clicking Data Control -> Set Save Directory""")
            error_msgBox.exec_()
            return

    def get_storage_location(self):
        """Select the location to save the data"""

        if onDesktop:
            self.storage_loc = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open file',
                                                                          '/media/pi/')
        else:
            self.storage_loc = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open file',
                                                                          '/media/pi/')

        self.rec_scroll.setMessage(self.storage_loc)

    def launch_server_pi(self):
        """Try to start the server on the RPi"""
        print("Launch server pi")
        self.shh_connection = shh_client()

        self.shh_connection_thread = threading.Thread(
            target=self.shh_connection.start_server)
        self.shh_connection_thread.start()

        print("WE got it")

    def shutdown_event(self):
        """Shut down button action. Opens a question box to ask if you want to power down"""
        shutdown_msg = "Power down PSAT?"

        reply = QtWidgets.QMessageBox.question(
            self, "Shutdown", shutdown_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:

            # Ask the server to shut down
            # Wait for the SHH
            if self.camera_backend_live:
                self.backend_camera.TCP_client_shutdown_server()  # Ask the server to shutdown
                self.shh_connection_thread.join()  # Wait till the server has shutdown
                self.shh_connection.shutdown_server_pi()
                subprocess.call(["sudo", "shutdown", "now"])

            # Quit Application. This will close down the RPis in future
            QtCore.QCoreApplication.instance().quit()
        else:
            print("Not closing")

        close_msg = 'Exit to Desktop?'
        reply2 = QtWidgets.QMessageBox.question(
            self, "Shutdown", close_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply2 == QtWidgets.QMessageBox.Yes:
            QtCore.QCoreApplication.instance().quit()  # Quit Application.

    def create_camera_backend(self):
        """Import the camera backend and establist a connection with the server"""

        self.attempt_connection = True

        self.startDialog = cameraSystemDialog(self)
        # IF closed clicked on the startDialog stop the thread
        self.startDialog.CloseButton.clicked.connect(
            self.stop_attempt_connection)
        self.camerasConnected.connect(self.startDialog.camerasLive)

        # Start a thread which will attempt to connect to the server
        threading.Thread(target=self.start_camera_backend_connection).start()

        # self.camera_backend_live = self.backend_camera.TCP_client_start() #If it connects set the backend_live = True

    #        self.camera_backend_live = True #Signal that the backend has been imported

    def start_camera_backend_connection(self):

        self.backend_camera = backend.posturalCam()

        self.camera_backend_live = False

        while self.attempt_connection:
            print("LOOKING FOR CONNECTION")
            self.camera_backend_live = self.backend_camera.TCP_client_start()

            if self.camera_backend_live:
                print("BACKEND LIVE")
                self.camerasConnected.emit()
                self.stop_attempt_connection()
            else:
                print("BACKEND NOT LIVE")

        print("NO LONGER LOOKING")

    def stop_attempt_connection(self):

        self.attempt_connection = False

    def create_widgets(self):

        self.central_widget = QtWidgets.QWidget()

        # Set the margins of the main window
        self.central_widget.setContentsMargins(10, 0, 2, 5)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(15)
        #        grid.setMargin(0)

        grid.addWidget(self.create_demographics_group(), 0, 0, 3, 1)
        grid.addWidget(self.create_camera_group(), 0, 0, 1, 1)
        #        grid.addWidget(self.create_recording_group(), 0, 0, 1, 1)
        grid.addWidget(self.create_processing_group(), 0, 0, 1, 1)

        #        grid.addWidget(self.create_keyboard(), 3, 0, 4, 1)

        if onDesktop:
            #            self.show_window_3()
            self.show_window_1()
        #            self.show_window_2()
        else:

            self.show_window_1()

        self.central_widget.setLayout(grid)
        self.setCentralWidget(self.central_widget)

        # Demographics box

    def create_info_group(self):

        self.info_box = QtWidgets.QGroupBox("Info")

        info_grid = QtWidgets.QGridLayout()

        Rec_loc = QtWidgets.QLabel('Rec Loc')

        info_grid.addWidget(Rec_loc, 0, 0, 1, 1)

        self.info_box.setLayout(info_grid)

        return info_grid

    def show_keyboard(self):

        print("SHOW")
        self.keyboard.show()

    def hide_keyboard(self):

        self.keyboard.hide()
        print("HIDE")

    def load_file(self):

        self.participant_list_fname = ''
        # IF this fails it probably needs a [0] after the next line. Needs removing to use with PyQt

        if onDesktop:
            self.participant_list_fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                                                                self.storage_loc)
        else:
            self.participant_list_fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                                                                self.storage_loc)[0]

        # Check that a file was selected and load the file

        if self.participant_list_fname != '':
            self.participant_list = pd.read_csv(self.participant_list_fname)
            self.participant_list_loaded = True

            print(self.participant_list_fname[0])

            # Add a completer to the ID_edit (NOTE: Only works when typing with read keyboard. Needs bug fix)
            completer = QtWidgets.QCompleter()
            self.ID_edit.setCompleter(completer)
            model = QtWidgets.QStringListModel(
                list(self.participant_list['ID'].unique()))
            completer.setModel(model)
            self.ID_edit.textChanged.connect(completer.setCompletionPrefix)

    def new_participant(self):

        print("NEW PARTICIPANT")

        self.Forename_edit.reset_text()
        self.Surname_edit.reset_text()
        self.ID_edit.reset_text()
        self.DOB_edit.reset_date()

        self.Condition_edit.clear()

        self.Record_Time.set_text("5")

    def find_participant(self):

        self.participant_found = True

        ID = self.ID_edit.text()  # Get the ID QLineEdit text (Case sensitive)

        if self.participant_list_loaded:

            part_data = self.participant_list[self.participant_list['ID'].astype(
                str) == str(ID)]

            # Number of entries for that participant
            N_entries = part_data.shape[0]

            if N_entries == 0:

                message_box = QtWidgets.QMessageBox()
                message_box.setText(
                    "ID not found in the participant list\nRemember ID is case sensitive")
                message_box.setIcon(QtWidgets.QMessageBox.Information)
                message_box.exec_()

            else:
                # Get the first index value
                self.current_entry_index = part_data.index[0]

                part_forename = self.participant_list.loc[self.current_entry_index]['Forename']
                part_surname = self.participant_list.loc[self.current_entry_index]['Surname']

                part_DOB = self.participant_list.loc[self.current_entry_index]['DOB']
                part_Gender = self.participant_list.loc[self.current_entry_index]['Gender']
                part_recTime = self.participant_list.loc[self.current_entry_index]['Recording_Time']
                print(ID, part_forename, part_surname, part_DOB, part_Gender)

                # Now set the QLine Edits to these
                self.Forename_edit.setText(part_forename)
                # This stops keyboard delete removing the entire entry
                self.Forename_edit.text2 = part_forename

                self.Surname_edit.setText(part_surname)
                self.Surname_edit.text2 = part_surname

                self.Record_Time.set_text(str(part_recTime))
                #                self.Record_Time.text2 = part_recTime

                if part_Gender.upper() == 'M':
                    self.Gender_edit.setCurrentIndex(0)
                elif part_Gender.upper() == 'F':
                    self.Gender_edit.setCurrentIndex(1)

                self.DOB_edit.setText(part_DOB)

                self.Condition_edit.clear()

                #                for entry in part_data['Condition'].values:
                #                    self.Condition_edit.addItem(entry)

                for e in range(len(part_data)):
                    entry = part_data.iloc[e]

                    self.Condition_edit.addItem(entry['Condition'])


        #                print(len(part_data))

        else:
            print("Load a participant list first")
            message_box = QtWidgets.QMessageBox()
            message_box.setText("No participant list loaded")
            message_box.setIcon(QtWidgets.QMessageBox.Information)
            message_box.exec_()

    def change_condition(self):
        """Causing IndexError loading the participant. Probably hasn't changed the inline text edit by this point        
        """

        ID = self.ID_edit.text()  # Get the ID QLineEdit text (Case sensitive)

        if self.participant_list_loaded:

            part_data = self.participant_list[self.participant_list['ID'].astype(
                str) == str(ID)]

            condition = self.Condition_edit.currentText()

            if condition == '':
                # Bit of a hack. Stops an error being thrown because this function is called when a participant is loaded up (erroneously) and when the condition is changed (working correctly)
                print("NO CONDITION SELECTED")
                return

            part_data_cond = part_data[part_data['Condition'] == condition]
            self.current_entry_index = part_data_cond.index[0]

            part_recTime = part_data_cond['Recording_Time'].values[0]

            self.Record_Time.set_text(str(part_recTime))

    #            self.Record_Time.text2 = part_recTime

    def check_participant_not_run(self):
        """Checks if this participant entry has already been run"""

        if self.participant_list_loaded:
            if self.participant_list.loc[self.current_entry_index, 'Run'] == 1:
                message_box = QtWidgets.QMessageBox()
                message_box.setText("This participant entry was already run\n")
                message_box.setIcon(QtWidgets.QMessageBox.Information)
                message_box.exec_()

                return True

    def update_participant_list(self):
        """Call at the end of recording. If self.participant_list is loaded then update the last recording entry to recorded"""

        if self.participant_list_loaded:
            # Once the recording has happened set the Run to 1
            self.participant_list.loc[self.current_entry_index, 'Run'] = 1

            # Update the participant list file
            self.participant_list.to_csv(
                self.participant_list_fname, index=False)

    def show_window_1(self):
        """Show the demographics window"""
        #        self.demographicsBox.show()
        #        self.camera_box.hide()
        #        self.recording_box.hide()
        self.camera_container.hide()
        self.demographics_container.show()
        self.show_keyboard()
        self.processingBox.hide()
        # Start the timer that controls the scrolling text on the menubar
        self.timer.start(33)

        if self.preview_live:
            self.toggle_preview_server()

    def show_window_2(self):
        """Show the check camera window. Only do this if the camera backend is online"""

        if self.camera_backend_live:
            self.timer.stop()  # Stop the timer that controls the scrolling text on the menubar
            self.camera_container.show()
            self.demographics_container.hide()

            self.hide_keyboard()
            self.processingBox.hide()

            if not self.preview_live:
                self.toggle_preview_server()

    def show_window_3(self):
        """Show the processing window"""
        #        self.timer.stop()#Stop the timer that controls the scrolling text on the menubar
        self.camera_container.hide()
        self.demographics_container.hide()
        self.load_directory()
        self.processingBox.show()
        #        self.demographicsBox.hide()
        #        self.camera_box.hide()
        #        self.recording_box.hide()
        self.hide_keyboard()

        if self.preview_live:
            self.toggle_preview_server()

    def create_processing_group(self):

        self.processingBox = QtWidgets.QGroupBox("Processing")

        get_dir_button = QtWidgets.QPushButton("Refresh data directory")
        # Fails to wipe the current tree (BUG). Removed for the moment
        get_dir_button.clicked.connect(self.load_directory)

        process_data_button = QtWidgets.QPushButton("Process data")
        process_data_button.clicked.connect(
            self.process_data_popup)  # Start the process data popup
        # Becomes a list in later functions. If none then later functions wont execute
        self.all_records = None

        # Show all the files in a directory with this
        self.pointListBox = QtWidgets.QTreeWidget()

        header = QtWidgets.QTreeWidgetItem(["Recorded_data"])
        # Another alternative is setHeaderLabels(["Tree","First",...])
        self.pointListBox.setHeaderItem(header)
        self.load_directory()

        #        processTab = QtWidgets.QTabWidget()
        #        tab1 = QtWidgets.QWidget()
        #        tab2 = QtWidgets.QWidget()
        #        tab3 = QtWidgets.QWidget()

        #        processTab.addTab(tab1,"3d Data")
        #        processTab.addTab(tab2,"Tab 2")
        #        processTab.addTab(tab3,"Tab 3")

        processing_grid = QtWidgets.QGridLayout()

        #        demographics_grid.addWidget(New_participant, 0, 0, 1, 1)

        processing_grid.addWidget(get_dir_button, 5, 0, 1, 1)
        processing_grid.addWidget(process_data_button, 5, 1, 1, 1)
        processing_grid.addWidget(self.pointListBox, 0, 0, 5, 5)
        #        processing_grid.addWidget(processTab, 0, 1, 5, 2)
        processing_grid.setSpacing(10)

        self.processingBox.setLayout(processing_grid)
        return self.processingBox

    def load_directory(self):
        """Load a directory tree to show all the records than can be processed
        Allows the operator to select data to analyse using a tree
        """
        try:
            data_directory = os.path.join(self.storage_loc, 'recorded_data')
        except:
            data_directory = os.path.join(os.getcwd(), 'recorded_data')

        experiments = glob.glob(os.path.join(data_directory, '*'))
        self.pointListBox.clear()  # Clear the treewidget
        self.all_records = []
        for exp in experiments:

            name = os.path.split(exp)[-1]
            parent = QtWidgets.QTreeWidgetItem(self.pointListBox)
            parent.setText(0, "{}".format(name))
            parent.setFlags(
                parent.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable)

            participants = glob.glob(os.path.join(exp, '*'))

            for part in participants:
                print(part)
                p_name = os.path.split(part)[-1]
                child = QtWidgets.QTreeWidgetItem(parent)
                child.setFlags(
                    child.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable)
                child.setText(0, "Participant: {}".format(p_name))
                child.setCheckState(0, QtCore.Qt.Unchecked)

                records = glob.glob(os.path.join(part, '*'))

                for rec in records:
                    print(rec)
                    r_name = os.path.split(rec)[-1]
                    cond_name = pd.Series.from_csv(os.path.join(
                        rec, 'demographics.csv'))['condition']
                    child2 = QtWidgets.QTreeWidgetItem(child)
                    child2.setFlags(child2.flags() |
                                    QtCore.Qt.ItemIsUserCheckable)
                    child2.setText(0, "{}: {}".format(r_name, cond_name))
                    child2.setCheckState(0, QtCore.Qt.Unchecked)

                    self.all_records.append([child2, rec])

        self.pointListBox.expandAll()
        self.pointListBox.update()

    def process_data_popup(self):
        """Process the records selected in load directory"""

        n_records_to_process = 0
        if not isinstance(self.all_records, type(None)):

            for rec, rec_name in self.all_records:

                if rec.checkState(0) == 2:
                    print(rec_name)
                    n_records_to_process += 1
        #        pop_up_processor = QtWidgets.QWidget()
        #        pop_up_processor.setGeometry(QtCore.QRect(100, 100, 400, 200))
        #        pop_up_processor.show()
        #
        self.msg = QtWidgets.QWidget()
        self.msg.setGeometry(QtCore.QRect(200, 100, 400, 300))
        self.msg.setWindowTitle("Data Processing")

        text1 = QtWidgets.QLabel(
            "You have selected {} records to process".format(n_records_to_process))

        process_button = QtWidgets.QPushButton("Process")
        process_button.clicked.connect(self.process_selected_data)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.msg.close)

        self.console = Console()

        pop_up_processor_grid = QtWidgets.QGridLayout()

        pop_up_processor_grid.addWidget(text1, 0, 0, 1, 1)

        pop_up_processor_grid.addWidget(self.console, 1, 0, 3, 3)
        pop_up_processor_grid.addWidget(process_button, 4, 0, 1, 1)
        pop_up_processor_grid.addWidget(cancel_button, 4, 1, 1, 1)

        self.msg.setLayout(pop_up_processor_grid)
        #        msg.setIcon(QtWidgets.QMessageBox.Information)
        #        msg.setText("Processing data")
        #        msg.setInformativeText("You have selected {} records to process".format(len(self.all_records)))
        #        msg.setWindowTitle("Data Processing")
        #        msg.setDetailedText("Click OK to process all selected data\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        #        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)

        #        b = QtWidgets.QPushButton( "HI", msg)
        retval = self.msg.show()

    def process_selected_data(self):
        """Process all the selected data from the process_data_popup"""
        print("PROCESS")
        # Loop over records
        # Check if 3d data process
        # If not process the data
        #        self.console.Text()

        fos = FileOrderingSystem()  # File processing system object

        if not isinstance(self.all_records, type(None)):

            for rec, rec_name in self.all_records:

                if rec.checkState(0) == 2:

                    self.console.addText('Record: {}'.format(rec_name))

                    # Check if file has been processed
                    record_status = fos.check_record_process_status(rec_name)
                    if record_status['summary.csv']:
                        self.console.addText("\tRecord already processed")
                    else:
                        self.console.addText(
                            "\tRecord not processed. Processing data")
                        self.console.addText(
                            "\tProcessing client video. Please wait")

                        proc = backend.posturalProc(v_fname=os.path.join(
                            rec_name, 'videos', 'testIR.h264'), kind='client')
                        proc_all_markers = ir_marker.markers2numpy(
                            proc.get_ir_markers())

                        self.console.addText(
                            "\tProcessing server video. Please wait")

                        proc2 = backend.posturalProc(v_fname=os.path.join(
                            rec_name, 'videos', 'testIR_server.h264'), kind='client')
                        proc2_all_markers = ir_marker.markers2numpy(
                            proc2.get_ir_markers()[:proc_all_markers.shape[0]])

                        self.console.addText("\tSaving marker data")

                        # Save marker data
                        np.save(
                            open(os.path.join(rec_name, 'client_IRpoints.npy'), 'wb'), proc_all_markers)
                        np.save(
                            open(os.path.join(rec_name, 'server_IRpoints.npy'), 'wb'), proc2_all_markers)

                        self.console.addText("\tTriangulating markers")
                        stereo = backend.stereo_process(proc, proc2)
                        markers3d = stereo.triangulate_all_get_PL(
                            proc_all_markers, proc2_all_markers)
                        markers3d_filt = stereo.kalman_smoother(markers3d)

                        marker_mid_3d = np.sum(markers3d, axis=1) / 2.0
                        marker_mid_3d_filt = np.sum(markers3d_filt, axis=1) / 2.0

                        distance_between_leds_filt = np.sqrt(
                            np.sum(np.square(np.diff(markers3d_filt, axis=1)), axis=2)).squeeze()
                        plt.plot(distance_between_leds_filt)
                        plt.show()

                        self.console.addText("\tSaving 3d data")
                        np.save(
                            open(os.path.join(rec_name, '3d_unfiltered.npy'), 'wb'), markers3d)
                        np.save(
                            open(os.path.join(rec_name, '3d_filtered.npy'), 'wb'), markers3d_filt)

                        self.console.addText("\tCreating summary file")

                        PL = np.sum(
                            np.sqrt(np.sum(np.square(np.diff(markers3d_filt, axis=0)), axis=2)), axis=0)
                        PL_mid = np.sum(
                            np.sqrt(np.sum(np.square(np.diff(marker_mid_3d_filt, axis=0)), axis=1)), axis=0)

                        # Write a csv with marker 1, marker 2 and mid marker point path lengths
                        summary = pd.Series(
                            {'PL1': PL[0], 'PL2': PL[1], 'MID': PL_mid})
                        summary.to_csv(os.path.join(rec_name, 'summary.csv'))
                        #                        pd.DataFrame({'PL': np.append(PL, PL_mid)}).to_csv(os.path.join(rec_name, 'summary.csv'))
                        self.console.addText(
                            "\tPath Length: {}".format(np.append(PL, PL_mid)))

                        self.console.newLine()

            self.create_summary_file()

    def create_summary_file(self):
        """Create a summary file of all the data that has been markered for processing"""

        self.console.addText("Creating summary file")

        master_dataFrame = pd.DataFrame()

        for rec, rec_name in self.all_records:

            if rec.checkState(0) == 2:
                summary = pd.Series.from_csv(
                    os.path.join(rec_name, 'summary.csv'))
                demographics = pd.Series.from_csv(
                    os.path.join(rec_name, 'demographics.csv'))

                joined = pd.concat((demographics, summary))
                print(joined)
                master_dataFrame = master_dataFrame.append(
                    joined, ignore_index=True)

        print(master_dataFrame)
        # Save a csv of the summarised data
        master_dataFrame.to_csv(os.path.join(
            os.getcwd(), 'recorded_data', 'master_data.csv'))

    #                print(summary)
    #                print(demographics)

    def create_demographics_group(self):

        # Main Groupings
        self.demographics_container = QtWidgets.QWidget()

        self.demographicsBox = QtWidgets.QGroupBox("Demographics")
        self.optionsBox = QtWidgets.QGroupBox("Options")

        VBox = QtWidgets.QVBoxLayout()

        HBox = QtWidgets.QHBoxLayout()
        HBox.setSpacing(20)
        #        HBox.setMargin(0)
        HBox.addWidget(self.demographicsBox)
        HBox.addWidget(self.optionsBox)

        VBox.addLayout(HBox)

        self.demographics_container.setLayout(VBox)

        # Labels and buttons
        #        rec_loc_label = QtWidgets.QLabel('Recording Loc')
        #        self.rec_loc_ = QtWidgets.QLabel('')

        Forename_label = QtWidgets.QLabel('Forename:')
        self.Forename_edit = MyLineEdit(self)
        #        self.Forename_edit.pressed.connect(self.show_keyboard)
        #        self.Forename_edit.end_focus.connect(self.hide_keyboard)

        Surname_label = QtWidgets.QLabel('Surname:')
        self.Surname_edit = MyLineEdit(self)
        #        self.Surname_edit.pressed.connect(self.show_keyboard)
        #        self.Surname_edit.end_focus.connect(self.hide_keyboard)

        Gender = QtWidgets.QLabel('Gender:')
        self.Gender_edit = QtWidgets.QComboBox()
        self.Gender_edit.addItem("Male")
        self.Gender_edit.addItem("Female")

        ID = QtWidgets.QLabel("ID")
        self.ID_edit = MyLineEdit(self)

        Condition = QtWidgets.QLabel("Condition:")
        self.Condition_edit = QtWidgets.QComboBox()
        self.Condition_edit.currentIndexChanged.connect(self.change_condition)
        self.Condition_edit.setStyleSheet("border-color: rgb(221, 89, 2)")

        #        self.ID_edit.pressed.connect(self.show_keyboard)
        #        self.ID_edit.end_focus.connect(self.hide_keyboard)

        #        d.show()
        #        self.d = myDateDialog()
        DOB = QtWidgets.QLabel('DOB:')
        self.DOB_edit = myDateLineEdit()
        self.DOB_edit.clicked.connect(self.getDOB)

        Record_Time_label = QtWidgets.QLabel("Time (Seconds):")
        #        self.Record_Time = QtWidgets.QDoubleSpinBox()
        #        self.Record_Time.setMinimum(0)
        #        self.Record_Time.setValue(10)
        self.Record_Time = MyTimeLineEdit(self)
        self.Record_Time.set_text("5")

        #        self.load_IDs = QtWidgets.QPushButton("Load Part List")
        #        self.load_IDs.setFixedSize(150,35)
        # self.load_IDs.clicked.connect(self.load_file)
        #        self.load_IDs.clicked.connect(self.toggle_load_part_button)

        check_ID = QtWidgets.QPushButton("Verify")
        #        check_ID.setFixedSize(5,30)
        check_ID.setStyleSheet("background-color: rgb(16, 159, 221)")
        check_ID.clicked.connect(self.find_participant)

        start_Rec = QtWidgets.QPushButton("Next")
        #        start_Rec.setFixedSize(5,30)
        start_Rec.clicked.connect(self.show_window_2)

        demographics_grid = QtWidgets.QGridLayout()
        demographics_grid.setSpacing(15)

        demographics_grid.setContentsMargins(10, 25, 10, 10)

        #        demographics_grid.addWidget(rec_loc_label, 0, 0, 1, 1)
        ##        demographics_grid.addWidget(self.rec_loc_, 0, 1, 1, 1)

        demographics_grid.addWidget(Forename_label, 0, 0, 1, 1)
        demographics_grid.addWidget(self.Forename_edit, 0, 1, 1, 1)

        demographics_grid.addWidget(Surname_label, 1, 0, 1, 1)
        demographics_grid.addWidget(self.Surname_edit, 1, 1, 1, 1)

        demographics_grid.addWidget(ID, 2, 0, 1, 1)
        demographics_grid.addWidget(self.ID_edit, 2, 1, 1, 1)

        demographics_grid.addWidget(Gender, 0, 2, 1, 1)
        demographics_grid.addWidget(self.Gender_edit, 0, 3, 1, 1)

        demographics_grid.addWidget(DOB, 1, 2, 1, 1)
        demographics_grid.addWidget(self.DOB_edit, 1, 3, 1, 1)

        demographics_grid.addWidget(check_ID, 2, 3, 1, 1)

        #        demographics_grid.addWidget(rec_loc_label, 3,0, 1, 1)

        self.demographicsBox.setLayout(demographics_grid)

        options_grid = QtWidgets.QGridLayout()
        options_grid.setSpacing(15)
        options_grid.setContentsMargins(10, 25, 10, 10)

        options_grid.addWidget(Record_Time_label, 0, 0, 1, 1)
        options_grid.addWidget(self.Record_Time, 0, 1, 1, 1)
        options_grid.addWidget(Condition, 1, 0, 1, 1)
        options_grid.addWidget(self.Condition_edit, 1, 1, 1, 1)
        options_grid.addWidget(start_Rec, 2, 1, 1, 1)

        self.optionsBox.setLayout(options_grid)
        #
        #        demographics_grid.addWidget(self.load_IDs, 3, 0, 1, 1)

        #
        #        demographics_grid.addWidget(Condition, 3, 2, 1, 1)
        #        demographics_grid.addWidget(self.Condition_edit, 3, 3, 1, 1)
        #
        #        demographics_grid.addWidget(start_Rec, 3, 5, 1, 1)
        # demographics_grid.addWidget(New_participant, 3, 5, 1, 1)

        VBox.addSpacing(5)
        self.create_keyboard()
        VBox.addWidget(self.keyboard)
        #        VBox.setMargin(0)
        VBox.setContentsMargins(0, 0, 0, 0)

        return self.demographics_container

    def getDOB(self):
        """Launch a QDIalog window to get the DOB for the participant"""
        #        #DOB should be a date format or a box to select the date
        self.d = myDateDialog(self)
        #        self.d.move(50,50)
        self.d.exec_()

        if self.d.result() == 0:
            DOB_val = self.d.save()
            #            DOB_val = DOB_val.toPyDateTime()
            #            DOB_str = DOB_val.strftime('%d/%m/%Y')
            self.DOB_edit.setText(DOB_val)

    def toggle_load_part_button(self):
        """
        Toggles the load participant button. If no participant list loaded it will open a window to select a participant list.
        Otherwise it will remove the currently loaded participant list and clear the participant information
        """

        print(self.part_listMenu.text())
        if self.part_listMenu.text() == "&Load Participant List":

            self.load_file()  # Load the participant list

            if self.participant_list_fname != '':
                self.new_participant()  # Get rid of anything in the demographics windows
                # Rename the unload button
                self.part_listMenu.setText("&Remove Participant List")

                # Disable editing of demographics
                self.Forename_edit.toggle_enable()
                self.Surname_edit.toggle_enable()
                self.Record_Time.toggle_enable()
                self.Gender_edit.setEditable(False)

        else:

            self.participant_list_loaded = False
            self.new_participant()
            self.part_listMenu.setText("&Load Participant List")

            # Enable editing of demographics
            self.Forename_edit.toggle_enable()
            self.Surname_edit.toggle_enable()
            self.Record_Time.toggle_enable()

    def create_camera_group(self):

        self.camera_container = QtWidgets.QWidget()  # Container for all widgets
        self.camera_container.setObjectName("cameraContainer")
        self.camera_container.setContentsMargins(0, 0, 0, 0)

        container_grid = QtWidgets.QVBoxLayout()
        container_grid.setContentsMargins(0, 0, 0, 0)
        #        container_grid.setMargin(0)

        # Create a group box for each pane
        camera_box = QtWidgets.QGroupBox("Camera Preview")
        triangulation_box = QtWidgets.QGroupBox("Triangulation")
        control_box = QtWidgets.QGroupBox("Control")

        camera_box.setSizePolicy(QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding))

        self.camera_container.setLayout(container_grid)

        # Control Pane
        control_layout_spacer = QtWidgets.QVBoxLayout()
        control_layout_spacer.addSpacing(10)

        control_layout = QtWidgets.QHBoxLayout()
        control_layout_spacer.addLayout(control_layout)
        #        control_layout.setMargin(15)

        checkbox_container = QtWidgets.QWidget()

        checkbox_container.setObjectName("camera_checkbox")
        #        checkbox_container.setStyleSheet(".QWidget {border: 1px solid rgb(221,89,2)}; ")
        checkbox_VLayout = QtWidgets.QVBoxLayout()

        self.preview_options_triangulate = QtWidgets.QCheckBox("Triangulate")
        self.preview_options_triangulate.setChecked(False)
        self.record_options_send_video = QtWidgets.QCheckBox("Send Video")
        self.record_options_send_video.setChecked(True)
        self.record_options_send_IRPoints = QtWidgets.QCheckBox("Process data")

        checkbox_VLayout.addWidget(self.preview_options_triangulate)
        checkbox_VLayout.addWidget(self.record_options_send_video)
        checkbox_VLayout.addWidget(self.record_options_send_IRPoints)
        checkbox_container.setLayout(checkbox_VLayout)

        control_layout.addWidget(checkbox_container)

        control_button_container = QtWidgets.QWidget()
        control_button_VLayout = QtWidgets.QVBoxLayout()

        self.preview = QtWidgets.QPushButton("Start Preview")
        self.preview.setObjectName("controlButton")
        self.preview_live = False
        self.preview.clicked.connect(self.toggle_preview_server)

        self.start_button = QtWidgets.QPushButton("Start Recording")
        self.start_button.setObjectName("controlButton")
        self.start_button.clicked.connect(
            self.start_recording)  # Also call the RPi Recording

        back = QtWidgets.QPushButton("Back")
        back.setObjectName("controlButton")
        back.clicked.connect(self.show_window_1)

        control_button_VLayout.addWidget(self.preview)
        control_button_VLayout.addWidget(self.start_button)
        control_button_VLayout.addWidget(back)

        control_button_container.setLayout(control_button_VLayout)

        control_layout.addWidget(control_button_container)

        control_box.setLayout(control_layout_spacer)

        # Camera Pane

        camera_layout = QtWidgets.QHBoxLayout()
        camera_layout.setContentsMargins(0, 0, 0, 0)

        # Server Camera
        server_cam_container = QtWidgets.QWidget()
        server_cam_VLayout = QtWidgets.QVBoxLayout()

        server_cam_label = QtWidgets.QLabel("Left Camera")
        server_cam_label.setObjectName("label")

        self.label2 = QtWidgets.QLabel(self)  # Label for the server cam image

        # Client Camera
        client_cam_container = QtWidgets.QWidget()
        client_cam_VLayout = QtWidgets.QVBoxLayout()

        client_cam_label = QtWidgets.QLabel("Right Camera")
        client_cam_label.setObjectName("labelClient")

        # Load temporal images

        # Camera feeds
        image_scale_factor = 0.28
        self.label1 = QtWidgets.QLabel(self)
        self.label1.setObjectName("camImage")
        self.label2 = QtWidgets.QLabel(self)
        self.label2.setObjectName("camImage")

        f1, f2 = load_demo_images()
        self.image_size = (
            int(f1.shape[1] * image_scale_factor), int(f1.shape[0] * image_scale_factor))

        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
        f1 = cv2.resize(f1, self.image_size)
        f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
        f2 = cv2.resize(f2, self.image_size)

        self.image1 = QtGui.QImage(f1, f1.shape[1], f1.shape[0],
                                   f1.strides[0], QtGui.QImage.Format_RGB888)

        self.image2 = QtGui.QImage(f2, f2.shape[1], f2.shape[0],
                                   f2.strides[0], QtGui.QImage.Format_RGB888)

        self.label1.setPixmap(QtGui.QPixmap.fromImage(self.image1))
        self.label2.setPixmap(QtGui.QPixmap.fromImage(self.image2))
        self.label1.setAlignment(QtCore.Qt.AlignCenter)
        self.label2.setAlignment(QtCore.Qt.AlignCenter)

        # server_cam_VLayout.addWidget(server_cam_label)
        server_cam_VLayout.addWidget(self.label2)

        server_cam_container.setLayout(server_cam_VLayout)
        # client_cam_VLayout.addWidget(client_cam_label)
        client_cam_VLayout.addWidget(self.label1)

        client_cam_container.setLayout(client_cam_VLayout)

        #        server_cam_VLayout.setStretchFactor(self.label1, 10)
        #        client_cam_VLayout.setStretchFactor(self.label2, 10)

        camera_layout.addWidget(server_cam_container)

        camera_layout.addWidget(client_cam_container)
        camera_box.setLayout(camera_layout)

        control_triangLayout = QtWidgets.QHBoxLayout()
        control_triangLayout.addWidget(triangulation_box)
        control_triangLayout.addWidget(control_box)
        container_grid.addLayout(control_triangLayout)
        #        container_grid.addWidget(control_box)
        container_grid.addWidget(camera_box)

        # Triangulation box

        triangulation_layout = QtWidgets.QVBoxLayout()
        triangulation_layout.setContentsMargins(10, 20, 10, 0)

        z_ind_container = QtWidgets.QWidget()
        z_ind_layout = QtWidgets.QHBoxLayout()

        self.z_ind = positionWidget(500, 1500, 1000, error=100)
        z_ind_layout.addWidget(self.z_ind)

        self.z_ind_label = positionLabel("1000")
        self.z_ind_label.setObjectName("zLabel")

        self.z_ind_label.setAlignment(QtCore.Qt.AlignCenter)
        #        z_ind_label.setStyleSheet(""" QLabel {background-color: rgb(16, 159, 221);}""")
        #        z_ind_label.setContentsMargins(0,0,0,0)

        z_ind_layout.addWidget(self.z_ind_label)

        z_ind_container.setLayout(z_ind_layout)
        triangulation_layout.addWidget(z_ind_container)
        triangulation_layout.addSpacing(75)

        triangulation_box.setLayout(triangulation_layout)

        return self.camera_container

    def create_recording_group(self):
        """DEPRECIATED"""
        self.recording_box = QtWidgets.QGroupBox("Recording")
        box_grid = QtWidgets.QGridLayout()

        Record_time_label = QtWidgets.QLabel('Record Time')
        Record_time = QtWidgets.QDoubleSpinBox()
        Record_time.setValue(10)
        Record_time.setMinimum(0.000001)

        # Start recording button
        Start = QtWidgets.QPushButton("Start Recording")

        # Progress bar
        progress = QtWidgets.QProgressBar(self)

        box_grid.addWidget(Start, 0, 2, 1, 2)
        box_grid.addWidget(Record_time_label, 0, 0, 1, 1)
        box_grid.addWidget(Record_time, 0, 1, 1, 1)
        box_grid.addWidget(progress, 1, 0, 1, 4)

        box_grid.setSpacing(1)
        self.recording_box.setLayout(box_grid)

        return self.recording_box

    #        self.participant_list

    def create_keyboard(self):
        """Virtual Keyboard"""

        self.keyboard = virtualKeyboard()

        # NOW CONNECT ALL THE KEYS TO THEIR OUTPUTS
        self.keyboard.connect_to_buttons(
            self.Forename_edit.recieve_input)  # Connect this to the buttons
        self.keyboard.connect_to_buttons(
            self.Surname_edit.recieve_input)  # Connect this to the buttons
        self.keyboard.connect_to_buttons(
            self.ID_edit.recieve_input)  # Connect this to the buttons
        #        self.keyboard.connect_to_buttons(self.DOB_edit.recieve_input) #Connect this to the buttons

        self.keyboard.connect_to_buttons(self.Record_Time.recieve_input)

        self.Forename_edit.text_length.connect(self.keyboard.cap_zero_len)
        self.Surname_edit.text_length.connect(self.keyboard.cap_zero_len)
        self.ID_edit.text_length.connect(self.keyboard.cap_zero_len)
        #         spacebar.pressed.connect(self.Forename_edit.recieve_input)
        return self.keyboard

    def start_recording(self):
        """Will start the RPi Recording"""

        # Open a pop up window if the participant has been run
        run = self.check_participant_not_run()

        if run:
            return

            # Check a storage location has been set
        if self.storage_loc == '':
            storage_msgBox = QtWidgets.QMessageBox()
            storage_msgBox.setText(
                """A storage location has not been set.\n\nSelect a storage location by clicking Data Control -> Set Save Directory""")
            storage_msgBox.exec_()
            return

        self.recording_Terminate = False

        self.start_button.setText("Recording")
        # Update the images
        QtWidgets.QApplication.processEvents()

        if self.preview_live:
            self.toggle_preview_server()  # If we are in preview mode close the preview first

        t = self.Record_Time.value()
        print("RECORD NOW FOR {}".format(t))

        #        progress_thread = threading.Thread(target = self.progressBar_update, args = (t,))
        #        progress_thread.start()
        print("REQUEST RECORDING")
        self.backend_camera.TCP_client_start_UDP(
            t, 'testIR.h264')  # Starts the video recording
        self.backend_camera.TCP_client_request_timestamps()  # Request time stamps

        # Check for errors in the timestamps. If there is raise an error and don't save the recording
        # Check if there are any problems with the time stamps
        ts_error = self.check_timestamp_error()

        if ts_error:
            ts_msg = "There was a problem with the recording: One of the cameras dropped frames\nYou should rerun the recording. No data will be saved"

            ts_msgBox = QtWidgets.QMessageBox()
            ts_msgBox.setText(ts_msg)
            ts_msgBox.exec_()
            return

        print("ALL RECORDING DONE")
        if self.record_options_send_video.isChecked():
            self.backend_camera.TCP_client_request_video()  # Request video

        #        IR_process_thread = threading.Thread(target = self.backend_camera.TCP_client_request_IRPoints)
        #        IR_process_thread.start()
        self.recording_Terminate = True
        #        progress_thread.join()

        QtWidgets.QApplication.processEvents()

        if self.record_options_send_IRPoints.isChecked():
            # Request the IR points (Will process the data)
            self.backend_camera.TCP_client_request_IRPoints()

        self.start_button.setText("Start Recording")

        # Update the participant list csv to recorded for this entry
        self.update_participant_list()

        self.archive_files()
        print("Change Window")
        self.show_window_1()  # Go back to the demographics window

    def check_timestamp_error(self):
        """Check that the timestamps for the client and server are not more than 5 millisecond different at any point

            Return True if there are timing problems

            Return False otherwise
        """
        time_client = np.loadtxt("time_stamps.csv", delimiter=',')
        time_server = np.loadtxt("timestamps_server.csv", delimiter=',')

        time_client = time_client[:min(
            time_client.shape[0], time_server.shape[0])]

        time_server = time_server[:min(
            time_client.shape[0], time_server.shape[0])]
        time_client[time_client < 0] = np.NaN
        time_server[time_server < 0] = np.NaN

        if np.any(np.abs(time_client - time_server) > 5):

            return True
        else:
            return False

    def get_demographics(self):
        """Retrieve the demographics information"""

        #        self.Forename_edit.text()
        #        self.Surname_edit.text()
        #        self.ID_edit.text()
        #        self.DOB_edit.date()
        #
        demographics = {'forename': self.Forename_edit.text(), 'surname': self.Surname_edit.text(),
                        'DOB': self.DOB_edit.text(), 'ID': self.ID_edit.text(),
                        'Gender': self.Gender_edit.currentText(), 'Experiment': 'ExpOne',
                        'condition': self.Condition_edit.currentText(), 'RecTime': self.Record_Time.value()}

        return demographics

    #        self.Gender_edit.setCurrentIndex(0)
    #
    #
    #        self.DOB_edit.Date)

    def archive_files(self):
        """Move the recorded files to the appropriate directory"""

        # If self.storage_loc == None, it will save to the main directory. Else it will save to the selected location
        fos = FileOrderingSystem(self.storage_loc)

        demographics = self.get_demographics()

        # Call this to check the participant directory exists. If not make one
        fos.find_participant_dir(demographics)
        fos.find_participant_recording()
        fos.prep_record_directory()

        fos.move_files_to_record()

        pd.Series(demographics).to_csv(os.path.join(
            fos.recording_directory, 'demographics.csv'))

    def get_IR_data(self):
        """Request IR data is processed"""

        self.backend_camera.TCP_client_request_IRPoints()

    def progressBar_update(self, t):

        self.ProgressBar.setMaximum(t)

        t0 = time.time()

        while True:

            timer = time.time() - t0

            self.ProgressBar.setValue(timer)

            if self.recording_Terminate:
                break  # If recording terminate exit

    def toggle_preview_server(self):
        """Toggle the server video preview on and off"""

        if onDesktop:
            return

        print(self.preview_live)

        if self.preview_live:

            print("Request End Preview")
            self.preview_live = False
            self.end_preview()
            self.preview.setText("Start Preview")

            print(self.preview_live)

        elif not self.preview_live:

            print("ON TOGGLE")
            self.preview_live = True
            self.preview.setText("Stop Preview")
            self.start_preview()

    def preview_triangulate(self, img1, img2):
        """Report the position of any markers visable in the preview"""

        ir1 = ir_marker.find_ir_markers(img1, n_markers=2, it=0, tval=150)
        ir2 = ir_marker.find_ir_markers(img2, n_markers=2, it=0, tval=150)

        if ir1 != None:

            for mark in ir1:
                #                 print(mark['pos'], mark['radius'])
                cv2.circle(img1, (int(mark['pos'][0]), int(mark['pos'][1])), int(
                    mark['radius']), (255, 0, 0), 10)

        if ir2 != None:

            for mark in ir2:
                #                 print(mark['pos'], mark['radius'])
                cv2.circle(img2, (int(mark['pos'][0]), int(mark['pos'][1])), int(
                    mark['radius']), (255, 0, 0), 10)

        # 3d marker tracking
        if (ir1 != None) and (ir2 != None):
            ir1 = ir_marker.markers2numpy([ir1])  # Pass as list
            ir2 = ir_marker.markers2numpy([ir2])

            markers3d = self.stereo.triangulate_all_get_PL(
                ir1, ir2).squeeze()  # Get the marker positions in 3d space

            self.marker_mid = np.sum(markers3d, axis=0) / 2.0

            # Calculate the distance of the marker from the center of the two camereas. Calculate the vector between the midpoint along T and the observed point. Then take it's magnitude
            marker_distance = vector_magnitude(
                self.marker_mid - (self.stereo.T.flatten() / 2.0))

            self.z_ind.setValue(marker_distance)
            self.z_ind_label.setText(str(int(marker_distance)))

        return img1, img2

    def start_preview(self):
        """Start the camera preview from the server"""

        clientPreviewProc = backend.posturalProc(kind='client_preview')
        serverPreviewProc = backend.posturalProc(kind='server_preview')
        self.stereo = backend.stereo_process(
            clientPreviewProc, serverPreviewProc)

        # Request video preview from server
        self.backend_camera.TCP_client_request_videoPreview()
        self.backend_camera.videoPreview()  # Request video preview from client

        while self.preview_live:

            # Poll the latest camera images
            f1 = self.backend_camera.poll_videoPreview()
            f2 = self.backend_camera.TCP_client_poll_videoPreview()

            # Draw markers to screen and trinagulate
            if self.preview_options_triangulate.isChecked() and (f1 != None) and (f2 != None):
                f1, f2 = self.preview_triangulate(f1, f2)

            if (f1 == 'DEAD') or (f2 == "DEAD"):
                print("DEADDDDD")
                break

            # Hand videos to rendering functions
            if f1 != None:
                self.render_preview(f1, self.label1)

            if f2 != None:
                self.render_preview(f2, self.label2)

            # Update the images
            QtWidgets.QApplication.processEvents()

    def end_preview(self):
        """End the video preview"""
        self.backend_camera.TCP_client_stop_videoPreview()
        self.backend_camera.stop_videoPreview()

    def render_preview(self, frame, Qobj):
        """pass a frame and a QLabel object and render the frame to it"""

        img = cv2.resize(frame, self.image_size)  # Resize the image

        image2 = QtGui.QImage(img, img.shape[1], img.shape[0],
                              img.strides[0], QtGui.QImage.Format_RGB888)

        # This allows the garbage collector to release the memory used here. Fixes memory leak
        ctypes.c_long.from_address(id(img)).value = 1

        Qobj.setPixmap(QtGui.QPixmap.fromImage(image2))


if __name__ == '__main__':
    if not onDesktop:
        # Needs to be set else the multiprocessing in the camera_backend fails for some reason
        multiprocessing.set_start_method("spawn")
    app = QtWidgets.QApplication(sys.argv)
    #    app.autoSipEnabled()
    ex = MainWindow()

    ex.setStyleSheet("""QMainWindow {background: rgb(13,11,13)}
    
                        QGroupBox {background: rgb(52,50,52); border: 0px solid rgb(255,255,255); border-radius: 0px; margin-top: 0.5em;
                        font-family: embrima; font-size: 20px; font-weight: 500}
                        
                        
                        QGroupBox::title {color: 'white'; subcontrol-origin: margin; left: 4px; padding: 0 3px 0 3px;}
                        
                        QLabel {color: 'white'; font-family: embrima; font-size: 18px }       
                        
                        QDialog {background: rgb(52,50,52); border-radius: 0px; border: 0px solid rgb(16, 159, 221);}
                        
                        QLineEdit {background: rgb(52,50,52); color: rgb(255, 255, 255); font-weight: 800;
                        border: 1px solid rgb(16, 159, 221); font-family: embrima; font-size: 15px; padding: 3px;}    
                        
                        QLineEdit#MyTimeLineEdit {border: 1px solid rgb(221,89,2);}
                        
                        QCheckBox {color: rgb(255, 255, 255); font-weight: 800; font-family: embrima; font-size: 15px; padding: 3px;}  
                        
                        QCheckBox::indicator {width: 10px; height: 10px;border: 1px solid rgb(0,0,0); background-color: rgb(255,255,255)}
                        
                        QCheckBox::indicator:checked { background-color: rgb(221, 89, 2);}
                              
                        MyDateLineEdit {background: rgb(52,50,52); color: rgb(255, 255, 255); font-weight: 800;
                        border: 1px solid rgb(221, 89, 2); font-family: embrima; font-size: 15px; padding: 3px;}   
                        
                        QDateEdit {background: rgb(52,50,52); color: rgb(255, 255, 255); font-weight: 800;
                        border: 1px solid rgb(16, 159, 221); font-family: embrima; font-size: 15px; padding: 3px;}
                        
                                       
                        QDoubleSpinBox {background: rgb(0,0,0); color: 'white';}
                        
                        
                        QPushButton {background: rgb(221,89,2); color: 'white'; font-family: embrima; font-size: 20px; font-weight: 500;
                                     border-radius: 0px; border: 0px solid rgb(0,0,0); padding: 2px;}
                                     
                                     
                        QPushButton:pressed {background-color: rgb(255, 105, 0); border-style: inset;}
                        
                        
                        
                        
                        
                        QToolButton {border-radius: 5px}
                        
                        QComboBox {background: rgb(52,50,52); color: rgb(255, 255, 255); font-weight: 800; font-size: 15px;
                        border: 1px solid rgb(16, 159, 221); padding: 3px}
                        
                        QComboBox QListView {background-color:rgb(13,11,13);  border: 1px solid rgb(16, 159, 221);}   
                        
                        Icon_Button {background-color: rgb(52,50,52); border: 1px solid rgb(91, 204, 58); border-radius: 0px;} 
                        
                        Text_Button {background-color: rgb(52,50,52); color: rgb(255,255,255); border: 1px solid rgb(91, 204, 58); 
                                     border-radius: 0px; font-family: embrima; font-size: 17px; font-weight: 700} 
                       
                        QWidget#camera_checkbox {border: 1px solid rgb(221,89,2); }
                        QLabel#label {color: rgb(16, 159, 221);}
                        QLabel#labelClient {color: rgb(221, 89, 2);}
                        
                        QPushButton#controlButton {padding: 5px;}
                        positionLabel {background-color: rgb(16, 159, 221); font-size: 24px; font-weight: 800; }     
                        
                        QMenuBar {background-color: rgb(52,50,52); color: rgb(255,255,255)}
                        
                        QAction {background-color: rgb(52,50,52); color: rgb(255,255,255)}
                        
                        recordingLocLabel {background-color: rgb(52,50,52); font-size: 10px; padding: 5px}

                                         
                        """)  # Change the background color
    if not onDesktop:
        ex.showFullScreen()  # Turn this on on the RPi

    sys.exit(app.exec_())

#    f1, f2 =  load_demo_images()
