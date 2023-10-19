# -*- coding: utf-8 -*-
"""

Postural Sway Assesment Tool GUI.

"""

import sys

import cv2
from PyQt5 import QtGui, QtCore, QtWidgets

import posturalCam_master_NETWORK as PSAT_core


def load_demo_images():
    """For demo purposes we can load 2 images"""
    frame1 = cv2.imread('calib_img_0.tiff')
    frame2 = cv2.imread('calib_img_1.tiff')

    return frame1, frame2


class Button(QtWidgets.QToolButton):
    """A button class for the keyboard"""

    pressed = QtCore.pyqtSignal(str)

    def __init__(self, text, parent=None):
        super(Button, self).__init__(parent)

        self.text = text
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Preferred)
        self.setText(text)

    def sizeHint(self):
        size = super(Button, self).sizeHint()
        size.setHeight(size.height() + 20)
        size.setWidth(max(size.width(), size.height()))
        return size

    def mousePressEvent(self, e):
        self.pressed.emit(self.text)
        print("Key: {}".format(self.text))


class MyLineEdit(QtWidgets.QLineEdit):
    """A child of the QLineEdit that emits a signal when it is pressed"""

    pressed = QtCore.pyqtSignal()  # Signal to emit when mousepress focused
    end_focus = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(MyLineEdit, self).__init__(parent)

        self.text = ''

        self.in_focus = None

    def mousePressEvent(self, e):

        self.pressed.emit()  # Emit a signal when the key is pressed
        print("Key Pressed")

    def focusInEvent(self, e):

        self.in_focus = True

        print("IN")

    def focusOutEvent(self, e):

        self.in_focus = False
        self.end_focus.emit()  # Emit signal that focus was lost
        print("OUT")

    @QtCore.pyqtSlot(str)
    def recieve_input(self, inp):

        if self.in_focus:

            print("Recieved key {}".format(inp))

            if inp == 'Spacebar':
                self.text += ' '
            elif inp == 'backspace':
                self.text = self.text[:-1]
            else:
                self.text += inp

            self.setText(self.text)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):

        self.setGeometry(400, 240, 800, 480)
        self.setWindowTitle('Postural Sway Assessment Tool')
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Cleanlooks'))

        self.create_widgets()

        self.show()

    def create_widgets(self):

        self.central_widget = QtWidgets.QWidget()

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.create_demographics_group(), 0, 0, 1, 2)
        grid.addWidget(self.create_camera_group(), 1, 0, 3, 2)
        grid.addWidget(self.create_recording_group(), 0, 3, 1, 3)

        # Exit button
        Shutdown = QtWidgets.QPushButton("Shut down")
        # New style shutdown function
        Shutdown.clicked.connect(self.shutdown_event)

        grid.addWidget(Shutdown, 4, 4, 1, 2)
        Preview_btn = QtWidgets.QPushButton("Start Camera Preview")
        grid.addWidget(Preview_btn, 4, 0, 1, 1)

        #        QtCore.QObject.connect(Shutdown, QtCore.pyqtSignal('clicked()'), self.shutdown_event) #Call shutdown function Old style
        grid.addWidget(self.create_keyboard(), 1, 0, 4, 6)

        self.central_widget.setLayout(grid)
        self.setCentralWidget(self.central_widget)

        # Demographics box

    def shutdown_event(self):
        """Shut down button action. Opens a question box to ask if you want to power down"""
        shutdown_msg = "Are you sure you want to power down PSAT?"

        reply = QtWidgets.QMessageBox.question(
            self, "Shutdown", shutdown_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            # Quit Application. This will close down the RPis in future
            QtCore.QCoreApplication.instance().quit()
        else:
            print("Not closing")

    def show_keyboard(self):

        print("SHOW")
        self.keyboard.show()

    def hide_keyboard(self):

        self.keyboard.hide()
        print("HIDE")

    def create_demographics_group(self):

        demographicsBox = QtWidgets.QGroupBox("Demographics")

        # Labels and buttons
        Forename_label = QtWidgets.QLabel('First Name(s)')
        self.Forename_edit = MyLineEdit(self)
        self.Forename_edit.pressed.connect(self.show_keyboard)
        self.Forename_edit.end_focus.connect(self.hide_keyboard)

        Surname_label = QtWidgets.QLabel('Surname')
        self.Surname_edit = MyLineEdit(self)
        self.Surname_edit.pressed.connect(self.show_keyboard)
        self.Surname_edit.end_focus.connect(self.hide_keyboard)

        Gender = QtWidgets.QLabel('Gender')
        self.Gender_edit = QtWidgets.QComboBox()
        self.Gender_edit.addItem("Male")
        self.Gender_edit.addItem("Female")

        ID = QtWidgets.QLabel("ID")
        self.ID_edit = MyLineEdit(self)
        self.ID_edit.pressed.connect(self.show_keyboard)
        self.ID_edit.end_focus.connect(self.hide_keyboard)

        # DOB should be a date format or a box to select the date
        DOB = QtWidgets.QLabel('DOB')
        self.DOB_edit = QtWidgets.QDateEdit()

        check_ID = QtWidgets.QPushButton("Verify ID")

        demographics_grid = QtWidgets.QGridLayout()

        demographics_grid.addWidget(Forename_label, 1, 0, 1, 1)
        demographics_grid.addWidget(self.Forename_edit, 1, 1, 1, 1)

        demographics_grid.addWidget(Surname_label, 1, 2, 1, 1)
        demographics_grid.addWidget(self.Surname_edit, 1, 3, 1, 1)

        demographics_grid.addWidget(Gender, 1, 4, 1, 1)
        demographics_grid.addWidget(self.Gender_edit, 1, 5, 1, 1)

        demographics_grid.addWidget(ID, 2, 0, 1, 1)
        demographics_grid.addWidget(self.ID_edit, 2, 1, 1, 1)

        demographics_grid.addWidget(DOB, 2, 2, 1, 1)
        demographics_grid.addWidget(self.DOB_edit, 2, 3, 1, 1)

        demographics_grid.addWidget(check_ID, 3, 0, 1, 1)

        demographics_grid.setSpacing(10)

        demographicsBox.setLayout(demographics_grid)

        return demographicsBox

    def create_camera_group(self):

        box = QtWidgets.QGroupBox("Camera Status")

        box_grid = QtWidgets.QGridLayout()
        #        box_grid.addWidget(sc)

        # Camera feeds
        image_size = (220, 150)
        label1 = QtWidgets.QLabel(self)
        label2 = QtWidgets.QLabel(self)

        f1, f2 = load_demo_images()
        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
        f1 = cv2.resize(f1, image_size)
        f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
        f2 = cv2.resize(f2, image_size)

        image1 = QtGui.QImage(f1, f1.shape[1], f1.shape[0],
                              f1.strides[0], QtGui.QImage.Format_RGB888)

        image2 = QtGui.QImage(f2, f2.shape[1], f2.shape[0],
                              f2.strides[0], QtGui.QImage.Format_RGB888)

        label1.setPixmap(QtGui.QPixmap.fromImage(image1))
        label2.setPixmap(QtGui.QPixmap.fromImage(image2))

        # Buttons

        box_grid.addWidget(label1, 0, 0, 1, 2)
        box_grid.addWidget(label2, 0, 2, 1, 2)

        box_grid.setSpacing(20)
        box.setLayout(box_grid)

        return box

    def create_recording_group(self):

        box = QtWidgets.QGroupBox("Recording")
        box_grid = QtWidgets.QGridLayout()

        Record_time_label = QtWidgets.QLabel('Record Time')
        Record_time = QtWidgets.QDoubleSpinBox()
        Record_time.setValue(10)
        Record_time.setMinimum(0.000001)

        # Start recording button
        Start = QtWidgets.QPushButton("Start Recording")
        Start.clicked.connect(lambda: PSAT_core.main(
            Record_time.value()))  # New style shutdown function

        # Progress bar
        progress = QtWidgets.QProgressBar(self)

        box_grid.addWidget(Start, 0, 2, 1, 2)
        box_grid.addWidget(Record_time_label, 0, 0, 1, 1)
        box_grid.addWidget(Record_time, 0, 1, 1, 1)
        box_grid.addWidget(progress, 1, 0, 1, 4)

        box_grid.setSpacing(1)
        box.setLayout(box_grid)

        return box

    def create_keyboard(self):
        """Virtual Keyboard"""
        self.keyboard = QtWidgets.QDialog()

        key_labels = [['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '_', 'backspace'],
                      ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
                      ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
                      ['z', 'x', 'c', 'v', 'b', 'n', 'm']]

        key_grid = QtWidgets.QGridLayout()

        keys = []  # List of key buttons
        for row in range(len(key_labels)):
            for k in range(len(key_labels[row])):
                key_ = Button(key_labels[row][k])

                # It must connect to each line input
                key_.pressed.connect(self.Forename_edit.recieve_input)
                key_.pressed.connect(self.Surname_edit.recieve_input)
                key_.pressed.connect(self.ID_edit.recieve_input)

                keys.append(key_)

                key_.setStyleSheet("""background-color: rgba(87, 87, 87, 70%); 
                                    border: 0px;
                                    color: rgb(255,255,255);""")

                key_grid.addWidget(key_, row, k)

        spacebar = Button("Spacebar")
        spacebar.pressed.connect(self.Forename_edit.recieve_input)
        spacebar.pressed.connect(self.Surname_edit.recieve_input)
        spacebar.pressed.connect(self.ID_edit.recieve_input)

        spacebar.setStyleSheet("""background-color: rgba(87, 87, 87, 70%); 
                                    border: 0px;
                                    color: rgb(255,255,255);""")

        key_grid.addWidget(spacebar, row + 1, 1, 1, 8)

        self.keyboard.setLayout(key_grid)

        self.keyboard.setStyleSheet("background-color: rgba(0, 0, 0, 100%);")

        self.keyboard.hide()  # Make the keyboard invisible

        #        self.Forename_edit.hasFocus()

        #        self.Forename_edit.cursorPositionChanged.connect(self.toggle_keyboard)

        return self.keyboard


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    #    app.autoSipEnabled()
    ex = MainWindow()

    sys.exit(app.exec_())

#    f1, f2 =  load_demo_images()
