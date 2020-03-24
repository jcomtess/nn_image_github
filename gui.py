import PyQt5
import sys
import settings
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QHBoxLayout,
                             QVBoxLayout, QMainWindow, QComboBox,
                             QLabel, QApplication)
                            # QDesktopWidget, QCalendarWidget, QLCDNumber, QSlider, QGridLayout, QTextEdit,
                            #  QFrame, QAction, QSplitter, QStyleFactory, QLineEdit, qApp)
from PyQt5.QtGui import (QFont, QPainter, QColor, QPen, QBrush)
                        # QIcon, QPixmap, QDrag)
from PyQt5.QtCore import (Qt, QObject, pyqtSignal, QBasicTimer)
                        # QMimeData, QDate, pyqtSlot, QRect, QPoint)
class Main_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.main_signal = pyqtSignal()
        self.image01lbl  = QLabel('image 01')
        self.image01widg = QLabel('me')
        self.image01widg.setFixedSize(32, 32)
        self.image01widg.setMinimumSize(32, 32)
        self.image01widg.setMaximumSize(64, 64)

        self.image02lbl  = QLabel('image 02')
        self.image02widg = QLabel('has')
        self.image02widg.setFixedSize(32, 32)
        self.image02widg.setMinimumSize(32, 32)
        self.image02widg .setMaximumSize(64, 64)

        self.image03lbl  = QLabel('image 03')
        self.image03widg = QLabel('some')
        self.image01widg.setFixedSize(32, 32)
        self.image03widg.setMinimumSize(32, 32)
        self.image03widg.setMaximumSize(64, 64)

        self.button = QPushButton()
        self.button.resize(self.button.sizeHint())
        self.button.clicked.connect(self.action)

        self.n_epoch_to_train = 10
        self.button_train = QPushButton('Train ' + str(self.n_epoch_to_train) + ' Epoch')
        self.button_train.resize(self.button_train.sizeHint())
        self.button_train.clicked.connect(self.start_train)

        self.button_show = QPushButton('Show image ')
        self.button_show.resize(self.button_show.sizeHint())
        self.button_show.clicked.connect(self.show_image)
        self.initUI()
        
    def initUI(self):
        vbox01 = QVBoxLayout()
        vbox01.addWidget(self.image01lbl)
        vbox01.addWidget(self.image01widg)
        vbox01.addWidget(self.image02lbl)
        vbox01.addWidget(self.image02widg)
        vbox01.addWidget(self.image03lbl)
        vbox01.addWidget(self.image03widg)
        vbox01.addStretch(1)
        vbox02 = QVBoxLayout()
        vbox02.addWidget(self.button_train)
        vbox02.addWidget(self.button_show)
        vbox02.addStretch(1)
        hbox = QHBoxLayout()
        hbox.addLayout(vbox01)
        hbox.addLayout(vbox02)
        hbox.addStretch(1)

        self.resize(800, 600)
        self.move(5, 5)
        self.setWindowTitle(settings.Name)
        self.setLayout(hbox)
        self.show()

    def action(self):
        print('some')
    def start_train(self):
        print('train')
    def show_image(self):
        print('show images')
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Main_Window()
    sys.exit(app.exec_())