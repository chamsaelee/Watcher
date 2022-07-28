import os, sys, time
import tkinter as tk
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
import pandas as pd
import numpy as np
import csv

class Ui_MainWindow(object):
    def __init__(self):
        super().__init__()
        self.timer = QTimer()
        self.timer.start(1000)
        self.timer.timeout.connect(self.printtime)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 800)
        MainWindow.setMinimumSize(QtCore.QSize(800, 800))
        MainWindow.setMaximumSize(QtCore.QSize(800, 800))
        MainWindow.setWindowIcon(QtGui.QIcon('image/label logo.png'))
        MainWindow.setStyleSheet("background-color:white;")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 10, 201, 81))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(150, 120, 521, 611))
        self.label_2.setObjectName("label_2")
        self.labeltime = QtWidgets.QLabel(self.centralwidget)
        self.labeltime.setGeometry(QtCore.QRect(290, 120, 211, 51))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        self.labeltime.setFont(font)
        self.labeltime.setObjectName("labeltime")
        self.labelvalue = QtWidgets.QLabel(self.centralwidget)
        self.labelvalue.setGeometry(QtCore.QRect(310, 330, 181, 171))
        font = QtGui.QFont()
        font.setFamily("Hancom Gothic")
        font.setPointSize(48)
        font.setBold(True)
        font.setWeight(75)
        self.labelvalue.setFont(font)
        self.labelvalue.setObjectName("labelvalue")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(380, 470, 51, 61))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setPixmap(QtGui.QPixmap("image/Original_Logo_small.png"))
        self.label_2.setPixmap(QtGui.QPixmap("image/g.png"))
        self.labeltime.setText(_translate("MainWindow", "6월 16일   17 : 10"))
        self.labelvalue.setText(_translate("MainWindow", "000"))
        self.label_3.setText(_translate("MainWindow", "준비"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))

    def printtime(self):
        result = pd.read_excel("result.xlsx")
        result = result.iloc[-1, :]
        predict = result["value"]
        self.labeltime.setText(result["time"])
        if predict < 70:
            predict = str(predict).rjust(3)
            self.label_2.setPixmap(QtGui.QPixmap("image/r.png"))
            self.labelvalue.setText(str(predict))
            self.label_3.setText("위험")
            self.label_3.setStyleSheet("Color : red")
        elif predict < 80:
            predict = str(predict).rjust(3)
            self.label_2.setPixmap(QtGui.QPixmap("image/y.png"))
            self.labelvalue.setText(str(predict))
            self.label_3.setText("주의")
            self.label_3.setStyleSheet("Color : orange")
        elif predict < 170:
            predict = str(predict).rjust(3)
            self.label_2.setPixmap(QtGui.QPixmap("image/g.png"))
            self.labelvalue.setText(str(predict))
            self.label_3.setText("양호")
            self.label_3.setStyleSheet("Color : green")
        elif predict < 200:
            predict = str(predict).rjust(3)
            self.label_2.setPixmap(QtGui.QPixmap("image/y.png"))
            self.labelvalue.setText(str(predict))
            self.label_3.setText("주의")
            self.label_3.setStyleSheet("Color : orange")
        else:
            predict = str(predict).rjust(3)
            self.label_2.setPixmap(QtGui.QPixmap("image/r.png"))
            self.labelvalue.setText(str(predict))
            self.label_3.setText("위험")
            self.label_3.setStyleSheet("Color : red")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())