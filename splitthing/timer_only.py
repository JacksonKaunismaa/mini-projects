#!/usr/bin/env python3

from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5 import QtGui
import sys
from pynput import keyboard
import time
import threading as th
import os

BIG_FONT = "64px"
SMALL_FONT = "32px"
GREEN = "#3fc611"
GRAY = "#a7aaaa"
BLUE = "#0d99bc"

class Window(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("SplitThing")
        self.setWindowIcon(QtGui.QIcon(os.path.dirname(os.path.realpath(__file__))+"/clock2.png"))
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        QGraphicsView().setStyleSheet("background:transparent;")


        layout = QGridLayout()
        self.setLayout(layout)
        layout.setHorizontalSpacing(0)

        self.timer_secs = QLabel()
        layout.addWidget(self.timer_secs, 0, 0)
        self.timer_secs.setAlignment(QtCore.Qt.AlignRight)

        self.timer_milli = QLabel()
        layout.addWidget(self.timer_milli, 0, 1)

        self.done = False

        self.time_thread = th.Thread(target=self.updater, daemon=True)
        self.time_thread.start()


        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet("QWidget {background:transparent;}\
                            p {font-family: Andale Mono; \
                               background: transparent}")

        self.last_check = 0
        self.running = False
        self.total_time = 0

        self.reset_timer()

        self.oldPos = self.pos()

    def updater(self):
        while not self.done:
            time.sleep(0.05)
            if self.running:
                self.total_time += time.time() - self.last_check
                self.last_check = time.time()
                self.timer_update(self.total_time, GREEN)

    def pause_unpause(self):
        if self.running:
            self.pause_timer()
        else:
            self.running = True
            self.last_check = time.time()

    def reset_timer(self):
        self.running = False
        self.total_time = 0
        self.timer_update(self.total_time, GRAY)

    def pause_timer(self):
        self.running = False
        self.timer_update(self.total_time, BLUE)

    def clean_up(self):
        self.done = True
        self.time_thread.join()


    def timer_update(self, seconds, color):
        if not seconds:
            seconds = 0
        hours = int(seconds // 3600)
        seconds -= hours *3600
        minutes = int(seconds // 60)
        seconds -= minutes*60
        full_seconds = int(seconds)
        milliseconds = int((seconds - full_seconds)*1e3)

        text_format = f"color:{color}; font-size: "
        time_secs = f'<p style="{text_format}{BIG_FONT};">'

        if hours:
            time_secs += f"{hours}:"
        if minutes:
            time_secs += f"{minutes:02}:" if hours else f"{minutes}:"

        time_secs += f'{full_seconds:02}</p>' if minutes else f'{full_seconds}</p>'
        time_milli = f'<p style="{text_format}{SMALL_FONT};">.{milliseconds:03}</p>'

        self.timer_secs.setText(time_secs)
        self.timer_milli.setText(time_milli)

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QtCore.QPoint(event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()



app = QApplication(sys.argv)

screen = Window()
screen.show()


def on_enter():
    if not lock.is_set():
        screen.pause_unpause()
        reset_allowed.set()

def on_bslash():
    screen.reset_timer()

lock = th.Event()
reset_allowed = th.Event()
h =  keyboard.GlobalHotKeys({
    "<ctrl>+<alt>+<enter>": on_enter,
    "<ctrl>+<alt>+<backspace>": on_bslash})
h.start()
try:
    app.exec_()     # not the best practice
except KeyboardInterrupt:
    print("hi")
    os._exit(0)
    h.stop()
    h.join()
    screen.clean_up()

