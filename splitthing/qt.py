#!/usr/bin/env python3

from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5 import QtGui
import sys
from pynput import keyboard
import time
import threading as th
import os
import json
import glob

BIG_FONT = "64px"
SMALL_FONT = "32px"
GREEN = "#3fc611"
GRAY = "#a7aaaa"
BLUE = "#0d99bc"
DELAY = 0.05
PRESS_TIME = 0.021
IGT_DELAY = 5.0

kctl = keyboard.Controller()
#minecraft_stats = "/home/test/.minecraft/spedrun-worlds/saves/*/stats/*.json"
minecraft_stats = "/home/test/E_Drive/Garbage/multimc/install/bin/instances/1.16.1/.minecraft/saves/*/stats/*.json"

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


        self.igt_secs = QLabel()
        layout.addWidget(self.igt_secs, 1, 0)
        self.igt_secs.setAlignment(QtCore.Qt.AlignRight)

        self.igt_milli = QLabel()
        layout.addWidget(self.igt_milli, 1, 1)



        self.done = False

        self.time_thread = th.Thread(target=self.updater, daemon=True)
        self.time_thread.start()

        self.most_recent = ""
        self.igt_thread = th.Thread(target=self.igt_getter, daemon=True)
        self.igt_thread.start()
        self.last_igts = {}
        self.world_name = ""

        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet("QWidget{background:transparent;}\
                            p {font-family: Andale Mono; \
                               background: transparent}")

        self.last_check = 0
        self.running = False
        self.total_time = 0

        self.reset_timer()

        self.oldPos = self.pos()

    def igt_getter(self):
        while not self.done:
            time.sleep(IGT_DELAY)
            all_stats = glob.glob(minecraft_stats)
            if self.most_recent:
                all_stats.insert(0, self.most_recent) # check the most recently changed one more frequently
            for fname in all_stats:
                curr_igt = get_igt(fname)
                try:
                    if self.last_igts[fname] != curr_igt and curr_igt:
                        self.world_name = fname
                        self.last_igts[fname] = curr_igt
                        self.igt_update(curr_igt)
                        self.most_recent = fname
                        break
                except KeyError:
                    self.last_igts[fname] = curr_igt



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
        self.igt_update(0)

    def pause_timer(self):
        self.running = False
        self.timer_update(self.total_time, BLUE)

    def clean_up(self):
        self.done = True
        self.time_thread.join()
        self.igt_thread.join()

    def igt_update(self, seconds):
        hours = int(seconds // 3600)
        seconds -= hours *3600
        minutes = int(seconds // 60)
        seconds -= minutes*60
        full_seconds = int(seconds)
        milliseconds = int((seconds - full_seconds)*1e3)

        text_format = f"color: {GREEN}; font-size: {SMALL_FONT};"
        time_secs = f'<p style="{text_format}">'

        if hours:
            time_secs += f"{hours}:"
        if minutes:
            time_secs += f"{minutes:02}:" if hours else f"{minutes}:"

        time_secs += f'{full_seconds:02}</p>' if minutes else f'{full_seconds}</p>'
        time_milli = f'<p style="{text_format}">.{milliseconds:03}</p>'

        self.igt_secs.setText(time_secs)
        self.igt_milli.setText(time_milli)


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

def tap(key, amount=1, mod=None):
    if mod:
        kctl.press(mod)
        time.sleep(PRESS_TIME)
    for _ in range(amount):
        kctl.press(key)
        time.sleep(PRESS_TIME)
        kctl.release(key)
        time.sleep(DELAY)
    if mod:
        kctl.release(mod)

def create_world():
    #tap(keyboard.Key.esc, 3)
    #tap(keyboard.Key.tab)
    #tap(keyboard.Key.enter)
    #tap(keyboard.Key.tab, 3)
    #tap(keyboard.Key.enter)
    #tap(keyboard.Key.tab, 2)
    #tap(keyboard.Key.enter, 3)
    #tap(keyboard.Key.tab, 5)
    #tap(keyboard.Key.enter)
    if reset_allowed.is_set():
        lock.set()
        tap(keyboard.Key.esc)
        tap(keyboard.Key.tab, mod=keyboard.Key.shift)
        tap(keyboard.Key.enter)
        lock.clear()
        on_bslash()
        reset_allowed.clear()


def get_igt(filename):
    try:
        with open(filename, "r") as f:
            stats = json.loads(f.read())
            return float(stats["stats"]["minecraft:custom"]["minecraft:play_one_minute"])/20
    except:
        return None

lock = th.Event()
reset_allowed = th.Event()
h =  keyboard.GlobalHotKeys({
    "<ctrl>+<alt>+<enter>": on_enter,
    "<ctrl>+<alt>+<backspace>": on_bslash,
    "<ctrl>+<alt>+y": create_world})
h.start()
try:
    app.exec_()     # not the best practice
except KeyboardInterrupt:
    print("hi")
    os._exit(0)
    h.stop()
    h.join()
    screen.clean_up()

