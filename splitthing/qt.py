#!/usr/bin/env python3

from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QPoint
from PyQt5 import QtGui
import sys
import argparse
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

class DragLabel(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.parent = parent

    def mousePressEvent(self, event):
        # self.parent.inWindowPos = event.pos()
        self.parent.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            # print('event pos', event.pos())
            self.parent.mouseMoveEvent(event)
        # if event.buttons() == QtCore.Qt.LeftButton:
        #     self.parent.setGeometry(event.globalPos().x()-self.parent.inWindowPos.x(), 
        #                     event.globalPos().y()-self.parent.inWindowPos.y(), 
        #                     self.width(), self.height())

class Window(QWidget):
    def __init__(self, args):
        QWidget.__init__(self)
        self.setWindowTitle("SplitThing")
        self.setWindowIcon(QtGui.QIcon(os.path.dirname(os.path.realpath(__file__))+"/clock2.png"))
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        QGraphicsView().setStyleSheet("background:transparent;")
        self.done = False
        self.track_igt = args.mc
        self.track_rta = args.mc or args.timer
        
        if self.track_rta:  # RTA timer
            layout = QGridLayout()
            self.setLayout(layout)
            layout.setHorizontalSpacing(0)

            self.timer_secs = DragLabel(self)
            layout.addWidget(self.timer_secs, 0, 0)
            self.timer_secs.setAlignment(QtCore.Qt.AlignRight)

            self.timer_milli = DragLabel(self)
            layout.addWidget(self.timer_milli, 0, 1)


        if self.track_igt:  # IGT timer
            self.igt_secs = DragLabel(self)
            layout.addWidget(self.igt_secs, 1, 0)
            self.igt_secs.setAlignment(QtCore.Qt.AlignRight)

            self.igt_milli = DragLabel(self)
            layout.addWidget(self.igt_milli, 1, 1)

            self.most_recent = ""
            self.igt_thread = th.Thread(target=self.igt_getter, daemon=True)
            self.igt_thread.start()
            self.last_igts = {}
            self.world_name = ""

        self.time_thread = th.Thread(target=self.updater, daemon=True)
        self.time_thread.start()

        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet("QWidget{background:transparent;}\
                            p {font-family: Andale Mono; \
                               background: transparent}")

        self.last_check = 0
        self.running = False
        self.total_time = 0

        self.reset_timer()
        self.press_control = 0
        self.inWindowPos = QPoint()
        

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
                self.total_time += (time.time() - self.last_check)
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
        if self.track_igt:
            self.igt_update(0)

    def pause_timer(self):
        self.running = False
        self.timer_update(self.total_time, BLUE)

    def clean_up(self):
        self.done = True
        self.time_thread.join()
        self.igt_thread.join()

    def format_time(self, seconds, secs_font, color):
        hours = int(seconds // 3600)
        seconds -= hours *3600
        minutes = int(seconds // 60)
        seconds -= minutes*60
        full_seconds = int(seconds)
        milliseconds = int((seconds - full_seconds)*1e3)

        text_format = f"color:{color}; font-size: "
        time_secs = f'<p style="{text_format}{secs_font};">'

        if hours:
            time_secs += f"{hours}:{minutes:02}:{full_seconds:02}"
        elif minutes:
            time_secs += f"{minutes}:{full_seconds:02}"
        else:
            time_secs += f"{full_seconds}"
        time_secs += "</p>"
        time_milli = f'<p style="{text_format}{SMALL_FONT};">.{milliseconds:03}</p>'
        return time_secs, time_milli
        
    def igt_update(self, seconds):
        time_secs, time_milli = self.format_time(seconds, SMALL_FONT, GREEN)
        self.igt_secs.setText(time_secs)
        self.igt_milli.setText(time_milli)


    def timer_update(self, seconds, color):
        if not seconds:
            seconds = 0
        time_secs, time_milli = self.format_time(seconds, BIG_FONT, color)
        self.timer_secs.setText(time_secs)
        self.timer_milli.setText(time_milli)

    def mousePressEvent(self, event):
        self.inWindowPos = self.pos() - event.globalPos()

    def mouseMoveEvent(self, event):
        self.setGeometry(event.globalPos().x()+self.inWindowPos.x(), 
                        event.globalPos().y()+self.inWindowPos.y(), 
                        self.width(), self.height())


def on_enter():
    """Pause/unpause the timer"""
    if not lock.is_set():
        screen.pause_unpause()
        reset_allowed.set()

def on_bslash():
    """Reset the timer"""
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
    """Create a new world"""
    if not args.reset:
        if reset_allowed.is_set():
            lock.set()
            tap(keyboard.Key.esc)
            tap(keyboard.Key.tab, mod=keyboard.Key.shift)
            tap(keyboard.Key.enter)
            lock.clear()
            on_bslash()
            reset_allowed.clear()
    else:
        tap(keyboard.Key.esc)
        tap(keyboard.Key.tab, mod=keyboard.Key.shift)
        tap(keyboard.Key.enter)


def get_igt(filename):
    try:
        with open(filename, "r") as f:
            stats = json.loads(f.read())
            return float(stats["stats"]["minecraft:custom"]["minecraft:play_one_minute"])/20
    except:
        return None
    
parser = argparse.ArgumentParser()

# Create a mutually exclusive group for the modes
mode_group = parser.add_mutually_exclusive_group()

# Add the options for the different modes
mode_group.add_argument("--mc", action="store_true", help="Enable timer and resetting")
mode_group.add_argument("--reset", action="store_true", help="Enable resetting only")
mode_group.add_argument("--timer", action="store_true", help="Enable timer only")

args = parser.parse_args()

if not args.reset:  # if dont need the timer, dont create it
    app = QApplication(sys.argv)
    screen = Window(args)
    screen.show()

lock = th.Event()
reset_allowed = th.Event()


if args.mc:
    hkey_map = {
        "<ctrl>+<alt>+=": on_enter,
        "<ctrl>+<alt>+-": on_bslash,
        "<ctrl>+<alt>+y": create_world,
    }
elif args.reset:
    hkey_map = {
        "<ctrl>+<alt>+y": create_world,
    }
elif args.timer:
    hkey_map = {
        "<ctrl>+<alt>+=": on_enter,
        "<ctrl>+<alt>+-": on_bslash,
    }

h =  keyboard.GlobalHotKeys(hkey_map)
h.start()
try:
    for k in hkey_map:
        print(k, hkey_map[k].__doc__)
    if not args.reset:  # if we're not just resetting, create the window for the timer
        app.exec_()
    else:
        h.join()
except KeyboardInterrupt:
    print("hi")
    os._exit(0)
    h.stop()
    h.join()
    screen.clean_up()

