#!/usr/bin/env python3

from pynput import keyboard
import time

kctl = keyboard.Controller()
PRESS_TIME = 0.021
DELAY = 0.05

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
    tap(keyboard.Key.esc)
    tap(keyboard.Key.tab, mod=keyboard.Key.shift)
    tap(keyboard.Key.enter)


h =  keyboard.GlobalHotKeys({
    "<ctrl>+<alt>+y": create_world})
h.start()
h.join()
