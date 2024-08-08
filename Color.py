import pyautogui
import cv2
import numpy as np
import pygetwindow as gw

class Color:
    """
    A simple class to represent an RGB color with priority.
    """
    def __init__(self, r, g, b, priority=0, size='Small'):
        self.r = r
        self.g = g
        self.b = b
        self.priority = priority
        self.size = size

    def __repr__(self):
        return f"Color(RGB=({self.r}, {self.g}, {self.b}), Priority={self.priority}, Size={self.size})"


class Cubes:
    """
    A class to store RGB values and priorities for different cube types.
    """
    SmallGreen = Color(109, 243, 162, priority=2, size='Small')
    SmallBlue = Color(104, 212, 243, priority=2, size='Small')
    MediumGreen = Color(82, 243, 76, priority=2, size='Medium')
    LargeGreen = Color(157, 253, 208, priority=3, size='Large')
    SmallGold = Color(241, 245, 116, priority=2, size='Small')

    SmallGlitch = Color(243, 74, 237, priority=6, size='Small')
    WeirdBulb = Color(174, 104, 154, priority=5, size='Weird')
    WeirdSphere = Color(96, 169, 251, priority=9, size='Weird')

    @classmethod
    def sorted_cubes_by_priority(cls):
        """
        Return a list of cube colors sorted by priority (highest first).
        """
        cubes = [cls.SmallGreen, cls.SmallBlue, cls.MediumGreen, cls.LargeGreen, cls.SmallGold, cls.SmallGlitch, cls.WeirdBulb, cls.WeirdSphere]
        return sorted(cubes, key=lambda color: color.priority, reverse=True)