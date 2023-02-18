import pygame
from pathlib import Path

FPS = 240
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ROWS = 49
COLS = 43
WIDTH = 490
HEIGHT = 430
PIXEL_SIZE = WIDTH // COLS
THICKNESS = 2

LABEL_STRINGS = [path.name for path in Path("C:\\proj\\ml_data\\greek_alphabet\\SUFF\\").iterdir()]
LABEL_CONVERSIONS = {label: i for i, label in enumerate(LABEL_STRINGS)}

def get_font(size):
    return pygame.font.SysFont("helvetica", size)