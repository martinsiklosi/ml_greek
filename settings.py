import pygame
from pathlib import Path

FPS = 240
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ROWS = 49
COLS = 43
HEIGHT = 490
WIDTH = 430
PIXEL_SIZE = WIDTH // COLS
THICKNESS = 2

BASE_FOLDERS = (
    "C:\\proj\\ml_data\\greek_alphabet\\SUFF\\", 
    "C:\\proj\\ml_data\\greek_alphabet\\NORM\\"
)

LABEL_STRINGS = [path.name for path in Path(BASE_FOLDERS[0]).iterdir()]
LABEL_CONVERSIONS = {label: i for i, label in enumerate(LABEL_STRINGS)}

def get_font(size):
    return pygame.font.SysFont("helvetica", size)