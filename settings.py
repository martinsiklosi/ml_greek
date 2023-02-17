import pygame

FPS = 240
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ROWS = 49
COLS = 37
WIDTH = 490
HEIGHT = 370
PIXEL_SIZE = WIDTH // COLS
THICKNESS = 2

MAX_TRAINING_FILES = 1000

def get_font(size):
    return pygame.font.SysFont("helvetica", size)