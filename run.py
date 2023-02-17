import pygame
from settings import *
from button import Button
#from analyze import predict

pygame.init()
pygame.font.init()

WINDOW = pygame.display.set_mode((HEIGHT, WIDTH))
pygame.display.set_caption("AnalyzerX2000")

def init_grid(rows, cols, color):
    grid = []
    for i in range(rows):
        grid.append([])
        for _ in range(cols):
            grid[i].append(color)
    return grid

def draw_grid(window, grid):
    for i, row in enumerate(grid):
        for j, pixel in enumerate(row):
            pygame.draw.rect(window, pixel, (j*PIXEL_SIZE, i*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

def draw(window, grid, buttons):
    window.fill(WHITE)
    draw_grid(window, grid)
    
    for button in buttons:
        button.draw(window)
    
    pygame.display.update()

def get_row_col_from_pos(pos):
    x, y = pos
    row = y // PIXEL_SIZE
    col = x // PIXEL_SIZE
    
    if row >= ROWS:
        raise IndexError
    
    return row, col

run = True
clock = pygame.time.Clock()
grid = init_grid(COLS, ROWS, WHITE)

buttons = [
    Button(10, 10, 50, 50, WHITE, "Clear", BLACK),
    Button(70, 10, 50, 50, WHITE, "Analyze", BLACK),
    Button(130, 10, 150, 50, WHITE, "Result: ", BLACK)
]

while run:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False 
            
        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            for button in buttons:
                if not button.clicked(pos):
                    continue
                if button.text == "Clear":
                    grid = init_grid(ROWS, COLS, WHITE)
                # if button.text == "Analyze":
                #     prediction, cert = predict(grid)
                #     buttons[2].text = f"Result: {prediction} ({cert*100:.0f}%)"
                break
            else:
                try:
                    row, col = get_row_col_from_pos(pos)
                    for i in range(-THICKNESS, THICKNESS):
                        for j in range(-THICKNESS, THICKNESS):
                            if 0 <= abs(i) + abs(j) < THICKNESS:
                                grid[row+i][col+j] = BLACK
                except IndexError:
                    pass
                
    draw(WINDOW, grid, buttons)
            
pygame.quit()
