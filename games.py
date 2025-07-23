import pygame
import sys
import mediapipe as mp
import numpy as np
import random
import cv2
import tkinter as tk
from tkinter import colorchooser
import os
import datetime
from predict import predict, load_feature_db

# Initialize Pygame and Tkinter
pygame.init()
tk_root = tk.Tk()
tk_root.withdraw()

# Window setup
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('AI Air Drawing')

# Colors
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
RED = (200, 0, 0)
YELLOW = (200, 200, 0)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
DARK_BLUE = (44, 62, 80)  # Modern dark blue for right panel background
BUTTON_HOVER_COLOR = (70, 130, 180)  # Steel blue for hover highlight
SHADOW_COLOR = (20, 20, 20, 120)  # Semi-transparent black shadow

# Fonts
font = pygame.font.SysFont('Segoe UI', 22)
big_font = pygame.font.SysFont('Segoe UI', 32)
font_bold = pygame.font.SysFont('Segoe UI', 18, bold=True)

#Emojis
check_emoji = pygame.image.load("D:/AI_virtual_air/assets/check.png")
sad_emoji = pygame.image.load("D:/AI_virtual_air/assets/sad.png")
multi_color = pygame.image.load("D:/AI_virtual_air/assets/multicolor.png")
check_emoji = pygame.transform.scale(check_emoji, (32, 32))
sad_emoji = pygame.transform.scale(sad_emoji, (32, 32))
multi_color = pygame.transform.scale(multi_color, (32, 32))

# Mediapipe setup
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)

# Dataset path: Update to your dataset root directory (folders named by class)
dataset_path = "D:/AI_virtual_air/cartoon_dataset"  # Adjust accordingly

# Load dataset class folders, excluding 'unknown'
random_words = []
if os.path.isdir(dataset_path):
    for entry in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, entry)
        if os.path.isdir(full_path) and entry.lower() != "unknown":
            random_words.append(entry)
if not random_words:
    random_words = ["apple", "mango"]
random_text = random.choice(random_words)

# Saved canvas area image
saved_images_folder = "saved_drawings"
os.makedirs(saved_images_folder, exist_ok=True)

# Load feature database
features_pkl_path = "D:/AI_virtual_air/features.pkl"  # Adjust as needed
feature_db = load_feature_db(features_pkl_path)

# To display prediction results on screen
prediction_text = ""

# To display prediction emoji on screen
prediction_emoji = None

# Drawing variables
current_color = MAGENTA
mode_left_hand = "Rest"
mode_right_hand = "Rest"
eraser_radius = 20
drawn_lines = []

undo_stack = []
redo_stack = []

clock = pygame.time.Clock()

# Position smoothing variables
smooth_factor = 5
position_history_left = []
position_history_right = []
last_pos_left = None
last_pos_right = None

# Canvas area (relative position and size)
canvas_x = WIDTH // 2 + 50
canvas_y = 150
canvas_width = 500
canvas_height = 400
canvas_area = pygame.Rect(canvas_x, canvas_y, canvas_width, canvas_height)

# Button sizes
button_width = 100
button_height = 40
button_padding = 10

# Create canvas surface
canvas_surface = pygame.Surface((canvas_width, canvas_height))
canvas_surface.fill(WHITE)

def draw_rounded_shadow(rect, radius=8, offset=(5,5)):
    # Draw subtle shadow behind a rect for depth
    shadow_rect = rect.move(offset)
    shadow_surf = pygame.Surface((shadow_rect.width + radius*2, shadow_rect.height + radius*2), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, SHADOW_COLOR, (radius, radius, shadow_rect.width, shadow_rect.height), border_radius=radius)
    screen.blit(shadow_surf, (shadow_rect.x - radius, shadow_rect.y - radius))

def redraw_canvas():
    canvas_surface.fill(WHITE)
    for line in drawn_lines:
        start_pos, end_pos, color = line
        pygame.draw.line(canvas_surface, color, start_pos, end_pos, 5)

def flood_fill(surface, x, y, fill_color):
    surface.lock()
    target_color = surface.get_at((x, y))
    fill_color = pygame.Color(*fill_color)
    if target_color == fill_color:
        surface.unlock()
        return
    width, height = surface.get_size()
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if cx < 0 or cx >= width or cy < 0 or cy >= height:
            continue
        current_color = surface.get_at((cx, cy))
        if current_color == target_color:
            surface.set_at((cx, cy), fill_color)
            stack.append((cx + 1, cy))
            stack.append((cx - 1, cy))
            stack.append((cx, cy + 1))
            stack.append((cx, cy - 1))
    surface.unlock()

check_button_width = 120
check_button_height = 50
check_button_x = canvas_x + (canvas_width - check_button_width) // 2
check_button_y = canvas_y + canvas_height + 20
check_button = pygame.Rect(check_button_x, check_button_y, check_button_width, check_button_height)

clear_canvas_button = pygame.Rect(WIDTH - 60, 70, 50, 50)
undo_button = pygame.Rect(WIDTH - 60, 130, 50, 50)
redo_button = pygame.Rect(WIDTH - 60, 190, 50, 50)

color_list = [
    (0, 0, 0), (255, 255, 255), (128, 128, 128),
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128)
]
color_buttons = []
button_size = 40
padding = 10
for i, color in enumerate(color_list):
    rect = pygame.Rect(WIDTH // 2 + 10 + i * (button_size + padding), 10, button_size, button_size)
    color_buttons.append({"rect": rect, "color": color})

palette_button = pygame.Rect(WIDTH - 60, 10, 50, 50)
reset_button_rect = pygame.Rect(WIDTH - 60, 250, 50, 50)
reset_button_hand_hovered = False

def save_canvas(canvas_surface):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"drawing_{timestamp}.png"
    filepath = os.path.join(saved_images_folder, filename)
    pygame.image.save(canvas_surface, filepath)
    return filepath

def check_drawing_and_predict(canvas_surface):
    global prediction_text, prediction_emoji
    img_path = save_canvas(canvas_surface)
    
    if random_text == "apple":
        classes_to_check = ["apple", "unknown"]
    elif random_text == "mango":
        classes_to_check = ["mango", "unknown"]
    else:
        classes_to_check = ["unknown"]
    
    predicted_class, confidence = predict(img_path, feature_db, classes_to_check)
    confidence_pct = confidence * 100
    
    if predicted_class.lower() == "unknown":
        prediction_text = "Not a recognized drawing. Try again!"
        prediction_emoji = sad_emoji
    else:
        prediction_text = f"Prediction: {predicted_class} ({confidence_pct:.1f}%)"
        prediction_emoji = check_emoji

def set_random_word(word):
    global random_text
    random_text = word

def draw_ui():
    # Draw right half background with dark blue
    pygame.draw.rect(screen, DARK_BLUE, (WIDTH // 2, 0, WIDTH // 2, HEIGHT))

    # Draw palette button with shadow
    draw_rounded_shadow(palette_button, radius=10, offset=(3,3))
    pygame.draw.rect(screen, WHITE, palette_button, border_radius=10)
    screen.blit(multi_color, (palette_button.x + 9, palette_button.y + 9))

    # Draw clear canvas button with shadow and hover
    draw_rounded_shadow(clear_canvas_button, radius=10, offset=(3,3))
    color_clear = BUTTON_HOVER_COLOR if clear_canvas_button.collidepoint(pygame.mouse.get_pos()) else BLACK
    pygame.draw.rect(screen, color_clear, clear_canvas_button, border_radius=10)
    cc_text = big_font.render("C", True, WHITE)
    screen.blit(cc_text, (clear_canvas_button.x + 15, clear_canvas_button.y + 10))

    # Draw undo button with shadow and hover
    draw_rounded_shadow(undo_button, radius=10, offset=(3,3))
    color_undo = BUTTON_HOVER_COLOR if undo_button.collidepoint(pygame.mouse.get_pos()) else BLACK
    pygame.draw.rect(screen, color_undo, undo_button, border_radius=10)
    undo_text = big_font.render("<", True, WHITE)
    screen.blit(undo_text, (undo_button.x + 17, undo_button.y + 10))

    # Draw redo button with shadow and hover
    draw_rounded_shadow(redo_button, radius=10, offset=(3,3))
    color_redo = BUTTON_HOVER_COLOR if redo_button.collidepoint(pygame.mouse.get_pos()) else BLACK
    pygame.draw.rect(screen, color_redo, redo_button, border_radius=10)
    redo_text = big_font.render(">", True, WHITE)
    screen.blit(redo_text, (redo_button.x + 17, redo_button.y + 10))

    # Draw Reset button "R" with shadow and hover
    draw_rounded_shadow(reset_button_rect, radius=10, offset=(3,3))
    color_reset = BUTTON_HOVER_COLOR if reset_button_rect.collidepoint(pygame.mouse.get_pos()) else (0, 150, 200)
    pygame.draw.rect(screen, color_reset, reset_button_rect, border_radius=10)
    reset_text = big_font.render("R", True, WHITE)
    reset_text_rect = reset_text.get_rect(center=reset_button_rect.center)
    screen.blit(reset_text, reset_text_rect)

    # Draw Check button with shadow and hover
    draw_rounded_shadow(check_button, radius=10, offset=(3,3))
    color_check = BUTTON_HOVER_COLOR if check_button.collidepoint(pygame.mouse.get_pos()) else GREEN
    pygame.draw.rect(screen, color_check, check_button, border_radius=10)
    check_text_big = big_font.render("Check", True, WHITE)
    check_text_rect = check_text_big.get_rect(center=check_button.center)
    screen.blit(check_text_big, check_text_rect)

    # Draw color buttons with border and shadow on hover
    for btn in color_buttons:
        btn_rect = btn["rect"]
        is_hover = btn_rect.collidepoint(pygame.mouse.get_pos())
        shadow_offset = (2,2)
        if is_hover:
            shadow_surf = pygame.Surface((btn_rect.width + 8, btn_rect.height + 8), pygame.SRCALPHA)
            pygame.draw.rect(shadow_surf, (100, 100, 100, 90), (4, 4, btn_rect.width, btn_rect.height), border_radius=6)
            screen.blit(shadow_surf, (btn_rect.x - 4 + shadow_offset[0], btn_rect.y -4 + shadow_offset[1]))
        pygame.draw.rect(screen, btn["color"], btn_rect, border_radius=6)
        if is_hover:
            pygame.draw.rect(screen, WHITE, btn_rect, width=3, border_radius=6)

    # Draw prediction message box bottom center of right half
    if prediction_text:
        text_surface = font.render(prediction_text, True, BLACK)
        padding = 15
        box_width = text_surface.get_width() + padding * 2
        box_height = text_surface.get_height() + padding * 2
        box_x = WIDTH // 2 + (WIDTH // 2 - box_width) // 2
        box_y = HEIGHT - box_height - 30  # 30px above bottom
        # Rounded rectangle background
        shadow_offset = 4
        shadow_surf = pygame.Surface((box_width + shadow_offset*2, box_height + shadow_offset*2), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (0,0,0,80), (shadow_offset, shadow_offset, box_width, box_height), border_radius=12)
        screen.blit(shadow_surf, (box_x - shadow_offset, box_y - shadow_offset))
        pygame.draw.rect(screen, ORANGE, (box_x, box_y, box_width, box_height), border_radius=12)
        screen.blit(text_surface, (box_x + padding, box_y + padding))

    if prediction_emoji:
        screen.blit(prediction_emoji, (box_x - 45, box_y + padding))  # Adjust position if needed

    # Show active mode and challenge with cleaner styling
    mode_box_rect = pygame.Rect(WIDTH // 2 + 10, 80, 350, 40)
    challenge_box_rect = pygame.Rect(WIDTH // 2 + 370, 80, 200, 40)
    pygame.draw.rect(screen, WHITE, mode_box_rect, border_radius=8)
    pygame.draw.rect(screen, WHITE, challenge_box_rect, border_radius=8)

    active_mode = "Rest"
    if mode_right_hand != "Rest":
        active_mode = mode_right_hand
    elif mode_left_hand != "Rest":
        active_mode = mode_left_hand

    mode_text = font_bold.render(f"Mode: {active_mode}", True, BLACK)
    screen.blit(mode_text, (mode_box_rect.x + 15, mode_box_rect.y + 8))

    challenge_text = font_bold.render(f"Draw: {random_text}", True, BLACK)
    screen.blit(challenge_text, (challenge_box_rect.x + 15, challenge_box_rect.y + 8))

    # Draw canvas background with subtle drop shadow
    canvas_shadow_rect = canvas_area.move(6, 6)
    shadow_surf = pygame.Surface((canvas_shadow_rect.width, canvas_shadow_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, 50), shadow_surf.get_rect(), border_radius=12)
    screen.blit(shadow_surf, canvas_shadow_rect)
    pygame.draw.rect(screen, WHITE, canvas_area, border_radius=12)
    screen.blit(canvas_surface, (canvas_x, canvas_y))


def handle_mode_right_hand(smoothed_pos, fingers_up_right):
    global current_color, last_pos_right, drawn_lines, redo_stack, reset_button_hand_hovered  # Declare all globals here
    rel_x = smoothed_pos[0] - canvas_x
    rel_y = smoothed_pos[1] - canvas_y

    # Determine mode by finger states
    if fingers_up_right == [1, 1, 0, 0]:
        mode = "Choose"
        last_pos_right = None
    elif fingers_up_right == [1, 0, 0, 0]:
        mode = "Drawing"
    elif fingers_up_right == [1, 1, 1, 1]:
        mode = "Eraser"
        last_pos_right = None
    else:
        mode = "Rest"
        last_pos_right = None

    if mode == "Choose":
        # Check if hovering over color buttons
        for btn in color_buttons:
            if btn["rect"].collidepoint(smoothed_pos):
                current_color = btn["color"]
        if clear_canvas_button.collidepoint(smoothed_pos):
            canvas_surface.fill(WHITE)
            drawn_lines.clear()
            undo_stack.clear()
            redo_stack.clear()
            last_pos_right = None
        elif undo_button.collidepoint(smoothed_pos):
            if drawn_lines:
                last_action = drawn_lines.pop()
                undo_stack.append(last_action)
                redraw_canvas()
            last_pos_right = None
        elif redo_button.collidepoint(smoothed_pos):
            if undo_stack:
                redo_action = undo_stack.pop()
                drawn_lines.append(redo_action)
                redraw_canvas()
            last_pos_right = None
        elif check_button.collidepoint(smoothed_pos):
            # Prediction handled in main loop mouse click
            pass
        
        # Handle reset button hand hover to change random word once
        if reset_button_rect.collidepoint(smoothed_pos):
            if not reset_button_hand_hovered:
                new_word = random.choice([w for w in random_words if w != random_text])
                set_random_word(new_word)
                reset_button_hand_hovered = True
        else:
            reset_button_hand_hovered = False

    elif mode == "Drawing":
        if canvas_area.collidepoint(smoothed_pos):
            if last_pos_right is not None:
                last_rel_x = last_pos_right[0] - canvas_x
                last_rel_y = last_pos_right[1] - canvas_y
                pygame.draw.line(canvas_surface, current_color,
                                 (last_rel_x, last_rel_y), (rel_x, rel_y), 5)
                drawn_lines.append(((last_rel_x, last_rel_y), (rel_x, rel_y), current_color))
                redo_stack.clear()
            last_pos_right = smoothed_pos
        else:
            last_pos_right = None
        reset_button_hand_hovered = False
    elif mode == "Eraser":
        if canvas_area.collidepoint(smoothed_pos):
            canvas_surface.lock()
            for dx in range(-eraser_radius, eraser_radius + 1):
                for dy in range(-eraser_radius, eraser_radius + 1):
                    px = rel_x + dx
                    py = rel_y + dy
                    if 0 <= px < canvas_width and 0 <= py < canvas_height:
                        if dx * dx + dy * dy <= eraser_radius * eraser_radius:
                            canvas_surface.set_at((px, py), WHITE)
            canvas_surface.unlock()
        last_pos_right = None
        reset_button_hand_hovered = False
    else:
        reset_button_hand_hovered = False

    return mode

def handle_mode_left_hand(smoothed_pos_left, fingers_up_left):
    global current_color, last_pos_left, drawn_lines, undo_stack  # Declare all globals here

    left_rel_x = smoothed_pos_left[0] - canvas_x
    left_rel_y = smoothed_pos_left[1] - canvas_y

    if fingers_up_left == [1, 1, 0, 0]:
        mode = "Choose Color"
        last_pos_left = None
    elif fingers_up_left == [1, 0, 0, 0]:
        mode = "Fill Color"
    else:
        mode = "Rest"
        last_pos_left = None

    if mode == "Choose Color":
        for btn in color_buttons:
            if btn["rect"].collidepoint(smoothed_pos_left):
                current_color = btn["color"]
        if clear_canvas_button.collidepoint(smoothed_pos_left):
            canvas_surface.fill(WHITE)
            drawn_lines.clear()
            undo_stack.clear()
            redo_stack.clear()
            last_pos_left = None
    elif mode == "Fill Color":
        if canvas_area.collidepoint(smoothed_pos_left):
            if 0 <= left_rel_x < canvas_width and 0 <= left_rel_y < canvas_height:
                flood_fill(canvas_surface, left_rel_x, left_rel_y, current_color)

    return mode

running = True
while running:
    screen.fill(BLACK)

    # Capture webcam frame
    ret, frame = cap.read()
    if not ret:
        print("Webcam Error")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    left_hand_index_pos = None
    right_hand_index_pos = None
    fingers_up_left = []
    fingers_up_right = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            detected_label = handedness.classification[0].label
            landmarks = hand_landmarks.landmark

            fingers_up = []
            # Thumb is ignored here for simplicity, only index, middle, ring, pinky checked
            for i in [8, 12, 16, 20]:
                fingers_up.append(1 if landmarks[i].y < landmarks[i - 2].y else 0)

            index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_raw = int(index_finger_tip.x * (WIDTH//2))
            y_raw = int(index_finger_tip.y * HEIGHT)
            flipped_x = WIDTH - x_raw

            # Clamp flipped_x inside canvas area to avoid jumps
            if flipped_x < WIDTH // 2:
                flipped_x = WIDTH // 2
            finger_pos = (flipped_x, y_raw)

            # SWITCHED HANDS HERE: Swap the label usage
            if detected_label == "Left":
                # Left hand acts as Right hand
                fingers_up_right = fingers_up
                right_hand_index_pos = finger_pos
                mp_drawing.draw_landmarks(
                    rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
            else:
                # Right hand acts as Left hand
                fingers_up_left = fingers_up
                left_hand_index_pos = finger_pos
                mp_drawing.draw_landmarks(
                    rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )

    # Display webcam feed on left half
    rgb_frame = cv2.resize(rgb_frame, (WIDTH // 2, HEIGHT))
    frame_surface = pygame.surfarray.make_surface(np.rot90(rgb_frame))
    screen.blit(frame_surface, (0, 0))

    # UI background on right half
    pygame.draw.rect(screen, YELLOW, (WIDTH // 2, 0, WIDTH // 2, HEIGHT))

    # Draw UI elements
    draw_ui()

    # Right hand interactions (which is originally left hand swapped)
    if right_hand_index_pos:
        position_history_right.append(right_hand_index_pos)
        if len(position_history_right) > smooth_factor:
            position_history_right.pop(0)

        avg_x = sum(p[0] for p in position_history_right) // len(position_history_right)
        avg_y = sum(p[1] for p in position_history_right) // len(position_history_right)
        smoothed_pos = (avg_x, avg_y)

        mode_right_hand = handle_mode_right_hand(smoothed_pos, fingers_up_right)

        # Visualize pointer
        if smoothed_pos[0] >= WIDTH // 2:
            pygame.draw.circle(screen, current_color, smoothed_pos, 10)
    else:
        mode_right_hand = "Rest"
        last_pos_right = None
        reset_button_hand_hovered = False

    # Left hand interactions (which is originally right hand swapped)
    if left_hand_index_pos:
        position_history_left.append(left_hand_index_pos)
        if len(position_history_left) > smooth_factor:
            position_history_left.pop(0)

        avg_x = sum(p[0] for p in position_history_left) // len(position_history_left)
        avg_y = sum(p[1] for p in position_history_left) // len(position_history_left)
        smoothed_pos_left = (avg_x, avg_y)

        mode_left_hand = handle_mode_left_hand(smoothed_pos_left, fingers_up_left)

        # Visualize pointer
        if smoothed_pos_left[0] >= WIDTH // 2 and mode_left_hand == "Fill Color":
            pygame.draw.circle(screen, current_color, smoothed_pos_left, 20)
        elif smoothed_pos_left[0] >= WIDTH // 2:
            pygame.draw.circle(screen, current_color, smoothed_pos_left, 10)
    else:
        mode_left_hand = "Rest"
        last_pos_left = None

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            # Palette button opens color chooser
            if palette_button.collidepoint(mouse_pos):
                color_code = colorchooser.askcolor(title="Choose a color")[0]
                if color_code:
                    current_color = tuple(map(int, color_code))
            # Check if check button clicked
            if check_button.collidepoint(mouse_pos):
                check_drawing_and_predict(canvas_surface)
            # Reset button clicked resets canvas and random word (mouse click)
            if reset_button_rect.collidepoint(mouse_pos):
                new_word = random.choice(random_words)
                set_random_word(new_word)
                canvas_surface.fill(WHITE)
                drawn_lines.clear()
                undo_stack.clear()
                redo_stack.clear()
                prediction_text = ""

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                # Clear canvas
                canvas_surface.fill(WHITE)
                drawn_lines.clear()
                undo_stack.clear()
                redo_stack.clear()
                prediction_text = ""
            elif event.key == pygame.K_r:
                # Reset challenge word and clear canvas
                new_word = random.choice(random_words)
                set_random_word(new_word)
                canvas_surface.fill(WHITE)
                drawn_lines.clear()
                undo_stack.clear()
                redo_stack.clear()
                prediction_text = ""
            elif event.key == pygame.K_u:
                # Undo last drawn line
                if drawn_lines:
                    last_action = drawn_lines.pop()
                    undo_stack.append(last_action)
                    redraw_canvas()
            elif event.key == pygame.K_o:
                # Redo last undone action
                if undo_stack:
                    redo_action = undo_stack.pop()
                    drawn_lines.append(redo_action)
                    redraw_canvas()

    pygame.display.update()
    clock.tick(30)

# Clean up
cap.release()
pygame.quit()
sys.exit()

