import pyautogui
import cv2
import numpy as np
from PIL import Image
import time
from pynput import keyboard
import random
import math
import os
import pydirectinput
import threading
import sys
import pygetwindow as gw


from Color import Color, Cubes



# Global variable to keep the main loop running
global KeepAlive
KeepAlive = True

# Monitor x and y resolution
monitorX = 2560
monitorY = 1440

# Toggle debug mode on or off
DEBUG_MODE = True

# Probability of selecting a random cube (1 in RANDOM_CHANCE)
RANDOM_CHANCE = 5

BOXES_DIR = "./blocks/"
def on_press(key):
    """
    Callback function that is called when a key is pressed.
    
    Parameters:
    key: The key that was pressed.
    """
    try:
        # Check if the pressed key is F1
        if key == keyboard.Key.f1:
            if DEBUG_MODE:
                print("F1 pressed. Exiting the program.")
            os._exit(0)  # Use os._exit to terminate the program immediately
    except AttributeError:
        # Handle the case where the key is not recognized
        pass

def save_mask(mask, filename, directory="./masks"):
    """
    Save a mask image to a directory.
    
    Parameters:
    mask (np.ndarray): The binary mask image.
    filename (str): The filename for the saved mask image.
    directory (str): The directory to save the mask image.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)
    cv2.imwrite(file_path, mask)

    if DEBUG_MODE:
        print(f"Saved mask to {file_path}")


def getKeepAlive():
    return KeepAlive
def setKeepAlive(value):
    global KeepAlive
    KeepAlive = value

def press_keys(keys):
    """
    Presses a sequence of keys given in an array.

    Parameters:
    keys (list of str): A list containing the keys to be pressed in sequence.
    """
    # Wait for a brief moment to switch to the target window if needed
    

    # Iterate over each key in the array and press it
    for key in keys:
        time.sleep(0.1)  # Wait for a brief moment before pressing the next key
        pydirectinput.press(key)

#list of keys to press, and each key has a duration


def pressKeysPrecise(keys):
    """
    Presses a sequence of keys given in an array with a specific duration.
    Parameters:
    keys (list of tuple): A list containing the keys to be pressed in sequence along with their durations.
    """
    for key, duration in keys:
        #key up key down
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)

def locate_on_screen(image_path, confidence=0.8):
    """Locate an image on the screen and return its coordinates."""
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    template = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if template is None:
        raise FileNotFoundError(f"Template image not found at {image_path}")

    if template.shape[2] == 4:  # If template has an alpha channel
        b, g, r, a = cv2.split(template)
        template_bgr = cv2.merge((b, g, r))
        mask = cv2.merge((a, a, a))  # Create a 3-channel mask

        if DEBUG_MODE:
            cv2.imwrite("screenshot.png", screenshot)
            cv2.imwrite("template_bgr.png", template_bgr)
            cv2.imwrite("mask.png", mask)

        result = cv2.matchTemplate(screenshot, template_bgr, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        template_bgr = template

        if DEBUG_MODE:
            cv2.imwrite("screenshot.png", screenshot)
            cv2.imwrite("template.png", template_bgr)

        result = cv2.matchTemplate(screenshot, template_bgr, cv2.TM_CCOEFF_NORMED)

    locations = np.where(result >= confidence)
    locations = list(zip(*locations[::-1]))  # Reverse x, y and zip

    if locations:
        return locations[0]  # Return the first location
    else:
        return None



def click_at_location(location, click="left"):
    """Move the mouse to a location and click."""
    if location:
        x, y = location
        pydirectinput.moveTo(x, y, duration=0.1)
        pydirectinput.click(button=click)
    else:
        if DEBUG_MODE:
            print("Location not found!")

def wait_for_key(key):
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    
def on_press(key):
    try:
        if key == keyboard.Key.delete:
            print("F7 key pressed. Performing action...")
            return False  # Stop listener after detecting F7
    except Exception as e:
        print(f"Error: {e}")


def getOverDumbassSign():
    pydirectinput.keyDown('w')
    pydirectinput.keyDown('space')
    time.sleep(0.5)
    pydirectinput.keyUp('w')
    pydirectinput.keyUp('space')
def wait_for_f7():
    print("Waiting for F7 key press...")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def getBirdsEyeView():
    #press o once
    pressKeysPrecise([('o', 0.1)])
    #move mouse up 100 pixels
    pydirectinput.move(0, -100)
    #drag mouse down 100 pixels
    #right
    pydirectinput.mouseDown(button="right")
    pydirectinput.drag(0, 250, duration=0.5)
    pydirectinput.mouseUp(button="right")
    #hold O, 1 second
    pydirectinput.keyDown('o')
    time.sleep(1)
    pydirectinput.keyUp('o')

def panRight(time=1):
    #pans right 90deg, using right arrow key
    pressKeysPrecise([('right', time)])

def panLeft(time=1):
    #pans left 90deg, using left arrow key
    pressKeysPrecise([('left', time)])
  
def capture_screenshot():
    """
    Capture a screenshot of the Roblox player window, crop the top and bottom 10%,
    and return it as a BGR image along with window position information.
    """
    try:
        roblox_window = gw.getWindowsWithTitle('Roblox')[0]
    except IndexError:
        if DEBUG_MODE:
            print("Roblox window not found.")
        return None, None, None

    # Check if the window is visible
    if not roblox_window.visible:
        if DEBUG_MODE:
            print("Roblox window is not visible.")
        return None, None, None

    # Get the bounding box of the window
    left, top, width, height = roblox_window.left, roblox_window.top, roblox_window.width, roblox_window.height

    # Capture the full window using pyautogui
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot_rgb = np.array(screenshot)  # Convert to NumPy array
    screenshot_bgr = cv2.cvtColor(screenshot_rgb, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

    #I want to crop 10% from the top and bottom
    crop_height = int(height * 0.2)
    cropped_screenshot = screenshot_bgr[crop_height:-crop_height, :, :]
    #save crop
    if DEBUG_MODE:
        save_image(cropped_screenshot, "cropped_screenshot.png")

    return cropped_screenshot, left, top + crop_height


def rgb_to_hsv(rgb_color):
    """
    Convert an RGB color to HSV.
    
    Parameters:
    rgb_color (tuple): A tuple of the RGB color (e.g., (255, 0, 0)).

    Returns:
    tuple: A tuple of the HSV color.
    """
    rgb_np = np.uint8([[rgb_color]])
    hsv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
    return tuple(hsv_np[0][0])

def create_color_mask(image, rgb_color, tolerance=5):
    """
    Create a color mask for a specific RGB color within a given tolerance and visualize exclusion zones.
    
    Parameters:
    image (np.ndarray): The BGR image.
    rgb_color (tuple): The target RGB color (e.g., (255, 0, 0)).
    tolerance (int): The tolerance for the color range.

    Returns:
    np.ndarray: The mask for the specified color range.
    """
    if image is None:
        if DEBUG_MODE:
            print("Error: Received None image in create_color_mask.")
        return None

    # Convert the RGB color to HSV
    hsv_color = rgb_to_hsv(rgb_color)

    # Define specific HSV ranges for each cube variant
    if rgb_color == (82, 243, 76):  # Large Green
        lower_bound = np.array([50, 200, 200])  # Adjusted HSV range for large green
        upper_bound = np.array([70, 255, 255])
    elif rgb_color == (109, 243, 162):  # Small Green
        lower_bound = np.array([70, 150, 150])  # Adjusted HSV range for small green
        upper_bound = np.array([90, 255, 255])
    elif rgb_color == (157, 253, 208):  # Wind Green
        lower_bound = np.array([80, 100, 200])  # Adjusted HSV range for wind green
        upper_bound = np.array([100, 255, 255])
    elif rgb_color == (96, 169, 251):  # Weird Blue
        # Slightly stricter HSV range for weird blue
        lower_bound = np.array([hsv_color[0] - 3, 120, 180])
        upper_bound = np.array([hsv_color[0] + 3, 255, 255])
    else:
        lower_bound = np.array([max(0, hsv_color[0] - tolerance), 100, 100])
        upper_bound = np.array([min(180, hsv_color[0] + tolerance), 255, 255])

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a binary mask based on the defined HSV range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Visualize the exclusion zones on the mask
    image_height, image_width = image.shape[:2]
    player_x = image_width // 2

    # Top Exclusion Zone: 30% of the image height, centered around the player
    top_exclusion_zone_top = 0
    top_exclusion_zone_bottom = int(image_height * 0.3)
    top_exclusion_zone_left = player_x - 250
    top_exclusion_zone_right = player_x

    # Draw the top exclusion zone
    cv2.rectangle(
        mask,
        (top_exclusion_zone_left, top_exclusion_zone_top),
        (top_exclusion_zone_right, top_exclusion_zone_bottom),
        (255, 255, 255),
        thickness=2,
    )

    # Bottom Exclusion Zone: Positioned 200px off the middle to the left
    bottom_exclusion_zone_height = 100  # Define a fixed height for the exclusion zone
    bottom_exclusion_zone_top = image_height - bottom_exclusion_zone_height
    bottom_exclusion_zone_bottom = image_height
    bottom_exclusion_zone_left = player_x - 200
    bottom_exclusion_zone_right = player_x + 50  # Adjust width to include the necessary area

    # Draw the bottom exclusion zone
    cv2.rectangle(
        mask,
        (bottom_exclusion_zone_left, bottom_exclusion_zone_top),
        (bottom_exclusion_zone_right, bottom_exclusion_zone_bottom),
        (255, 255, 255),
        thickness=2,
    )

    # Save the mask for debugging purposes
    if DEBUG_MODE:
        mask_filename = f"mask_{rgb_color[0]}_{rgb_color[1]}_{rgb_color[2]}.png"
        save_mask(mask, mask_filename)

    return mask



def locate_box_with_opencv(screenshot, template_path, scales=[1.0, 0.9, 0.8], threshold=0.8):
    """
    Locate a box in a screenshot using OpenCV template matching with multiple scales.

    Parameters:
    screenshot (np.ndarray): The BGR screenshot.
    template_path (str): The path to the template image.
    scales (list): List of scales to resize the template for matching.
    threshold (float): The matching threshold for detection.

    Returns:
    tuple: (x, y) coordinates of the detected box center or None if not found.
    """
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        if DEBUG_MODE:
            print(f"Template image not found at {template_path}")
        return None

    for scale in scales:
        # Resize the template
        width = int(template.shape[1] * scale)
        height = int(template.shape[0] * scale)
        resized_template = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)

        # Match template
        result = cv2.matchTemplate(screenshot, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Check if match is above threshold
        if max_val >= threshold:
            center_x = max_loc[0] + width // 2
            center_y = max_loc[1] + height // 2
            if DEBUG_MODE:
                print(f"Box found with scale {scale} at ({center_x}, {center_y})")
            return center_x, center_y

    return None

def preprocess_images(directory, target_width, target_height):
    """
    Preprocess images in a directory to a target resolution.

    Parameters:
    directory (str): The path to the directory containing images.
    target_width (int): The target width for resizing.
    target_height (int): The target height for resizing.
    """
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path)
        if image is not None:
            resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(image_path, resized_image)
            if DEBUG_MODE:
                print(f"Resized {filename} to {target_width}x{target_height}")

def hopboxbyimage():
    """
    Continuously search for boxes in the game window using images from the /boxes/ directory.
    Capture a screenshot every 15 seconds, and move to the closest box when found.
    """
    # Preprocess box images to ensure they are 50x50 pixels
    preprocess_images(BOXES_DIR, target_size=(50, 50))

    # Start the timer for 5 minutes
    start_time = time.time()
    reset_interval = 300  # 5 minutes in seconds
    screenshot_interval = 15  # 15 seconds in seconds

    while KeepAlive:
        # Check if it's time to reset the search
        if time.time() - start_time >= reset_interval:
            if DEBUG_MODE:
                print("Resetting search...")
            start_time = time.time()  # Reset the timer

        # Capture a screenshot of the game window
        screenshot, window_left, window_top = capture_screenshot()

        # Skip processing if screenshot failed
        if screenshot is None:
            print("Failed to capture Roblox window.")
            time.sleep(screenshot_interval)  # Wait before trying again
            continue

        if DEBUG_MODE:
            print("Screenshot captured. Searching for boxes...")

        # Initialize variables to track the closest box
        closest_box = None
        min_distance = float('inf')

        # Search for each box image in the directory
        for box_image in os.listdir(BOXES_DIR):
            box_path = os.path.join(BOXES_DIR, box_image)

            # Locate the box image on the screen using PyAutoGUI
            location = pyautogui.locateOnScreen(box_path, confidence=0.9)

            if location:
                # Calculate the center of the box
                center_x = location.left + location.width // 2
                center_y = location.top + location.height // 2

                # Calculate the distance from the screen center
                distance = np.sqrt((center_x - monitorX // 2) ** 2 + (center_y - monitorY // 2) ** 2)

                # Update the closest box if this one is nearer
                if distance < min_distance:
                    min_distance = distance
                    closest_box = (center_x, center_y)

                # Debug output for each detected box
                if DEBUG_MODE:
                    print(f"Detected box {box_image} at ({center_x}, {center_y}), Distance: {distance:.2f}")

        # Move to the closest box and right-click if one was found
        if closest_box:
            pyautogui.moveTo(closest_box[0], closest_box[1], duration=0.1)
            pyautogui.click(button='right')
            if DEBUG_MODE:
                print(f"Right-clicked closest box at ({closest_box[0]}, {closest_box[1]})")
        else:
            # If no boxes are found, go to the center
            resetToMain()

        # Wait for the screenshot interval before capturing again
        time.sleep(screenshot_interval)


def hopboxbyimage_opencv():
    """
    Continuously search for boxes in the game window using OpenCV template matching.
    Capture a screenshot every 15 seconds, and move to the closest box when found.
    """
    start_time = time.time()
    reset_interval = 300  # 5 minutes in seconds
    screenshot_interval = 15  # 15 seconds in seconds

    while KeepAlive:
        if time.time() - start_time >= reset_interval:
            if DEBUG_MODE:
                print("Resetting search...")
            start_time = time.time()

        screenshot, window_left, window_top = capture_screenshot()

        if screenshot is None:
            print("Failed to capture Roblox window.")
            time.sleep(screenshot_interval)
            continue

        if DEBUG_MODE:
            print("Screenshot captured. Searching for boxes...")

        closest_box = None
        min_distance = float('inf')

        for box_image in os.listdir(BOXES_DIR):
            box_path = os.path.join(BOXES_DIR, box_image)
            location = locate_box_with_opencv(screenshot, box_path)

            if location:
                center_x, center_y = location
                distance = np.sqrt((center_x - monitorX // 2) ** 2 + (center_y - monitorY // 2) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_box = (center_x, center_y)

                if DEBUG_MODE:
                    print(f"Detected box {box_image} at ({center_x}, {center_y}), Distance: {distance:.2f}")

        if closest_box:
            pyautogui.moveTo(closest_box[0] + window_left, closest_box[1] + window_top, duration=0.1)
            
            pyautogui.click(button='right')
            if DEBUG_MODE:
                print(f"Right-clicked closest box at ({closest_box[0]}, {closest_box[1]})")
        else:
            goToCenter()

        time.sleep(screenshot_interval)



def detect_blobs(mask):
    """
    Detect blobs in a binary mask using contours.
    
    Parameters:
    mask (np.ndarray): The binary mask.

    Returns:
    list: List of contours found in the mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG_MODE:
        if not contours:
            print("No contours found.")
        else:
            print(f"Found {len(contours)} contours.")
    return contours



def draw_and_click_contour(image, contours, window_left, window_top, priority):
    """
    Draw contours on an image and perform a right-click while holding shift
    on the highest priority detected contour, then wait before proceeding.

    Parameters:
    image (np.ndarray): The BGR image.
    contours (list): The list of contours.
    window_left (int): The x-coordinate of the window's left boundary.
    window_top (int): The y-coordinate of the window's top boundary.
    priority (int): The priority of the detected contour.

    Returns:
    np.ndarray: The image with contours drawn, or None if a contour is in the exclusion zone.
    bool: True if a valid contour is processed, False if a contour is in the exclusion zone.
    """
    # Get the dimensions of the image to calculate the player's position
    image_height, image_width = image.shape[:2]

    # Calculate the player's position as the center of the screen
    player_x = image_width // 2
    player_y = image_height // 2

    # Define the exclusion zone to the left of the player
    exclusion_zone_left = player_x - 250
    exclusion_zone_top = player_y - 125  # Center the exclusion zone vertically around the player
    exclusion_zone_bottom = player_y + 125

    # Check for any contours in the exclusion zone
    for contour in contours:
        # Calculate the bounding box and center of the contour
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # If a contour's center is within the exclusion zone, skip processing
        if exclusion_zone_left <= center_x <= player_x and exclusion_zone_top <= center_y <= exclusion_zone_bottom:
            if DEBUG_MODE:
                print("Object detected in exclusion zone. Skipping this tier.")
            return None, False

    # Find the largest contour that is not within the exclusion zone
    largest_contour = None
    largest_area = 0

    for contour in contours:
        # Calculate the bounding box and center of the contour
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # Check if the contour's center is outside the exclusion zone
        if center_x < exclusion_zone_left or center_y < exclusion_zone_top or center_y > exclusion_zone_bottom:
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_contour = contour
                largest_area = area

    # If no valid contour is found, return the image as is
    if largest_contour is None:
        if DEBUG_MODE:
            print("No valid contour found outside the exclusion zone.")
        return image, False

    # Draw and interact with the largest valid contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    center_x = x + w // 2
    center_y = y + h // 2

    # Draw a rectangle around the detected object
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Add debug information for area if debug mode is enabled
    if DEBUG_MODE:
        cv2.putText(image, f'Area: {int(largest_area)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Move the mouse to the center of the largest contour
    pydirectinput.moveTo(window_left + center_x, window_top + center_y)

    # Hold the shift key
    

    # Perform a right-click
    pydirectinput.click(button="right")
    #wait for 0.5s
    time.sleep(0.5)
    # Release the shift key
    pydirectinput.keyDown('shift')
    pydirectinput.click(button="right", duration=0.5)
    # Wait for 10 + 1.5 * priority seconds before restarting the process
    wait_time = 15 + 2 * priority
    if DEBUG_MODE:
        print(f"Waiting for {wait_time} seconds before starting the process again.")
    time.sleep(wait_time)
    #do a little wasd movement
    pressKeysPrecise([('w', 0.1), ('a', 0.1), ('s', 0.1), ('d', 0.1)])
    pydirectinput.keyUp('shift')
    return image, True




def save_images(original, mask, result, directory="./temp"):
    """
    Save the original, mask, and result images to a directory.
    
    Parameters:
    original (np.ndarray): The original BGR image.
    mask (np.ndarray): The binary mask image.
    result (np.ndarray): The result image with contours drawn.
    directory (str): The directory to save the images.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    cv2.imwrite(os.path.join(directory, "original.png"), original)
    cv2.imwrite(os.path.join(directory, "mask.png"), mask)
    cv2.imwrite(os.path.join(directory, "result.png"), result)

def process_image_based_on_priority():
    """
    Process the screenshot to detect loot boxes of different colors based on priority and save the results.
    """
    screenshot, window_left, window_top = capture_screenshot()

    # Check if screenshot is captured successfully
    if screenshot is None:
        print("Failed to capture Roblox window.")
        return

    if DEBUG_MODE:
        print("Screenshot captured successfully.")

    # Process each cube color based on priority
    for color in Cubes.sorted_cubes_by_priority():
        if DEBUG_MODE:
            print(f"Processing color with priority {color.priority}: {color}")

        mask = create_color_mask(screenshot, (color.r, color.g, color.b))
        contours = detect_blobs(mask)

        if contours:
            # Pass priority to the draw_and_click_contour function
            result_image, valid_contour = draw_and_click_contour(
                screenshot.copy(), contours, window_left, window_top, color.priority
            )

            # If a valid contour is processed, save the images and stop further processing
            if valid_contour:
                save_images(screenshot, mask, result_image)
                return





def goToMailbox():
    #filepath images/mailbox.png
    mailbox_location = locate_on_screen("images/mailbox.png", confidence=0.6)
    if(not mailbox_location):
       mailbox_location = locate_on_screen("images/mailboxsnowy.png", confidence=0.8)
    if(mailbox_location):
        #increment the mailbox x by 50
        mailbox_location = (mailbox_location[0] + 50, mailbox_location[1])
        click_at_location(mailbox_location, click="right")
        return True
    return False
def getToMiddle():
    #click

    #press left arrow 0.5 s
    #spam I like 10 times
    for i in range(6):
        pressKeysPrecise([('i', 0.1)])
    time.sleep(0.5)
    pressKeysPrecise([('left', 0.75)])
    #key down W and Space, 0.5s duration
    getOverDumbassSign()
    #hold w, 2 seconds
    #hold shift
    pydirectinput.keyDown('shift')
    pressKeysPrecise([('w', 3.7)])
    #pan left 0.35
    panLeft(0.35)
    #hold w, 1.5s
    pressKeysPrecise([('w', 1.5)])
    getBirdsEyeView()
    time.sleep(0.5)
    panRight()
    #hold a
    pydirectinput.keyDown('a')
    time.sleep(0.1)
    pydirectinput.keyDown('space')
    time.sleep(0.5)
    pydirectinput.keyUp('a')
    pydirectinput.keyUp('space')
    #keydown shift
    pydirectinput.keyUp('shift')
    if(goToMailbox()):
        #KEYDOWN SHIFT
        pydirectinput.keyDown('shift')
        #wait for 10s
        time.sleep(5)
        #keyup shift
        pydirectinput.keyUp('shift')
        #panLeft 0.7s
        panLeft(0.85)
        #hold w, 1s
        pressKeysPrecise([('w', 1)])
        time.sleep(0.5)
        sys.exit(0)
    else:
        resetToMain()


def resetToMain():
    resetKeys = [('esc', 0.08), ('r', 0.08), ('enter', 0.08)]
    for i in range(5):
        pressKeysPrecise(resetKeys)
    time.sleep(0.5)
    getToMiddle()

   #directinput 
   #escape, r enter


def draw_and_click_contour(image, contours, window_left, window_top, priority):
    """
    Draw contours on an image and perform a right-click while holding shift
    on the highest priority detected contour, then wait before proceeding.

    Parameters:
    image (np.ndarray): The BGR image.
    contours (list): The list of contours.
    window_left (int): The x-coordinate of the window's left boundary.
    window_top (int): The y-coordinate of the window's top boundary.
    priority (int): The priority of the detected contour.

    Returns:
    np.ndarray: The image with contours drawn, or None if a contour is in the exclusion zone.
    bool: True if a valid contour is processed, False if a contour is in the exclusion zone.
    """
    # Get the dimensions of the image to calculate the player's position
    image_height, image_width = image.shape[:2]

    # Calculate the player's position as the center of the screen
    player_x = image_width // 2

    # Define the exclusion zones
    top_exclusion_zone_top = 0
    top_exclusion_zone_bottom = int(image_height * 0.3)
    top_exclusion_zone_left = player_x - 250
    top_exclusion_zone_right = player_x

    bottom_exclusion_zone_top = image_height - 100
    bottom_exclusion_zone_bottom = image_height
    bottom_exclusion_zone_left = player_x - 200
    bottom_exclusion_zone_right = player_x + 50

    # Check for any contours in the exclusion zones
    for contour in contours:
        # Calculate the bounding box and center of the contour
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # If a contour's center is within any exclusion zone, skip processing
        if (
            top_exclusion_zone_left <= center_x <= top_exclusion_zone_right
            and top_exclusion_zone_top <= center_y <= top_exclusion_zone_bottom
        ) or (
            bottom_exclusion_zone_left <= center_x <= bottom_exclusion_zone_right
            and bottom_exclusion_zone_top <= center_y <= bottom_exclusion_zone_bottom
        ):
            if DEBUG_MODE:
                print("Object detected in exclusion zone. Skipping this tier.")
            return None, False

    # Find the largest contour that is not within the exclusion zones
    largest_contour = None
    largest_area = 0

    for contour in contours:
        # Calculate the bounding box and center of the contour
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # Check if the contour's center is outside the exclusion zones
        if (
            (center_x < top_exclusion_zone_left or center_x > top_exclusion_zone_right)
            or (center_y < top_exclusion_zone_top or center_y > top_exclusion_zone_bottom)
        ) and (
            (center_x < bottom_exclusion_zone_left or center_x > bottom_exclusion_zone_right)
            or (center_y < bottom_exclusion_zone_top or center_y > bottom_exclusion_zone_bottom)
        ):
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_contour = contour
                largest_area = area

    # If no valid contour is found, return the image as is
    if largest_contour is None:
        if DEBUG_MODE:
            print("No valid contour found outside the exclusion zones.")
        return image, False

    # Draw and interact with the largest valid contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    center_x = x + w // 2
    center_y = y + h // 2

    # Draw a rectangle around the detected object
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Add debug information for area if debug mode is enabled
    if DEBUG_MODE:
        cv2.putText(image, f'Area: {int(largest_area)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Move the mouse to the center of the largest contour
    pydirectinput.moveTo(window_left + center_x, window_top + center_y)
    time.sleep(0.5)
    pydirectinput.click(button="right")
    time.sleep(0.5)

    # Hold the shift key
    pydirectinput.keyDown('shift')

    # Perform a right-click again
    pydirectinput.click(button="right", duration=0.5)

    # Release the shift key
    pydirectinput.keyUp('shift')

    # Wait for 8 + 2 * priority seconds before restarting the process
    wait_time = 14 + 2 * priority
    if DEBUG_MODE:
        print(f"Waiting for {wait_time} seconds before starting the process again.")
    time.sleep(wait_time)

    # Perform a small WASD movement
    pressKeysPrecise([('w', 0.1), ('a', 0.1), ('s', 0.1), ('d', 0.1)])

    return image, True

def save_image(image, filename, directory="/temp"):
    """
    Save an image to a directory.
    
    Parameters:
    image (np.ndarray): The image to save.
    filename (str): The filename for the saved image.
    directory (str): The directory to save the image.
    """
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)
    print(f"Saving image to {file_path}")  # Debug print

    # Save the image
    cv2.imwrite(file_path, image)

def lookForCubesBase():
    process_image_based_on_priority()

def mainLoop():
    wait_for_f7()
    while getKeepAlive():
        #reset to main
        resetToMain()
        #wait for key n to be pressed
        #look for cubes in base
        lookForCubesBase()

def monitor_keypress():
    """
    Monitor for the F1 key press and exit the program when detected.
    """
    if DEBUG_MODE:
        print("Press F1 to exit the program.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    # Start the keypress monitoring thread
    keypress_thread = threading.Thread(target=monitor_keypress, daemon=True)
    keypress_thread.start()
    
    # Main program logic
    try:
        while True:
            mainLoop()
    except KeyboardInterrupt:
        print("Program terminated by user.")
        sys.exit(0)




# if __name__ == "__main__":
#     # Start the hopboxbyimage function in a separate thread
#     wait_for_f7()
#     hopbox_thread = threading.Thread(target=hopboxbyimage, daemon=True)
#     hopbox_thread.start()

#     # Start the keypress monitoring thread
#     keypress_thread = threading.Thread(target=monitor_keypress, daemon=True)
#     keypress_thread.start()
#     #preproc
#     preprocess_images(BOXES_DIR, target_size=(50, 50))
#     # Main program logic
#     try:
#         while KeepAlive:
#             # Placeholder for other operations
#             time.sleep(1)  # Keep the main thread alive
#     except KeyboardInterrupt:
#         print("Program terminated by user.")
#         sys.exit(0)