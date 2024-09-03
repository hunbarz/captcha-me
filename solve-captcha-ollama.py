#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pathlib
import shutil
import ollama
import base64

# helpers
def get_file_directory(file_path):
    return str(pathlib.Path(file_path).parent)


def list_directory_contents(directory_path):
    path = pathlib.Path(directory_path)
    return [str(item) for item in path.iterdir()]


def remove_file(file_path):
    path = pathlib.Path(file_path)
    if path.is_file():
        path.unlink()


def copy_file(source_file_path, destination_file_path):
    shutil.copy(source_file_path, destination_file_path)


def rename_file(old_name, new_name):
    file = pathlib.Path(old_name)
    if file.exists() and file.is_file():
        file.rename(new_name)


def separate_pics_from_captcha(captcha_path, output_folder):
    # See if output_folder exists, else create it
    folder = pathlib.Path(output_folder)

    if not folder.exists():
        folder.mkdir(parents=True)

    # Read the CAPTCHA image
    image = cv2.imread(captcha_path)
    if image is None:
        raise ValueError("[-] Image not found or unable to read -> separate_pics_from_captcha()")

    # Convert to grayscale, detect edges using the Canny method and find the contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Contours should be around 192x192 or 252x252, everything else is not interesting (found with testing)
    cnt = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        subimage = image[y:y+h, x:x+w]
        if (
            ((subimage.shape[0] > 190 and subimage.shape[0] < 195) and (subimage.shape[1] > 190 and subimage.shape[1] < 195)) or 
            ((subimage.shape[0] > 250 and subimage.shape[0] < 255) and (subimage.shape[1] > 250 and subimage.shape[1] < 255))
            ):
                # if suitable subimage is found, save to output folder
                cnt += 1
                output_path = f"{output_folder}/picture_{cnt:02d}.png"
                cv2.imwrite(output_path, subimage)


def highlight_solution(subimage_path, captcha_path):
    # Read the images
    subimage, image = cv2.imread(subimage_path), cv2.imread(captcha_path)
    
    # https://www.flaticon.com/free-icons/foursquare-check-in (64x64)
    b64_overlay = b"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"
    overlay = cv2.imdecode(np.frombuffer(base64.b64decode(b64_overlay), np.uint8), cv2.IMREAD_COLOR)

    # If images were not found, raise an error
    if any(v is None for v in (subimage, image, overlay)):
        raise ValueError("[-] Image not found or unable to read - highlight_solution()")

    # Find the contours, as in separate_pics_from_captcha
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Go through the CAPTCHA, identify location of subimages
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Find subimage in the CAPTCHA and overwrite with overlay image
        if (np.array_equal(image[y:y+h, x:x+w], subimage)):
 
            # Get the dimensions of the overlay image
            overlay_height, overlay_width = overlay.shape[:2]

            # Get the region that will be overwritten with the overlay
            roi = image[y:y+overlay_height, x:x+overlay_width]

            # Create a mask of the overlay and its inverse mask
            overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(overlay_gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Black-out the area of the overlay in the ROI
            background_roi = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only the region of the overlay from the overlay image
            overlay_fg = cv2.bitwise_and(overlay, overlay, mask=mask)

            # Put the overlay in the ROI and modify the main image
            dst = cv2.add(background_roi, overlay_fg)
            image[y:y+overlay_height, x:x+overlay_width] = dst

    cv2.imwrite(captcha_path, image)


def chat_ollama(question = "Where is D.B. Cooper?", images = []):
    # Chat with ollama, ask questions about the images
    res = ollama.chat(model="llava:7b", messages=[
    #res = ollama.chat(model="llava:34b", messages=[ --> aborted after 35m, still not a single one solved
        {
            'role': 'user',
            'content': question,
            'images': images
        }
    ])
    
    return res['message']['content']


def solve_captcha(captcha_path):
    solved_captcha_path = captcha_path[:-4] + ".solving.png"

    copy_file(captcha_path, solved_captcha_path)
    output_folder = get_file_directory(solved_captcha_path) + "/splitted"

    print("[>] Getting the CAPTCHA instructions...")
    with open (solved_captcha_path, "rb") as f:
        instruction = chat_ollama("Which squares or images should be selected according to the text in the blue box at the top of the image? Answer with maximum 2 words", [base64.b64encode(f.read())]).lower().strip()
        print(f"[>] Select all '{instruction}'")

    print("[>] Separating the subimages from the CAPTCHA...")
    separate_pics_from_captcha(solved_captcha_path, output_folder)
    subimages_captcha = sorted(list_directory_contents(output_folder))

    print("[>] Identifying the correct subimages...")
    results = {}
    for i, img in enumerate(subimages_captcha):
        with open (img, "rb") as f:
            b64_img = base64.b64encode(f.read())

            res = chat_ollama(f"Does the image contain {instruction}, or parts of it? Answer with a simple YES or NO.", [b64_img]).strip().lower()
            results.update({img : res})

            # debug prints
            #print(f"[?] Does the image ./{img} contain {instruction}?")
            #print(f"[!] --> {res}")

    print("[>] Highlighting the solution...")
    for r in results:
        if results[r] == "yes":
            highlight_solution(r, solved_captcha_path)

    rename_file(solved_captcha_path, solved_captcha_path.replace(".solving.png", ".solved.png"))

    print("[>] Cleaning up...")
    for f in list_directory_contents(output_folder):
        remove_file(f)
        

#solve_captcha("./captchas-ollama/image-01.png")

for captcha_path in sorted(list_directory_contents("./captchas-ollama")):
    if "solved" not in captcha_path or "solving" not in captcha_path:
        print(f"\n[+] Solving CAPTCHA -> {captcha_path}")
        solve_captcha(captcha_path)

# $ time python3 solve-captcha-ollama.py
# single image --> 47s
# 25 images -----> 13m55s