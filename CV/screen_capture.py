import mss
import pyautogui
import cv2
import numpy as np
import time
import multiprocessing
import originalSnake
import subprocess

from templates import TEMPLATES

import settings

class ScreenCaptureAgent:
    #couple declarations, blah blah blah
    def __init__(self) -> None:
        self.capture_process = None
        self.fps = None
        self.img = None
        self.img_gry = None

        self.w, self.h = pyautogui.size()
        #print(f"Screen Resolution: {self.w},{self.h}")
        self.monitor = {
            'top': settings.COMP_VIZ_TOP_LEFT[1],
            'left': settings.COMP_VIZ_TOP_LEFT[0],
            'width': settings.COMP_VIZ_BOTTOM_RIGHT[0],
            'height': settings.COMP_VIZ_BOTTOM_RIGHT[1]
        }

        self.templates = {}
        template_count = 0
        for t in TEMPLATES:
            template_count += 1
            self.templates[t] = cv2.imread(TEMPLATES[t], cv2.IMREAD_GRAYSCALE)
            print(f"{template_count}: {t}")

    

    def capture_screen(self):
        game_process = subprocess.Popen(["python", "originalSnake.py"])
        with mss.mss() as sct:
            while True:
                self.img = np.array(sct.grab(self.monitor))[:, :, :3]
                self.img_gry = cv2.cvtColor(self.img.copy(), cv2.COLOR_BGRA2GRAY)


                #self.find_templates()

                if settings.ENABLE_PREVIEW:
                    preview = cv2.resize(self.img, (0, 0), fx=0.5, fy=0.5)

                    cv2.imshow('Computer Vision', preview)
                    cv2.moveWindow("Computer Vision", 640, 0)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
        cv2.destroyAllWindows()
    
    def find_templates(self):
        #food = cv2.matchTemplate(self.img_gry, self.templates["food"], cv2.TM_CCOEFF_NORMED)
        #food_min_val, food_max_val, food_min_loc, food_max_loc = cv2.minMaxLoc(food)
        head = cv2.matchTemplate(self.img_gry, self.templates["head"], cv2.TM_CCOEFF_NORMED)
        head_min_val, head_max_val, head_min_loc, head_max_loc = cv2.minMaxLoc(head)
        
        
        '''if food_max_val >= 0.8:
            #print(f"Food Max: {food_max_val}")
            self.img = cv2.circle(self.img, (food_max_loc[0] + 10, food_max_loc[1] + 20), 30, (0, 255, 0), 3, 2)
            self.img = cv2.putText(self.img, "Food", (food_max_loc[0] + 45, food_max_loc[1] + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)'''

        if head_max_val >= 0.8:
            #print(f"Head Max: {head_max_val}")
            self.img = cv2.circle(self.img, (head_max_loc[0] + 10, head_max_loc[1] + 20), 30, (0, 255, 0), 3, 2)
            self.img = cv2.putText(self.img, "Head", (head_max_loc[0] + 45, head_max_loc[1] + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


if __name__ == '__main__':
    agent = ScreenCaptureAgent()
    agent.capture_screen()
    