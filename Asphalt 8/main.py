import cv2
import numpy as np
import time
import math
import os
import sys
try:
    import tkinter as tk
    from tkinter import messagebox
except (ImportError, Exception):
    tk = None
    messagebox = None

# Ensure the Asphalt 8 directory is in sys.path so directkeys imports reliably
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from directkeys import (
    PressKey,
    ReleaseKey,
    up_pressed,
    down_pressed,
    left_pressed,
    right_pressed,
    nitro_pressed,
    shift_pressed,
    w_pressed,
    s_pressed,
    a_pressed,
    d_pressed,
)

# Universal game controls: Dispatches to both Arrow Keys and WASD + Space/Shift
# Ensures immediate compatibility with online web games (Chrome/Edge/Firefox) and desktop games
SCAN_UP = (up_pressed, w_pressed)            # Accelerate / Gas (Up Arrow + W)
SCAN_DOWN = (down_pressed, s_pressed)        # Brake / Reverse (Down Arrow + S)
SCAN_LEFT = (left_pressed, a_pressed)        # Steer Left (Left Arrow + A)
SCAN_RIGHT = (right_pressed, d_pressed)      # Steer Right (Right Arrow + D)
SCAN_NITRO = (nitro_pressed, shift_pressed)  # Nitro / Boost / Drift (Spacebar + Shift)


class HandTracker:
    def __init__(self):
        self.backend = None
        # Preferred: cvzone HandDetector (supports MediaPipe 1.x Tasks API and legacy)
        try:
            from cvzone.HandTrackingModule import HandDetector
            self.detector = HandDetector(maxHands=2, detectionCon=0.7)
            self.backend = "cvzone"
        except Exception:
            pass

        # Fallback: legacy mediapipe.solutions (MediaPipe <= 0.10)
        if self.backend is None:
            try:
                import mediapipe as mp
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                    self.mp_hands = mp.solutions.hands
                    self.hands = self.mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5,
                    )
                    self.mp_draw = mp.solutions.drawing_utils
                    self.backend = "mediapipe_solutions"
            except Exception:
                pass

        if self.backend is None:
            raise RuntimeError(
                "No supported Hand Tracking backend found! Please ensure 'cvzone' and 'mediapipe' are installed."
            )

    def find_hands(self, img, draw=True):
        hands_data = []

        if self.backend == "cvzone":
            cv_hands, img = self.detector.findHands(img, draw=draw, flipType=False)
            for h in cv_hands:
                lm_list = [(pt[0], pt[1]) for pt in h["lmList"]]
                hands_data.append({"lm_list": lm_list})

        elif self.backend == "mediapipe_solutions":
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            if results.multi_hand_landmarks:
                h_img, w_img, _ = img.shape
                for hand_lms in results.multi_hand_landmarks:
                    lm_list = []
                    for lm in hand_lms.landmark:
                        cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                        lm_list.append((cx, cy))
                    if draw:
                        self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    hands_data.append({"lm_list": lm_list})

        return hands_data, img

    def analyze_hand(self, hand):
        """
        Analyzes hand landmarks and detects:
        - open: open palm (3+ fingers extended) -> Accelerate
        - fist: all fingers curled down into a fist -> Brake / Reverse
        - thumb_up: thumbs-up gesture (thumb elevated, other fingers curled) -> Nitro / Boost
        """
        lm = hand["lm_list"]

        # 4 Fingers: Index (8), Middle (12), Ring (16), Pinky (20)
        # Finger is extended if tip Y is smaller than PIP joint Y (higher in image)
        extended_count = sum(
            1 for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]) if lm[tip][1] < lm[pip][1]
        )

        # Check Thumb elevation (Landmarks: Tip=4, IP=3, MCP=2)
        thumb_is_up = (lm[4][1] < lm[3][1]) and (lm[4][1] < lm[2][1])

        is_open = extended_count >= 3
        is_fist = (extended_count == 0) and not thumb_is_up
        is_thumbs_up = thumb_is_up and (extended_count <= 1)

        return {
            "open": is_open,
            "fist": is_fist,
            "thumb_up": is_thumbs_up,
            "extended_count": extended_count,
        }


class ModernHUD:
    @staticmethod
    def draw_steering_wheel(img, angle):
        h, w, _ = img.shape
        center = (w // 2, h // 2)
        radius = 80

        # Dynamic color based on angle
        if abs(angle) <= 12:
            color = (0, 255, 0)       # Green (Straight / Deadzone)
        elif abs(angle) <= 35:
            color = (0, 165, 255)     # Orange (Steering)
        else:
            color = (0, 0, 255)       # Red (Hard turn)

        # Outer rim and center hub
        cv2.circle(img, center, radius, (60, 60, 60), 7)
        cv2.circle(img, center, radius, color, 3)
        cv2.circle(img, center, 8, color, -1)

        # Rotating spokes
        rad = math.radians(angle)
        p1 = (int(center[0] - radius * math.cos(rad)), int(center[1] - radius * math.sin(rad)))
        p2 = (int(center[0] + radius * math.cos(rad)), int(center[1] + radius * math.sin(rad)))
        cv2.line(img, p1, p2, color, 4)

        # Vertical T-spoke
        p3 = (int(center[0] - (radius * 0.7) * math.sin(rad)), int(center[1] + (radius * 0.7) * math.cos(rad)))
        cv2.line(img, center, p3, color, 3)

        # Steering status text
        if angle < -12:
            text = f"STEER LEFT ({int(-angle)}deg)"
        elif angle > 12:
            text = f"STEER RIGHT ({int(angle)}deg)"
        else:
            text = "STRAIGHT"

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        text_x = center[0] - text_size[0] // 2
        cv2.putText(
            img,
            text,
            (text_x, center[1] + radius + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    @staticmethod
    def draw_single_hand_hud(img, hand_x, steer_state):
        h, w, _ = img.shape
        center_x = w // 2
        deadzone_w = 60
        bar_y = h // 2 + 70
        bar_w = 260
        bar_h = 18

        # Draw slider track
        x_start = center_x - bar_w // 2
        x_end = center_x + bar_w // 2
        cv2.rectangle(img, (x_start, bar_y), (x_end, bar_y + bar_h), (50, 50, 50), -1)

        # Deadzone zone
        cv2.rectangle(img, (center_x - deadzone_w, bar_y), (center_x + deadzone_w, bar_y + bar_h), (70, 70, 70), -1)
        cv2.rectangle(img, (center_x - deadzone_w, bar_y), (center_x + deadzone_w, bar_y + bar_h), (0, 200, 0), 1)

        # Clamp indicator
        hand_indicator_x = max(x_start, min(x_end, hand_x))
        color = (0, 255, 0)
        if steer_state == "LEFT":
            color = (0, 165, 255)
        elif steer_state == "RIGHT":
            color = (0, 165, 255)

        # Draw indicator dot
        cv2.circle(img, (hand_indicator_x, bar_y + bar_h // 2), 10, color, -1)
        cv2.circle(img, (hand_indicator_x, bar_y + bar_h // 2), 12, (255, 255, 255), 2)

        # Labels
        text = f"STEERING: {steer_state}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        cv2.putText(
            img,
            text,
            (center_x - text_size[0] // 2, bar_y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        cv2.putText(img, "< LEFT", (x_start - 65, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "RIGHT >", (x_end + 10, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    @staticmethod
    def draw_status_bar(img, label, value, pos, color):
        x, y = pos
        bar_len = 150
        bar_h = 24

        # Label
        cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
        # Background bar
        cv2.rectangle(img, (x, y), (x + bar_len, y + bar_h), (40, 40, 40), -1)

        # Active fill
        if value:
            cv2.rectangle(img, (x, y), (x + bar_len, y + bar_h), color, -1)
            cv2.putText(
                img,
                "ACTIVE",
                (x + 42, y + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
            )

        # Border
        cv2.rectangle(img, (x, y), (x + bar_len, y + bar_h), (120, 120, 120), 1)


class InstructionUI:
    def __init__(self, on_start):
        self.root = tk.Tk()
        self.root.title("Gestura - Universal Web & Desktop Game Controller")
        self.root.geometry("540x630")
        self.root.configure(bg="#141414")
        self.root.resizable(False, False)
        self.on_start = on_start
        self.should_start = False

        title_font = ("Arial", 17, "bold")
        normal_font = ("Arial", 10)

        main_frame = tk.Frame(self.root, bg="#141414", padx=25, pady=25)
        main_frame.pack(fill="both", expand=True)

        tk.Label(
            main_frame,
            text="UNIVERSAL GESTURE RACING CONTROLLER",
            font=title_font,
            fg="#00ff66",
            bg="#141414",
        ).pack(pady=(0, 6))

        tk.Label(
            main_frame,
            text="Supports Online Web Games (CrazyGames, Poki, WebGL) & Desktop Games",
            font=("Arial", 9, "italic"),
            fg="#aaaaaa",
            bg="#141414",
        ).pack(pady=(0, 12))

        instructions = [
            ("2-HAND STEERING", "Hold both hands up like a steering wheel. Tilt left / right to steer."),
            ("1-HAND STEERING", "Hold one hand up: move left of center to steer left, right to steer right."),
            ("ACCELERATE", "Open palm(s) with fingers pointing up (Sends Up Arrow & W)."),
            ("BRAKE / REVERSE", "Make a closed fist (Sends Down Arrow & S)."),
            ("NITRO / BOOST", "Thumbs-up gesture with either hand (Sends Space & Shift)."),
            ("QUIT", "Press 'Q' inside the camera feed window to stop."),
        ]

        for title, desc in instructions:
            f = tk.Frame(main_frame, bg="#222222", pady=6, padx=12)
            f.pack(fill="x", pady=3)
            tk.Label(
                f,
                text=title,
                font=("Arial", 9, "bold"),
                fg="#00c8ff",
                bg="#222222",
            ).pack(anchor="w")
            tk.Label(
                f,
                text=desc,
                font=normal_font,
                fg="#e0e0e0",
                bg="#222222",
                wraplength=460,
                justify="left",
            ).pack(anchor="w")

        tk.Button(
            main_frame,
            text="START CONTROLLER",
            bg="#00ff66",
            fg="#000000",
            font=("Arial", 12, "bold"),
            command=self.start,
            padx=20,
            pady=10,
            activebackground="#00cc52",
            cursor="hand2",
        ).pack(pady=18)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start(self):
        self.should_start = True
        self.root.destroy()
        if self.on_start:
            self.on_start()

    def on_close(self):
        self.should_start = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()


class Asphalt8Controller:
    def __init__(self, cam_index=0):
        self.tracker = HandTracker()
        self.hud = ModernHUD()
        self.cam_index = cam_index
        self.cap = None
        self.current_keys = set()
        self.running = False

    def init_camera(self):
        # On Windows, DirectShow is far more reliable and avoids Media Foundation locking
        if sys.platform == "win32":
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.cam_index)

        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.cam_index)

        if not self.cap or not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warm up camera by reading initial frames
        for _ in range(5):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.05)

        return True

    def press(self, keys):
        for key in keys:
            if key not in self.current_keys:
                PressKey(key)
                self.current_keys.add(key)

    def release(self, keys):
        for key in keys:
            if key in self.current_keys:
                ReleaseKey(key)
                self.current_keys.discard(key)

    def release_all(self):
        for key in list(self.current_keys):
            ReleaseKey(key)
        self.current_keys.clear()

    def start(self):
        if not self.init_camera():
            print("[ERROR] Could not open webcam! Please verify camera is connected.")
            if tk is not None:
                try:
                    root = tk.Tk()
                    root.withdraw()
                    messagebox.showerror(
                        "Camera Error",
                        "Could not access webcam. Please make sure your camera is connected and not used by another application.",
                    )
                    root.destroy()
                except Exception:
                    pass
            return

        self.running = True
        print("[SUCCESS] Controller Started! Focus on your online web game or Asphalt window.")

        prev_time = time.time()
        consecutive_failures = 0
        try:
            while self.running:
                success, img = self.cap.read()
                if not success:
                    consecutive_failures += 1
                    if consecutive_failures > 15:
                        print("[WARNING] Failed to read frame from webcam.")
                        break
                    time.sleep(0.03)
                    continue
                consecutive_failures = 0

                img = cv2.flip(img, 1)
                hands, img = self.tracker.find_hands(img)

                h, w, _ = img.shape

                # Action states
                accel = False
                brake = False
                nitro = False
                mode_text = "SEARCHING FOR HANDS"

                # ----------------------------------------------------
                # TWO-HAND STEERING WHEEL MODE
                # ----------------------------------------------------
                if len(hands) >= 2:
                    mode_text = "MODE: 2-HAND STEERING WHEEL"
                    # Sort hands by screen horizontal position (wrist X)
                    hands_sorted = sorted(hands, key=lambda h: h["lm_list"][0][0])
                    l_hand = hands_sorted[0]
                    r_hand = hands_sorted[1]

                    p1 = l_hand["lm_list"][0]  # Left wrist (x, y)
                    p2 = r_hand["lm_list"][0]  # Right wrist (x, y)

                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]

                    angle = 0
                    if dx != 0:
                        angle = math.degrees(math.atan2(dy, dx))

                    # Steering controls with deadzone of +/- 12 degrees
                    if angle > 12:
                        self.press(SCAN_RIGHT)
                        self.release(SCAN_LEFT)
                    elif angle < -12:
                        self.press(SCAN_LEFT)
                        self.release(SCAN_RIGHT)
                    else:
                        self.release(SCAN_LEFT)
                        self.release(SCAN_RIGHT)

                    # Hand gesture analysis
                    l_status = self.tracker.analyze_hand(l_hand)
                    r_status = self.tracker.analyze_hand(r_hand)

                    # Accelerate: Both hands open
                    if l_status["open"] and r_status["open"]:
                        accel = True
                        self.press(SCAN_UP)
                        self.release(SCAN_DOWN)
                    # Brake / Reverse: Both hands fists
                    elif l_status["fist"] and r_status["fist"]:
                        brake = True
                        self.press(SCAN_DOWN)
                        self.release(SCAN_UP)
                    else:
                        self.release(SCAN_UP)
                        self.release(SCAN_DOWN)

                    # Nitro / Boost: Thumbs-up on either hand
                    if l_status["thumb_up"] or r_status["thumb_up"]:
                        nitro = True
                        self.press(SCAN_NITRO)
                    else:
                        self.release(SCAN_NITRO)

                    # Render 2-Hand steering wheel
                    self.hud.draw_steering_wheel(img, angle)

                # ----------------------------------------------------
                # SINGLE-HAND CONTROLLER MODE
                # ----------------------------------------------------
                elif len(hands) == 1:
                    mode_text = "MODE: 1-HAND CASUAL"
                    hand = hands[0]
                    wrist_x = hand["lm_list"][0][0]
                    center_x = w // 2
                    deadzone_w = 60

                    steer_state = "STRAIGHT"
                    if wrist_x < (center_x - deadzone_w):
                        steer_state = "LEFT"
                        self.press(SCAN_LEFT)
                        self.release(SCAN_RIGHT)
                    elif wrist_x > (center_x + deadzone_w):
                        steer_state = "RIGHT"
                        self.press(SCAN_RIGHT)
                        self.release(SCAN_LEFT)
                    else:
                        self.release(SCAN_LEFT)
                        self.release(SCAN_RIGHT)

                    # Gestures
                    status = self.tracker.analyze_hand(hand)

                    # Accelerate: Open hand
                    if status["open"]:
                        accel = True
                        self.press(SCAN_UP)
                        self.release(SCAN_DOWN)
                    # Brake / Reverse: Fist
                    elif status["fist"]:
                        brake = True
                        self.press(SCAN_DOWN)
                        self.release(SCAN_UP)
                    else:
                        self.release(SCAN_UP)
                        self.release(SCAN_DOWN)

                    # Nitro: Thumbs-up
                    if status["thumb_up"]:
                        nitro = True
                        self.press(SCAN_NITRO)
                    else:
                        self.release(SCAN_NITRO)

                    # Render 1-Hand HUD
                    self.hud.draw_single_hand_hud(img, wrist_x, steer_state)

                # ----------------------------------------------------
                # NO HANDS VISIBLE
                # ----------------------------------------------------
                else:
                    self.release_all()
                    cv2.putText(
                        img,
                        "SHOW HAND(S) TO DRIVE",
                        (w // 2 - 180, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                # Render Status Bars
                self.hud.draw_status_bar(img, "ACCEL (UP/W)", accel, (20, 50), (0, 255, 0))
                self.hud.draw_status_bar(img, "BRAKE (DN/S)", brake, (20, 110), (0, 0, 255))
                self.hud.draw_status_bar(img, "NITRO (SPC/SHFT)", nitro, (20, 170), (0, 165, 255))

                # Mode Banner at top
                cv2.putText(
                    img,
                    mode_text,
                    (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # Calculate & Render FPS
                curr_time = time.time()
                time_diff = curr_time - prev_time
                fps = 1.0 / time_diff if time_diff > 0 else 0
                prev_time = curr_time
                cv2.putText(
                    img,
                    f"FPS: {int(fps)}",
                    (w - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Gestura - Universal Controller HUD", img)

                # Exit on 'q' or window close
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.running = False
                    break
        finally:
            self.release_all()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Controller stopped and resources released safely.")


def main():
    def launch():
        controller = Asphalt8Controller()
        controller.start()

    ui = InstructionUI(on_start=launch)
    ui.run()


if __name__ == "__main__":
    main()
