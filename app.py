import os
import sys
import math
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="Gestura - AI Gesture Game Controller",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Modern Racing HUD CSS Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark Theme & Typography */
    .main {
        background-color: #0e1117;
    }
    
    .hud-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00ff66, #00c8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .hud-subtitle {
        font-size: 1.05rem;
        color: #a0aec0;
        margin-bottom: 1.5rem;
    }

    .telemetry-card {
        background: #171b26;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    .telemetry-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #718096;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .telemetry-value-active {
        font-size: 1.25rem;
        font-weight: 700;
        color: #00ff66;
    }

    .telemetry-value-nitro {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffaa00;
    }

    .telemetry-value-brake {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ff3344;
    }

    .telemetry-value-idle {
        font-size: 1.25rem;
        font-weight: 600;
        color: #4a5568;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Hand Tracker Class (Cross-Platform)
# ---------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                # palm base
]

def draw_skeleton(img, lm_list):
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(lm_list) and end_idx < len(lm_list):
            p1 = lm_list[start_idx]
            p2 = lm_list[end_idx]
            cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 128), 2)
    for pt in lm_list:
        cv2.circle(img, (pt[0], pt[1]), 4, (0, 200, 255), -1)
        cv2.circle(img, (pt[0], pt[1]), 5, (255, 255, 255), 1)

def get_model_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "hand_landmarker.task"),
        os.path.join(current_dir, "Asphalt 8", "hand_landmarker.task"),
        os.path.abspath("hand_landmarker.task"),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 1000000:
            return p

    target = os.path.join(current_dir, "hand_landmarker.task")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    try:
        import requests
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if resp.status_code == 200:
            with open(target, "wb") as f:
                f.write(resp.content)
            return target
    except Exception:
        pass

    try:
        import urllib.request
        urllib.request.urlretrieve(url, target)
        return target
    except Exception:
        pass

    return target


class WebHandTracker:
    def __init__(self, detection_con=0.7):
        self.backend = None
        self.error_msg = None
        self.detection_con = detection_con
        model_path = get_model_path()

        # 1. Primary: Direct MediaPipe Tasks API (modern mediapipe >= 0.10.0)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if os.path.exists(model_path):
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=2,
                    running_mode=vision.RunningMode.IMAGE,
                    min_hand_detection_confidence=float(self.detection_con),
                    min_hand_presence_confidence=float(self.detection_con),
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.backend = "mediapipe_tasks"
        except Exception as e:
            self.error_msg = str(e)

        # 2. Secondary fallback: cvzone HandDetector
        if self.backend is None:
            try:
                from cvzone.HandTrackingModule import HandDetector
                self.detector = HandDetector(maxHands=2, detectionCon=float(self.detection_con))
                self.backend = "cvzone"
            except Exception as e:
                if not self.error_msg:
                    self.error_msg = str(e)

        # 3. Tertiary fallback: legacy mediapipe solutions
        if self.backend is None:
            try:
                import mediapipe as mp
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                    self.mp_hands = mp.solutions.hands
                    self.hands = self.mp_hands.Hands(
                        static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=float(self.detection_con),
                        min_tracking_confidence=0.5,
                    )
                    self.backend = "mediapipe_solutions"
            except Exception as e:
                if not self.error_msg:
                    self.error_msg = str(e)

        if self.backend is None:
            st.error(f"Could not initialize Hand Tracking engine: {self.error_msg or 'Model could not be loaded'}")

    def find_hands(self, img, draw=True):
        hands_data = []
        if self.backend is None:
            return hands_data, img

        h, w, _ = img.shape
        if self.backend == "mediapipe_tasks":
            import mediapipe as mp
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = self.detector.detect(mp_image)

            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    lm_list = []
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append((cx, cy))
                    if draw:
                        draw_skeleton(img, lm_list)
                    hands_data.append({"lm_list": lm_list})

        elif self.backend == "cvzone":
            cv_hands, img = self.detector.findHands(img, draw=draw, flipType=False)
            for hnd in cv_hands:
                lm_list = [(int(pt[0]), int(pt[1])) for pt in hnd["lmList"]]
                hands_data.append({"lm_list": lm_list})

        elif self.backend == "mediapipe_solutions":
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    lm_list = []
                    for lm in hand_lms.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append((cx, cy))
                    if draw:
                        draw_skeleton(img, lm_list)
                    hands_data.append({"lm_list": lm_list})

        return hands_data, img

    def analyze_hand(self, hand):
        lm = hand["lm_list"]
        extended_count = sum(
            1 for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]) if lm[tip][1] < lm[pip][1]
        )
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


# ---------------------------------------------------------
# Modern HUD Drawing Engine
# ---------------------------------------------------------
class StreamlitHUD:
    @staticmethod
    def draw_steering_wheel(img, angle):
        h, w, _ = img.shape
        center = (w // 2, h // 2)
        radius = min(80, h // 4)

        if abs(angle) <= 12:
            color = (0, 255, 0)       # Green
        elif abs(angle) <= 35:
            color = (0, 165, 255)     # Orange
        else:
            color = (0, 0, 255)       # Red

        cv2.circle(img, center, radius, (60, 60, 60), 6)
        cv2.circle(img, center, radius, color, 3)
        cv2.circle(img, center, 8, color, -1)

        rad = math.radians(angle)
        p1 = (int(center[0] - radius * math.cos(rad)), int(center[1] - radius * math.sin(rad)))
        p2 = (int(center[0] + radius * math.cos(rad)), int(center[1] + radius * math.sin(rad)))
        cv2.line(img, p1, p2, color, 4)

        p3 = (int(center[0] - (radius * 0.7) * math.sin(rad)), int(center[1] + (radius * 0.7) * math.cos(rad)))
        cv2.line(img, center, p3, color, 3)

        if angle < -12:
            text = f"STEER LEFT ({int(-angle)} deg)"
        elif angle > 12:
            text = f"STEER RIGHT ({int(angle)} deg)"
        else:
            text = "STRAIGHT"

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        text_x = max(10, center[0] - text_size[0] // 2)
        cv2.putText(
            img,
            text,
            (text_x, center[1] + radius + 32),
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
        bar_y = h // 2 + 60
        bar_w = min(260, w - 40)
        bar_h = 18

        x_start = center_x - bar_w // 2
        x_end = center_x + bar_w // 2
        cv2.rectangle(img, (x_start, bar_y), (x_end, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(img, (center_x - deadzone_w, bar_y), (center_x + deadzone_w, bar_y + bar_h), (0, 200, 0), 1)

        hand_indicator_x = max(x_start, min(x_end, hand_x))
        color = (0, 255, 0) if steer_state == "STRAIGHT" else (0, 165, 255)

        cv2.circle(img, (hand_indicator_x, bar_y + bar_h // 2), 10, color, -1)
        cv2.circle(img, (hand_indicator_x, bar_y + bar_h // 2), 12, (255, 255, 255), 2)

        text = f"STEERING: {steer_state}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(img, text, (center_x - text_size[0] // 2, bar_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @staticmethod
    def draw_status_bar(img, label, value, pos, color):
        x, y = pos
        bar_len = 140
        bar_h = 22
        cv2.putText(img, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.rectangle(img, (x, y), (x + bar_len, y + bar_h), (40, 40, 40), -1)
        if value:
            cv2.rectangle(img, (x, y), (x + bar_len, y + bar_h), color, -1)
            cv2.putText(img, "ACTIVE", (x + 36, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.rectangle(img, (x, y), (x + bar_len, y + bar_h), (120, 120, 120), 1)


# ---------------------------------------------------------
# Core Image Processing Pipeline
# ---------------------------------------------------------
def process_gesture_frame(tracker, hud, image_bgr, deadzone=12, draw_landmarks=True):
    annotated = image_bgr.copy()
    hands, annotated = tracker.find_hands(annotated, draw=draw_landmarks)
    h, w, _ = annotated.shape

    telemetry = {
        "hands_detected": len(hands),
        "mode": "SEARCHING FOR HANDS",
        "steering": "STRAIGHT",
        "angle": 0.0,
        "accel": False,
        "brake": False,
        "nitro": False,
        "keys": [],
    }

    if len(hands) >= 2:
        telemetry["mode"] = "2-HAND STEERING WHEEL"
        hands_sorted = sorted(hands, key=lambda hand: hand["lm_list"][0][0])
        l_hand, r_hand = hands_sorted[0], hands_sorted[1]
        p1 = l_hand["lm_list"][0]
        p2 = r_hand["lm_list"][0]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx)) if dx != 0 else 0
        telemetry["angle"] = round(angle, 1)

        if angle > deadzone:
            telemetry["steering"] = f"RIGHT (+{int(angle)}°)"
            telemetry["keys"].append("Right Arrow / D")
        elif angle < -deadzone:
            telemetry["steering"] = f"LEFT ({int(angle)}°)"
            telemetry["keys"].append("Left Arrow / A")
        else:
            telemetry["steering"] = "STRAIGHT"

        l_stat = tracker.analyze_hand(l_hand)
        r_stat = tracker.analyze_hand(r_hand)

        if l_stat["open"] and r_stat["open"]:
            telemetry["accel"] = True
            telemetry["keys"].append("Up Arrow / W")
        elif l_stat["fist"] and r_stat["fist"]:
            telemetry["brake"] = True
            telemetry["keys"].append("Down Arrow / S")

        if l_stat["thumb_up"] or r_stat["thumb_up"]:
            telemetry["nitro"] = True
            telemetry["keys"].append("Spacebar / Shift")

        hud.draw_steering_wheel(annotated, angle)

    elif len(hands) == 1:
        telemetry["mode"] = "1-HAND CASUAL"
        hand = hands[0]
        wrist_x = hand["lm_list"][0][0]
        center_x = w // 2
        deadzone_w = 60

        steer_state = "STRAIGHT"
        if wrist_x < (center_x - deadzone_w):
            steer_state = "LEFT"
            telemetry["keys"].append("Left Arrow / A")
        elif wrist_x > (center_x + deadzone_w):
            steer_state = "RIGHT"
            telemetry["keys"].append("Right Arrow / D")

        telemetry["steering"] = steer_state
        stat = tracker.analyze_hand(hand)

        if stat["open"]:
            telemetry["accel"] = True
            telemetry["keys"].append("Up Arrow / W")
        elif stat["fist"]:
            telemetry["brake"] = True
            telemetry["keys"].append("Down Arrow / S")

        if stat["thumb_up"]:
            telemetry["nitro"] = True
            telemetry["keys"].append("Spacebar / Shift")

        hud.draw_single_hand_hud(annotated, wrist_x, steer_state)

    else:
        cv2.putText(
            annotated,
            "SHOW HAND(S) TO DRIVE",
            (max(20, w // 2 - 170), h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            2,
        )

    hud.draw_status_bar(annotated, "ACCEL (UP/W)", telemetry["accel"], (20, 45), (0, 255, 0))
    hud.draw_status_bar(annotated, "BRAKE (DN/S)", telemetry["brake"], (20, 100), (0, 0, 255))
    hud.draw_status_bar(annotated, "NITRO (SPC/SHFT)", telemetry["nitro"], (20, 155), (0, 165, 255))

    cv2.putText(annotated, telemetry["mode"], (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return annotated, telemetry


@st.cache_resource
def load_tracker(detection_con):
    return WebHandTracker(detection_con=detection_con)


# ---------------------------------------------------------
# Streamlit Web App Interface
# ---------------------------------------------------------
def main():
    # Header Section
    st.markdown('<div class="hud-title">🏎️ GESTURA</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hud-subtitle">AI-Powered Universal Gesture Controller for Racing Games (Asphalt 8, CrazyGames, Poki, WebGL)</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        app_mode = st.radio(
            "Navigation Mode",
            ["📷 Camera Gesture Tester", "📁 Image Upload Tester", "📖 Controls & Instructions", "🚀 Desktop Game Launcher"],
        )
        st.markdown("---")
        st.subheader("🎯 HUD & AI Sensitivity")
        deadzone = st.slider("Steering Deadzone (Degrees)", min_value=5, max_value=25, value=12, step=1)
        detection_con = st.slider("Detection Confidence", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
        draw_landmarks = st.checkbox("Draw Skeleton Landmarks", value=True)
        st.markdown("---")
        st.markdown(
            "**GitHub Repository**: [SAFAL-TIWARI/Gestura](https://github.com/SAFAL-TIWARI/Gestura)\n\n"
            "**Supported Games**:\n"
            "* Asphalt 8: Airborne\n"
            "* Online WebGL & HTML5 Games\n"
            "* Any PC Racing Game with Arrow/WASD keys"
        )

    # Initialize Tracker & HUD
    tracker = load_tracker(detection_con)
    hud = StreamlitHUD()

    # ---------------------------------------------------------
    # TAB 1: Camera Gesture Tester (Webcam)
    # ---------------------------------------------------------
    if app_mode == "📷 Camera Gesture Tester":
        st.info("💡 Grant camera permission in your browser. Hold up one or both hands to test the virtual steering wheel and gestures!")
        
        img_file = st.camera_input("Take a photo to test gesture detection & HUD overlay")

        if img_file is not None:
            bytes_data = img_file.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            # Flip horizontally for natural mirror behavior
            cv_img = cv2.flip(cv_img, 1)

            with st.spinner("Analyzing gestures..."):
                processed_img, telemetry = process_gesture_frame(
                    tracker, hud, cv_img, deadzone=deadzone, draw_landmarks=draw_landmarks
                )
                rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([1.6, 1.0])
            with col1:
                st.subheader("HUD Video Output")
                st.image(rgb_img, use_container_width=True)

            with col2:
                st.subheader("📊 Live Telemetry")
                
                # Steering Card
                steer_class = "telemetry-value-active" if "STRAIGHT" not in telemetry["steering"] else "telemetry-value-idle"
                st.markdown(
                    f"""
                    <div class="telemetry-card">
                        <div class="telemetry-label">🕹️ Steering Position</div>
                        <div class="{steer_class}">{telemetry['steering']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Throttle Card
                accel_class = "telemetry-value-active" if telemetry["accel"] else "telemetry-value-idle"
                accel_text = "ACCELERATING (W / Up Arrow)" if telemetry["accel"] else "OFF"
                st.markdown(
                    f"""
                    <div class="telemetry-card">
                        <div class="telemetry-label">🏎️ Throttle (Gas)</div>
                        <div class="{accel_class}">{accel_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Brake Card
                brake_class = "telemetry-value-brake" if telemetry["brake"] else "telemetry-value-idle"
                brake_text = "BRAKING / REVERSE (S / Down Arrow)" if telemetry["brake"] else "OFF"
                st.markdown(
                    f"""
                    <div class="telemetry-card">
                        <div class="telemetry-label">🛑 Brake / Reverse</div>
                        <div class="{brake_class}">{brake_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Nitro Card
                nitro_class = "telemetry-value-nitro" if telemetry["nitro"] else "telemetry-value-idle"
                nitro_text = "NITRO BOOST (Spacebar / Shift)" if telemetry["nitro"] else "OFF"
                st.markdown(
                    f"""
                    <div class="telemetry-card">
                        <div class="telemetry-label">⚡ Nitro Boost</div>
                        <div class="{nitro_class}">{nitro_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Mode & Keys
                st.caption(f"**Controller Mode**: `{telemetry['mode']}` | **Hands Detected**: `{telemetry['hands_detected']}`")
                if telemetry["keys"]:
                    st.success(f"**Dispatched Keys**: {', '.join(telemetry['keys'])}")

    # ---------------------------------------------------------
    # TAB 2: Image Upload Tester
    # ---------------------------------------------------------
    elif app_mode == "📁 Image Upload Tester":
        st.write("Upload any image or snapshot to test the hand detection and HUD visualization.")
        uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_np = np.array(image)
            cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            processed_img, telemetry = process_gesture_frame(
                tracker, hud, cv_img, deadzone=deadzone, draw_landmarks=draw_landmarks
            )
            rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([1.6, 1.0])
            with col1:
                st.subheader("HUD Visualization")
                st.image(rgb_img, use_container_width=True)
            with col2:
                st.subheader("Telemetry Analysis")
                st.json(telemetry)

    # ---------------------------------------------------------
    # TAB 3: Controls & Instructions
    # ---------------------------------------------------------
    elif app_mode == "📖 Controls & Instructions":
        st.subheader("🎮 Universal Gesture Controls")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                ### 🚗 Steering Gestures
                * **2-Hand Steering Wheel Mode**:
                  * Hold both hands up at 9 and 3 o'clock positions like gripping a real steering wheel.
                  * **Tilt Left** (Left hand lower, Right hand higher) $\\rightarrow$ **Steer Left**
                  * **Tilt Right** (Right hand lower, Left hand higher) $\\rightarrow$ **Steer Right**
                * **1-Hand Casual Mode**:
                  * Move your hand to the left of the center screen $\\rightarrow$ **Steer Left**
                  * Move your hand to the right of the center screen $\\rightarrow$ **Steer Right**
                """
            )
        with c2:
            st.markdown(
                """
                ### ⚡ Driving Actions
                * ✋ **Accelerate / Gas**:
                  * Open palms with fingers pointing up (Sends **Up Arrow** and **W**).
                * ✊ **Brake / Reverse**:
                  * Close both hands into a tight fist (Sends **Down Arrow** and **S**).
                * 👍 **Nitro / Drift Boost**:
                  * Make a thumbs-up gesture with either hand (Sends **Spacebar** and **Shift**).
                """
            )

        st.markdown("---")
        st.subheader("🕹️ Key Mappings Summary")
        st.table(
            [
                {"Gesture": "Tilt Left / Move Left", "Action": "Steer Left", "Hardware Keys": "Left Arrow / A"},
                {"Gesture": "Tilt Right / Move Right", "Action": "Steer Right", "Hardware Keys": "Right Arrow / D"},
                {"Gesture": "Open Palm (3+ fingers up)", "Action": "Accelerate (Gas)", "Hardware Keys": "Up Arrow / W"},
                {"Gesture": "Closed Fist", "Action": "Brake / Reverse", "Hardware Keys": "Down Arrow / S"},
                {"Gesture": "Thumbs Up", "Action": "Nitro / Drift", "Hardware Keys": "Spacebar / Shift"},
            ]
        )

    # ---------------------------------------------------------
    # TAB 4: Desktop Game Launcher
    # ---------------------------------------------------------
    elif app_mode == "🚀 Desktop Game Launcher":
        st.subheader("🖥️ Run Gestura on Local PC (Direct Game Control)")
        st.markdown(
            """
            When playing games locally on your Windows PC (e.g., Asphalt 8: Airborne or browser games):
            
            1. **Open your Terminal** in the Gestura project folder.
            2. Run:
            ```bash
            python main.py
            ```
            3. The interactive controller HUD will launch with zero latency.
            4. Switch focus to your game window and enjoy hands-free motion driving!
            """
        )
        if sys.platform == "win32":
            st.success("You are running on Windows. You can launch `main.py` directly from your command line!")
        else:
            st.info("You are currently viewing the cloud deployment of Gestura.")


if __name__ == "__main__":
    main()
