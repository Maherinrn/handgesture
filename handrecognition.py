import cv2
import mediapipe as mp
import pyautogui

# Initialize mediapipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize the video capture object
cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

def control_mouse(x, y):
    pyautogui.moveTo(x, y)

def left_click():
    pyautogui.click()

def right_click():
    pyautogui.click(button='right')

def scroll(direction):
    if direction == "up":
        pyautogui.scroll(10)
    else:
        pyautogui.scroll(-10)

def detect_gestures(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    thumb_index_dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    index_middle_dist = ((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2) ** 0.5
    all_fingers_open = all(finger.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y for finger in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip])

    return thumb_tip, index_tip, middle_tip, thumb_index_dist, index_middle_dist, all_fingers_open

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip, index_tip, middle_tip, thumb_index_dist, index_middle_dist, all_fingers_open = detect_gestures(hand_landmarks)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            screen_x, screen_y = int(screen_width * index_tip.x), int(screen_height * index_tip.y)
            control_mouse(screen_x, screen_y)

            if thumb_index_dist < 0.05 and index_middle_dist < 0.05:
                left_click()
            elif thumb_index_dist < 0.05:
                right_click()
            elif all_fingers_open:
                scroll("up")
            else:
                scroll("down")

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
