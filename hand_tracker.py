import cv2
import mediapipe as mp
import time

# --- INITIALIZE ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
prev_time = 0

print("👋 Hand Tracker Started! Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Flip for a mirror effect
    frame = cv2.flip(frame, 1)
    # MediaPipe needs RGB, OpenCV uses BGR
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(img_rgb)

    # If hands are found, draw them
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Draw dots (landmarks) and lines (connections)
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            

    # FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Tracker - Machine Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
