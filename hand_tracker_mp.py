import cv2
import mediapipe as mp

# --- INITIALIZATION ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
# model_complexity=0 is the lightest version for i5 CPU
hands = mp_hands.Hands(
    model_complexity=0, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Landmarks for finger tips: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
tip_ids = [4, 8, 12, 16, 20]

print("🖐️  Multi-Hand Finger Counter Active! Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. Image Pre-processing
    frame = cv2.flip(frame, 1)  # Mirror view
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    total_fingers = 0

    # 2. Detection Logic
    if results.multi_hand_landmarks:
        # Loop through each detected hand
        for hand_index, hand_lms in enumerate(results.multi_hand_landmarks):
            # Get the Hand Label (Left or Right)
            # MediaPipe flips these internally, so we handle the logic below
            hand_label = results.multi_handedness[hand_index].classification[0].label
            
            # Draw the skeleton lines
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to pixel coordinates
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) != 0:
                fingers = []

                # --- THUMB LOGIC  ---
                if hand_label == "Left":
                    if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else: # Right Hand
                    if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # --- 4 FINGERS LOGIC ---
                # Check if the TIP (y) is higher (smaller value) than the knuckle PIP (y)
                for id in range(1, 5):
                    if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Add this hand's fingers to the total
                total_fingers += fingers.count(1)

        # 3. UI DISPLAY
        # Draw the "Eraser" background box
        cv2.rectangle(frame, (20, 20), (150, 150), (0, 255, 0), cv2.FILLED)
        # Display the combined finger count from all hands
        cv2.putText(frame, str(total_fingers), (45, 125), 
                    cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 15)

    # 4. Show output
    cv2.imshow("Hand Tracker - Mediapipe Branch", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
