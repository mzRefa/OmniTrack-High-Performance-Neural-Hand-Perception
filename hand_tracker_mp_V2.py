import cv2
import mediapipe as mp

# --- INITIALIZATION ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
tip_ids = [4, 8, 12, 16, 20]

print("✌️  Individual Hand Counter Active! Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_index, hand_lms in enumerate(results.multi_hand_landmarks):
            # 1. Identify Hand Label
            hand_label = results.multi_handedness[hand_index].classification[0].label
            
            # 2. Get Landmark Pixels
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) != 0:
                fingers = []

                # --- THUMB LOGIC ---
                # Fixed for Mirror View
                if hand_label == "Left":
                    if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # --- 4 FINGERS LOGIC ---
                for id in range(1, 5):
                    if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                count = fingers.count(1)

                # --- 3. DYNAMIC UI (Above each hand) ---
                # We use the wrist coordinates (ID 0) to place the box
                x_wrist, y_wrist = lm_list[0][1], lm_list[0][2]
                
                # Draw a smaller box near the wrist
                box_x1, box_y1 = x_wrist - 50, y_wrist - 150
                box_x2, box_y2 = x_wrist + 50, y_wrist - 50
                
                # Make the box color change based on which hand it is
                color = (0, 255, 0) if hand_label == "Right" else (255, 0, 255)
                
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, cv2.FILLED)
                cv2.putText(frame, str(count), (box_x1 + 20, box_y2 - 20), 
                            cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 10)
                
                # Label the hand
                cv2.putText(frame, hand_label, (box_x1, box_y1 - 10), 
                            cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            # Draw the skeleton dots
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Individual Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
