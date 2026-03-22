import cv2
import mediapipe as mp
import numpy as np

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

print("🖥️  Simulation Mode Active! Two windows opening...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # 1. Create a Blank Black Canvas for the Simulation
    sim_window = np.zeros((h, w, 3), dtype=np.uint8)
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_index, hand_lms in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[hand_index].classification[0].label
            
            # Draw on Camera Window
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # --- 2. DRAW ON SIMULATION WINDOW ---
            # We change the color for the simulation to make it look "Cyber"
            sim_color = mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
            conn_color = mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)
            
            mp_draw.draw_landmarks(
                sim_window, 
                hand_lms, 
                mp_hands.HAND_CONNECTIONS,
                sim_color,
                conn_color
            )

            # --- Logic for Labels (Shared between windows) ---
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                lm_list.append([id, int(lm.x * w), int(lm.y * h)])

            if len(lm_list) != 0:
                # Add a label in the simulation window
                x_wrist, y_wrist = lm_list[0][1], lm_list[0][2]
                cv2.putText(sim_window, hand_label, (x_wrist - 30, y_wrist + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 3. SHOW BOTH WINDOWS
    cv2.imshow("Camera Feed (Reality)", frame)
    cv2.imshow("Hand Simulation (Digital Twin)", sim_window)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
