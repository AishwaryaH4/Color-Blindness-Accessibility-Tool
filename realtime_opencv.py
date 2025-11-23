import cv2
import time
from utils import simulate_cvd_pil, daltonize_pil
from PIL import Image
import numpy as np
import os


def cv2_to_pil(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def pil_to_cv2(pil_img):
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def stack_horiz(frames, target_h=None):
    """Horizontally stack frames (list of BGR images) resizing to same height."""
   
    if target_h is None:
        target_h = min(f.shape[0] for f in frames)
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        if h != target_h:
            scale = target_h / h
            f = cv2.resize(f, (int(w*scale), target_h))
        resized.append(f)
    return cv2.hconcat(resized)

def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  
    if not cap.isOpened():
        print("ERROR: Could not open camera. Try a different camera index (0,1,...).")
        return

    
    mode = 'protanopia'         
    strength = 1.0              
    save_dir = "captures"
    os.makedirs(save_dir, exist_ok=True)

    print("Controls:")
    print("  1 / 2 / 3 : switch modes (1=protanopia,2=deuteranopia,3=tritanopia)")
    print("  + / -     : increase/decrease correction strength")
    print("  s         : save current corrected frame")
    print("  q / Esc   : quit")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Frame not read from camera. Exiting.")
            break

        
        cur_time = time.time()
        dt = cur_time - prev_time
        if dt > 0:
            fps = 0.9*fps + 0.1*(1.0/dt) if fps else 1.0/dt
        prev_time = cur_time

        
        pil = cv2_to_pil(frame)

        
        simulated_pil = simulate_cvd_pil(pil, mode)
        corrected_pil = daltonize_pil(pil, mode, strength=strength)

       
        simulated_bgr = pil_to_cv2(simulated_pil)
        corrected_bgr = pil_to_cv2(corrected_pil)

        
        def label_img(img, text):
            out = img.copy()
            cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            return out

        orig_lbl = label_img(frame, f"Original  FPS:{fps:.1f}")
        sim_lbl  = label_img(simulated_bgr, f"Simulated ({mode})")
        corr_lbl = label_img(corrected_bgr, f"Corrected  strength:{strength:.2f}")

        
        combined = stack_horiz([orig_lbl, sim_lbl, corr_lbl], target_h=400)

        cv2.imshow("Color Blindness Accessibility - Live", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  
            break
        elif key == ord('1'):
            mode = 'protanopia'
            print("Mode -> protanopia")
        elif key == ord('2'):
            mode = 'deuteranopia'
            print("Mode -> deuteranopia")
        elif key == ord('3'):
            mode = 'tritanopia'
            print("Mode -> tritanopia")
        elif key == ord('+') or key == ord('='):
            strength = min(2.0, strength + 0.1)
            print(f"Strength -> {strength:.2f}")
        elif key == ord('-') or key == ord('_'):
            strength = max(0.0, strength - 0.1)
            print(f"Strength -> {strength:.2f}")
        elif key == ord('s'):
           
            timestamp = int(time.time())
            fname = os.path.join(save_dir, f"corrected_{timestamp}.png")
            cv2.imwrite(fname, corrected_bgr)
            print(f"Saved corrected frame -> {fname}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(camera_index=0)