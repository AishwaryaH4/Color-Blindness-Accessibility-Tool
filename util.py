import numpy as np
from PIL import Image, ImageDraw

def clamp_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


M_PROT = np.array([[0.567, 0.433, 0.0],
                   [0.558, 0.442, 0.0],
                   [0.0,   0.242, 0.758]])

M_DEUT = np.array([[0.625, 0.375, 0.0],
                   [0.7,   0.3,   0.0],
                   [0.0,   0.3,   0.7]])

M_TRIT = np.array([[0.95,  0.05,  0.0],
                   [0.0,   0.433, 0.567],
                   [0.0,   0.475, 0.525]])

SIM_MATRICES = {
    'protanopia': M_PROT,
    'deuteranopia': M_DEUT,
    'tritanopia': M_TRIT
}

def simulate_cvd_pil(pil_img: Image.Image, mode: str) -> Image.Image:
    """
    Simulate how an image looks under a color vision deficiency.
    pil_img must be RGB.
    """
    if mode not in SIM_MATRICES:
        raise ValueError("mode must be 'protanopia'|'deuteranopia'|'tritanopia'")
    M = SIM_MATRICES[mode]
    arr = np.array(pil_img).astype(np.float32)  
    simulated = arr @ M.T
    simulated = clamp_uint8(simulated)
    return Image.fromarray(simulated)

def daltonize_pil(pil_img: Image.Image, mode: str, strength: float = 1.0) -> Image.Image:
    """
    Simple daltonize-like correction:
    - simulate the deficient view
    - compute error (orig - simulated)
    - add scaled error back into the original to increase separability
    """
    orig = np.array(pil_img).astype(np.float32)
    simulated = np.array(simulate_cvd_pil(pil_img, mode)).astype(np.float32)
    error = orig - simulated
    corrected = orig + error * float(strength)
    corrected = clamp_uint8(corrected)
    return Image.fromarray(corrected)

def add_pattern_overlay(original_pil: Image.Image,
                        corrected_pil: Image.Image,
                        mode: str,
                        threshold: float = 60.0,
                        pattern: str = "dots",
                        tile: int = 12,
                        color: tuple = (0, 0, 0, 140)) -> Image.Image:
    """
    Add a pattern (dots or stripes) over regions likely confusing to color-blind users.
    - original_pil: original RGB PIL image
    - corrected_pil: corrected RGB PIL image (we'll paste pattern onto this)
    - mode: used to compute simulated view to find confusing regions
    - threshold: pixel-difference threshold for marking (higher => fewer marks)
    - pattern: 'dots' or 'stripes'
    - tile: pattern tile size in pixels (smaller => denser pattern)
    - color: RGBA tuple for the overlay pattern
    Returns: corrected image (RGB) with pattern overlay applied where mask is True.
    """
    
    orig_arr = np.array(original_pil).astype(np.float32)
    sim_arr = np.array(simulate_cvd_pil(original_pil, mode)).astype(np.float32)
    diff = np.abs(orig_arr - sim_arr).sum(axis=2)  
    mask_bool = diff > float(threshold)

    
    if not mask_bool.any():
        return corrected_pil.copy()

    
    mask_img = Image.fromarray((mask_bool.astype('uint8') * 255)).convert('L')

    
    w, h = original_pil.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if pattern == "dots":
        r = max(1, tile // 4)
        for y in range(0, h, tile):
            
            row_offset = (tile // 2) if (y // tile) % 2 else 0
            for x in range(0, w, tile):
                cx = x + row_offset + tile // 2
                cy = y + tile // 2
               
                if cx < 0 or cx > w or cy < 0 or cy > h:
                    continue
                bbox = (cx - r, cy - r, cx + r, cy + r)
                draw.ellipse(bbox, fill=color)
    else:  
        step = tile
        thickness = max(1, tile // 3)
        
        for offset in range(-h, w, step):
            x1 = offset
            y1 = 0
            x2 = offset + h
            y2 = h
            draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)

   
    base = corrected_pil.convert("RGBA").copy()
    base.paste(overlay, (0, 0), mask=mask_img)

    return base.convert("RGB")