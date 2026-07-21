"""
Text-anchor based OCR extraction for Instagram Reel/Post insights screenshots.
Works across iOS/Android, Reel/Post, and different screen sizes because it locates
fields relative to always-present text anchors ("Overview", "Views", "Follows", etc.)
instead of fixed pixel coordinates. No AI/LLM calls — OpenCV + Tesseract + geometry only.
"""
import io
import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image

FIELD_KEYS = [
    "views", "accounts_reached", "profile_visits", "follows",
    "likes", "comments", "shares", "sends", "saves",
]

# Icons always appear in this left-to-right order in Instagram's engagement row:
# heart (likes) -> comment -> repost/share -> paper-plane (sends) -> bookmark (saves)
ICON_ORDER = ["likes", "comments", "shares", "sends", "saves"]

# Multi-language label text used to find each Summary-grid card
LABEL_KEYWORDS = {
    "views": ["views", "vistas", "visualizacoes", "visualizaciones", "vues", "reproducciones"],
    "accounts_reached": ["accounts reached", "cuentas alcanzadas", "contas alcancadas",
                          "contas alcançadas", "comptes atteints"],
    "profile_visits": ["profile visits", "visitas al perfil", "visitas del perfil", "visitas ao perfil",
                        "visites du profil"],
    "follows": ["follows", "seguidores nuevos", "seguidores", "seguiram", "abonnements"],
}

# Anchors used to locate the bottom of the 5-icon engagement row across languages
ANCHOR_KEYWORDS = ["overview", "resumen", "visão geral", "visao geral", "aperçu", "apercu"]

# A full numeric token: 606 / 4.9K / 1,170,214 / 68,460 -- but NOT things like "5:14" or "79%"
NUMBER_TOKEN_RE = re.compile(r"^[\d][\d.,]*[KkMm]?$")
NUMBER_SEARCH_RE = re.compile(r"[\d][\d.,]*\s?[KkMm]?")


def _preprocess(image_bytes: bytes):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    scale = 1.0
    if max(h, w) > 1600:
        scale = 1400.0 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    elif max(h, w) < 900:
        scale = 1.5
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Invert dark mode screenshots (white text on dark bg) to black text on white bg for optimal Tesseract OCR
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return thresh, scale


def _words(image_array) -> list[dict]:
    try:
        data = pytesseract.image_to_data(
            image_array, lang="eng+spa+por+fra", output_type=Output.DICT
        )
    except Exception:
        data = pytesseract.image_to_data(
            image_array, lang="eng", output_type=Output.DICT
        )
    out = []
    for i in range(len(data["text"])):
        t = data["text"][i].strip()
        if not t:
            continue
        out.append({
            "text": t, "x": data["left"][i], "y": data["top"][i],
            "w": data["width"][i], "h": data["height"][i],
            "block": data["block_num"][i], "par": data["par_num"][i], "line": data["line_num"][i],
        })
    return out


def _lines_from_words(words: list[dict]) -> list[dict]:
    """Group OCR words into lines (by block/paragraph/line id)."""
    groups = {}
    for w in words:
        key = (w["block"], w["par"], w["line"])
        groups.setdefault(key, []).append(w)

    lines = []
    for group in groups.values():
        group.sort(key=lambda w: w["x"])
        text = " ".join(w["text"] for w in group)
        x0 = min(w["x"] for w in group)
        y0 = min(w["y"] for w in group)
        lines.append({"text": text, "x": x0, "y0": y0, "words": group})

    lines.sort(key=lambda l: l["y0"])
    return lines


def _is_number_word(text: str) -> bool:
    return bool(NUMBER_TOKEN_RE.match(text)) and any(c.isdigit() for c in text)


def _extract_icon_row(words: list[dict], overview_y: int | None, scale: int) -> dict:
    """Find the 5 engagement numbers sitting just above the anchor tab,
    ordered left-to-right, and map them by position to likes/comments/shares/sends/saves."""
    if overview_y is not None:
        candidates = [w for w in words if w["y"] < overview_y and _is_number_word(w["text"])]
    else:
        candidates = [w for w in words if _is_number_word(w["text"])]

    if not candidates:
        return {}

    # The row closest above the anchor (largest y among candidates) is the icon row —
    # this naturally skips status-bar numbers (battery %, clock) which sit much higher up.
    candidates.sort(key=lambda w: -w["y"])
    band_y = candidates[0]["y"]
    band_tolerance = 40 * scale
    band = [w for w in candidates if abs(w["y"] - band_y) < band_tolerance]
    band.sort(key=lambda w: w["x"])

    result = {}
    for key, w in zip(ICON_ORDER, band[:5]):
        result[key] = w["text"]
    return result


def _closest_number_near_label(lines: list[dict], label_line: dict, img_h: int) -> str | None:
    """Find the number line that best matches this label: nearby (below or above) and left-aligned
    with it (handles multi-column grid cards in various IG UI layouts)."""
    best_val, best_score = None, None
    max_gap = img_h * 0.15
    for line in lines:
        if line == label_line:
            continue
        dy = abs(line["y0"] - label_line["y0"])
        if dy > max_gap:
            continue
        m = NUMBER_SEARCH_RE.search(line["text"])
        if not m:
            continue
        dx = abs(line["x"] - label_line["x"])
        score = dx + dy * 0.4
        if best_score is None or score < best_score:
            best_score = score
            best_val = m.group().strip()
    return best_val


def _extract_summary_grid(lines: list[dict], img_h: int) -> dict:
    result = {}
    for field, keywords in LABEL_KEYWORDS.items():
        for line in lines:
            norm = line["text"].lower()
            if any(kw in norm for kw in keywords):
                val = _closest_number_near_label(lines, line, img_h)
                if val:
                    result[field] = val
                break
    return result


def extract_fields_from_image(image_bytes: bytes) -> dict:
    processed, scale = _preprocess(image_bytes)
    img_h = processed.shape[0]

    words = _words(processed)
    lines = _lines_from_words(words)

    overview_y = None
    for line in lines:
        line_norm = line["text"].lower()
        if any(anchor in line_norm for anchor in ANCHOR_KEYWORDS):
            overview_y = line["y0"]
            break

    found = {k: None for k in FIELD_KEYS}
    found.update(_extract_icon_row(words, overview_y, scale))
    found.update(_extract_summary_grid(lines, img_h))
    return found


def extract_fields_from_images(images: list[bytes]) -> dict:
    """Merge across multiple screenshots. First non-null match per field wins,
    so you can upload e.g. a Reel screenshot and a Post screenshot together if needed."""
    merged = {k: None for k in FIELD_KEYS}
    for image_bytes in images:
        try:
            partial = extract_fields_from_image(image_bytes)
        except Exception:
            partial = {}
        for k, val in partial.items():
            if merged.get(k) is None and val:
                merged[k] = val
    return merged

