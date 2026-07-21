"""
Text-anchor based OCR extraction for Instagram Reel/Post insights screenshots.
Works across iOS/Android, Reel/Post, and different screen sizes because it locates
fields relative to always-present text anchors ("Overview", "Views", "Follows", etc.)
instead of fixed pixel coordinates. No AI/LLM calls — OpenCV + Tesseract + geometry only.
"""
import base64
import io
import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image

FIELD_KEYS = [
    "views", "accounts_reached", "profile_visits", "average_watch_time", "follows",
    "likes", "comments", "shares", "sends", "saves", "post_type", "post_thumbnail"
]

ICON_ORDER = ["likes", "comments", "shares", "sends", "saves"]

LABEL_KEYWORDS_REEL = {
    "views": ["views", "vistas", "visualizacoes", "visualizaciones", "vues", "reproducciones"],
    "accounts_reached": ["accounts reached", "cuentas alcanzadas", "contas alcancadas",
                          "contas alcançadas", "comptes atteints"],
    "average_watch_time": ["average watch time", "watch time", "tiempo de reproducción",
                            "tiempo promedio", "tempo de reprodução", "durée moyenne"],
    "follows": ["follows", "seguidores nuevos", "seguidores", "seguiram", "abonnements"],
}

LABEL_KEYWORDS_POST = {
    "views": ["views", "vistas", "visualizacoes", "visualizaciones", "vues", "reproducciones", "impresiones", "impressions"],
    "accounts_reached": ["accounts reached", "cuentas alcanzadas", "contas alcancadas",
                          "contas alcançadas", "comptes atteints"],
    "profile_visits": ["profile visits", "visitas al perfil", "visitas del perfil", "visitas ao perfil",
                        "visites du profil"],
    "follows": ["follows", "seguidores nuevos", "seguidores", "seguiram", "abonnements"],
}

ANCHOR_KEYWORDS = ["overview", "resumen", "visão geral", "visao geral", "aperçu", "apercu"]

NUMBER_TOKEN_RE = re.compile(r"^[\d][\d.,]*[KkMm]?$")
NUMBER_SEARCH_RE = re.compile(r"[\d][\d.,\s]*[\d][KkMm]?|[\d]+[KkMm]?")
TIME_SEARCH_RE = re.compile(r"[\d]+\s?[sS]|[\d]+[:\.]\d{2}|[\d]+\s?[mM]\s?[\d]+\s?[sS]|[\d]+")


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

    # Invert dark mode screenshots (white text on dark bg) to black text on white bg
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    # Contrast normalization without destructive adaptive thresholding
    gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return gray, scale


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
    """Group OCR words into spatial lines based on vertical center alignment,
    and split text into separate column line items if separated by horizontal gaps (> 45px)."""
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w["y"], w["x"]))
    line_groups = []

    for w in sorted_words:
        w_mid = w["y"] + w["h"] / 2.0
        target = None
        for group in line_groups:
            group_mid = group["y0"] + group["h"] / 2.0
            if abs(w_mid - group_mid) < max(14, w["h"] * 0.55):
                target = group
                break

        if target:
            target["words"].append(w)
            min_y = min(target["y0"], w["y"])
            max_y = max(target["y0"] + target["h"], w["y"] + w["h"])
            target["y0"] = min_y
            target["h"] = max_y - min_y
        else:
            line_groups.append({
                "y0": w["y"],
                "h": w["h"],
                "words": [w]
            })

    result = []
    for group in line_groups:
        line_words = sorted(group["words"], key=lambda w: w["x"])

        # Split words into separate sub-lines if separated by horizontal column gaps (> 35px or 1.2x font height)
        sub_clusters = []
        curr_cluster = []
        for w in line_words:
            if not curr_cluster:
                curr_cluster.append(w)
            else:
                prev = curr_cluster[-1]
                gap = w["x"] - (prev["x"] + prev["w"])
                max_word_gap = max(35, w["h"] * 1.2)
                if gap > max_word_gap:
                    sub_clusters.append(curr_cluster)
                    curr_cluster = [w]
                else:
                    curr_cluster.append(w)
        if curr_cluster:
            sub_clusters.append(curr_cluster)

        for cluster in sub_clusters:
            full_text = ""
            for i, w in enumerate(cluster):
                if i > 0:
                    prev_w = cluster[i - 1]
                    prev_text = prev_w["text"]
                    curr_text = w["text"]
                    gap_x = w["x"] - (prev_w["x"] + prev_w["w"])
                    if gap_x < max(20, w["h"] * 0.8) and (prev_text.endswith(",") or prev_text.endswith(".")) and curr_text[0].isdigit():
                        full_text += curr_text
                    else:
                        full_text += " " + curr_text
                else:
                    full_text = w["text"]

            x0 = min(w["x"] for w in cluster)
            result.append({
                "text": full_text,
                "x": x0,
                "y0": group["y0"],
                "words": cluster
            })

    result.sort(key=lambda l: l["y0"])
    return result


def _is_number_word(text: str) -> bool:
    return bool(NUMBER_TOKEN_RE.match(text)) and any(c.isdigit() for c in text)


def _detect_post_type(lines: list[dict]) -> str:
    """Check the top header area of the screenshot to determine if it is a Reel or Post."""
    header_lines = lines[:10]
    header_text = " ".join(l["text"].lower() for l in header_lines)

    if "reel" in header_text or "carrete" in header_text:
        return "reel"
    if "post" in header_text or "publicación" in header_text or "publicacao" in header_text or "impresiones" in header_text or "impressions" in header_text:
        return "post"

    full_text = " ".join(l["text"].lower() for l in lines)
    if "average watch time" in full_text or "watch time" in full_text or "reproducción" in full_text or "reprodução" in full_text or "watch" in full_text:
        return "reel"
    if "profile visits" in full_text or "visitas al perfil" in full_text or "visitas ao perfil" in full_text or "visites du profil" in full_text:
        return "post"

    return "post"


def _extract_icon_row(words: list[dict], overview_y: int | None, scale: float) -> dict:
    """Find the 5 engagement numbers sitting just above the anchor tab,
    ordered left-to-right, and map them by position to likes/comments/shares/sends/saves."""
    if overview_y is not None:
        candidates = [w for w in words if w["y"] < overview_y and _is_number_word(w["text"])]
    else:
        candidates = [w for w in words if _is_number_word(w["text"])]

    if not candidates:
        return {}

    candidates.sort(key=lambda w: -w["y"])
    band_y = candidates[0]["y"]
    band_tolerance = 40 * scale
    band = [w for w in candidates if abs(w["y"] - band_y) < band_tolerance]
    band.sort(key=lambda w: w["x"])

    result = {}
    for key, w in zip(ICON_ORDER, band[:5]):
        result[key] = w["text"]
    return result


def _closest_number_near_label(lines: list[dict], label_line: dict, img_h: int, img_w: int, exclude_values: set, is_time: bool = False) -> str | None:
    """Find the number line that best matches this label: strictly below the label and within the same grid column (dx <= img_w * 0.22)."""
    best_val, best_score = None, None
    max_gap = img_h * 0.18
    max_dx = img_w * 0.22  # Strict 22% image width column bound

    regex = TIME_SEARCH_RE if is_time else NUMBER_SEARCH_RE

    for line in lines:
        if line == label_line:
            continue

        # Must be located below the card label line (dy_down >= -10)
        dy_down = line["y0"] - label_line["y0"]
        if dy_down < -10 or dy_down > max_gap:
            continue

        dx = abs(line["x"] - label_line["x"])
        if dx > max_dx:
            continue

        m = regex.search(line["text"])
        if not m:
            continue
        val = m.group().strip()
        val_clean = re.sub(r"\s+", "", val).rstrip(".,") if not is_time else val.strip()
        if not val_clean or val_clean in exclude_values:
            continue

        score = dx + dy_down * 0.5
        if best_score is None or score < best_score:
            best_score = score
            best_val = val_clean
    return best_val


def _extract_summary_grid(lines: list[dict], img_h: int, img_w: int, icon_values: dict, post_type: str) -> dict:
    result = {}
    used_values = set(v for v in icon_values.values() if v)
    label_keywords = LABEL_KEYWORDS_REEL if post_type == "reel" else LABEL_KEYWORDS_POST

    for field, keywords in label_keywords.items():
        for line in lines:
            norm = line["text"].lower()
            if any(kw in norm for kw in keywords):
                is_time = (field == "average_watch_time")
                val = _closest_number_near_label(lines, line, img_h, img_w, exclude_values=used_values, is_time=is_time)
                if val:
                    result[field] = val
                    used_values.add(val)
                break
    return result


def _crop_post_thumbnail(image_rgb: np.ndarray, overview_y: int | None, img_h: int, img_w: int) -> str | None:
    """Detect and crop the exact post picture/video content sitting above the engagement icons."""
    try:
        y1 = int(img_h * 0.135)

        if overview_y and overview_y > img_h * 0.25:
            y2 = int(overview_y - int(img_h * 0.07))
        else:
            y2 = int(img_h * 0.425)

        if y2 <= y1 + 50:
            return None

        x1 = int(img_w * 0.20)
        x2 = int(img_w * 0.80)

        roi = image_rgb[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[0], roi.shape[1]

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect = None
        max_area = 0
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area > max_area and bw > roi_w * 0.35 and bh > roi_h * 0.30:
                max_area = area
                best_rect = (bx, by, bw, bh)

        if best_rect:
            bx, by, bw, bh = best_rect
            by = max(0, by - 2)
            bh = min(roi_h - by, bh + 4)
            bx = max(0, bx - 2)
            bw = min(roi_w - bx, bw + 4)
            crop = roi[by:by+bh, bx:bx+bw]
        else:
            crop = roi[int(roi_h * 0.05):int(roi_h * 0.95), int(roi_w * 0.10):int(roi_w * 0.90)]

        # Trim away any outer white border/padding pixels
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, non_white_mask = cv2.threshold(crop_gray, 240, 255, cv2.THRESH_BINARY_INV)
        non_white_pts = cv2.findNonZero(non_white_mask)
        if non_white_pts is not None:
            tx, ty, tw, th = cv2.boundingRect(non_white_pts)
            if tw > crop.shape[1] * 0.5 and th > crop.shape[0] * 0.5:
                crop = crop[ty:ty+th, tx:tx+tw]

        pil_crop = Image.fromarray(crop)
        pil_crop.thumbnail((600, 600), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        pil_crop.save(buffer, format="JPEG", quality=95)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception:
        return None


def extract_fields_from_image(image_bytes: bytes) -> dict:
    processed, scale = _preprocess(image_bytes)
    img_h, img_w = processed.shape[0], processed.shape[1]

    pil_orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_rgb = np.array(pil_orig)

    words = _words(processed)
    lines = _lines_from_words(words)
    post_type = _detect_post_type(lines)

    overview_y = None
    for line in lines:
        line_norm = line["text"].lower()
        if any(anchor in line_norm for anchor in ANCHOR_KEYWORDS):
            overview_y = line["y0"]
            break

    found = {k: None for k in FIELD_KEYS}
    found["post_type"] = post_type
    found["post_thumbnail"] = _crop_post_thumbnail(orig_rgb, overview_y, orig_rgb.shape[0], orig_rgb.shape[1])

    icon_row = _extract_icon_row(words, overview_y, scale)
    found.update(icon_row)
    found.update(_extract_summary_grid(lines, img_h, img_w, icon_row, post_type))
    return found


def extract_fields_from_images(images: list[bytes]) -> dict:
    results = []
    merged = {k: None for k in FIELD_KEYS}
    for image_bytes in images:
        try:
            partial = extract_fields_from_image(image_bytes)
            results.append(partial)
        except Exception:
            partial = {}
        for k, val in partial.items():
            if merged.get(k) is None and val:
                merged[k] = val

    merged["posts"] = results
    return merged

