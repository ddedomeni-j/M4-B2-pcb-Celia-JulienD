from pathlib import Path
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def load_binary_image(image_path: Path):
    """Load image and convert it to a clean binary mask.

    Foreground = white PCB traces / pads
    Background = black
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Optional denoising
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Otsu threshold
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Ensure foreground is white and background is black
    if np.mean(binary == 255) > 0.5:
        binary = cv2.bitwise_not(binary)

    return img, binary


def line_angle_deg(x1, y1, x2, y2):
    """Return line angle in degrees between 0 and 180."""
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angle = angle % 180
    return angle


def same_lengths(lengths, tolerance=0.03):

    if len(lengths) <= 1:
        return True

    max_length = max(lengths)

    return all(
        length >= max_length * (1 - tolerance)
        for length in lengths
    )


def check_line_lengths(lines, tolerance=0.15):
    """
    Vérifie si les lignes verticales ont des longueurs similaires
    et si les lignes horizontales ont des longueurs similaires.

    tolerance = 0.15 => 15% d'écart accepté.

    Returns: bool
    """

    vertical_lengths = []
    horizontal_lengths = []

    for line in lines:
        x1, y1, x2, y2 = map(int, line["points"])

        # length = np.hypot(
        #     x2 - x1,
        #     y2 - y1
        # )

        if line["orientation"] == "vertical":
            vertical_lengths.append(abs(y2 - y1))

        elif line["orientation"] == "horizontal":
            horizontal_lengths.append(abs(x2 - x1))

    return same_lengths(vertical_lengths, tolerance) and same_lengths(horizontal_lengths, tolerance)


def classify_orientation(angle, tolerance=10):
    """Classify line as horizontal, vertical or oblique."""
    if angle < tolerance or angle > 180 - tolerance:
        return "horizontal"

    if abs(angle - 90) < tolerance:
        return "vertical"

    return "oblique"


def remove_duplicate_lines(lines, tolerance=3):
    best_vertical = {}
    best_horizontal = {}
    oblique_lines = []

    for line in lines:
        orientation = line["orientation"]

        x1, y1, x2, y2 = map(int, line["points"])

        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        if orientation == "vertical":
            x = round((x1 + x2) / 2 / tolerance) * tolerance

            if (
                x not in best_vertical
                or length > best_vertical[x][0]
            ):
                best_vertical[x] = (length, line)

        elif orientation == "horizontal":
            y = round((y1 + y2) / 2 / tolerance) * tolerance

            if (
                y not in best_horizontal
                or length > best_horizontal[y][0]
            ):
                best_horizontal[y] = (length, line)

        else:
            # on garde toutes les obliques
            oblique_lines.append(line)

    return (
        [v[1] for v in best_vertical.values()]
        + [v[1] for v in best_horizontal.values()]
        + oblique_lines
    )


def detect_lines(binary, show=True):
    """Detect line segments using probabilistic Hough transform."""
    edges = cv2.Canny(binary, 50, 150)

    h, w = binary.shape

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=int(min(h, w) * 0.35),
        maxLineGap=5
    )

    detected = []

    if lines is None:
        return detected

    for line in lines:
        x1, y1, x2, y2 = line

        angle = line_angle_deg(x1, y1, x2, y2)
        orientation = classify_orientation(angle)

        detected.append({
            "points": (x1, y1, x2, y2),
            "angle": angle,
            "orientation": orientation,
        })

    detected = remove_duplicate_lines(detected)

    return detected


def cluster_positions(positions, tolerance=5):
    """Cluster close x/y positions into one physical line."""
    if not positions:
        return []

    positions = sorted(positions)
    clusters = [[positions[0]]]

    for pos in positions[1:]:
        if abs(pos - np.mean(clusters[-1])) <= tolerance:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])

    return [int(round(np.mean(c))) for c in clusters]


def count_lines(lines, merge_tolerance=5):
    vertical_positions = []
    horizontal_positions = []
    oblique_lines = []

    for line in lines:
        x1, y1, x2, y2 = line["points"]
        orientation = line["orientation"]

        if orientation == "vertical":
            vertical_positions.append((x1 + x2) / 2)

        elif orientation == "horizontal":
            horizontal_positions.append((y1 + y2) / 2)

        else:
            oblique_lines.append(line)

    vertical_clusters = cluster_positions(
        vertical_positions,
        tolerance=merge_tolerance
    )

    horizontal_clusters = cluster_positions(
        horizontal_positions,
        tolerance=merge_tolerance
    )

    return {
        "vertical_count": len(vertical_clusters),
        "horizontal_count": len(horizontal_clusters),
        "oblique_count": len(oblique_lines),
        "vertical_positions": vertical_clusters,
        "horizontal_positions": horizontal_clusters,
        "oblique_lines": oblique_lines,
    }


def detect_circular_blobs(binary, show=True):
    """
    Detect circular/compact white blobs in a binary PCB image.

    Args:
        binary: grayscale binary image, foreground white = 255.
        show: display debug visualization.

    Returns:
        List of dicts with x, y, r, area, circularity, fill_ratio.
    """

    # Ensure binary is 0/255
    _, bin_img = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    # Optional: slight opening to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    circles = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 10:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        (x, y), r = cv2.minEnclosingCircle(contour)

        x = int(round(x))
        y = int(round(y))
        r = int(round(r))

        if r < 2:
            continue

        circle_area = np.pi * r * r
        fill_ratio = area / circle_area if circle_area > 0 else 0

        # Filters to tune
        is_candidate = (
            0.45 <= circularity <= 1.25
            and 0.35 <= fill_ratio <= 1.10
            and 3 <= r <= 12
        )

        if not is_candidate:
            continue

        circles.append({
            "x": x,
            "y": y,
            "r": r,
            "area": float(area),
            "circularity": float(circularity),
            "fill_ratio": float(fill_ratio),
        })

        cv2.circle(vis, (x, y), r, (0, 255, 0), 1)
        cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        cv2.putText(
            vis,
            f"r={r}",
            (x + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 0),
            1
        )

    if show:
        # cv2.imshow("Circular blobs", cv2.resize(vis, None, fx=8, fy=8))
        # cv2.waitKey(10)


        plt.imshow(
            cv2.cvtColor(
                cv2.resize(vis, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_BGR2RGB
            )
        )

        plt.title("Circular blobs")
        plt.axis("off")
        plt.show()

    return circles


def line_coverage(binary, line, radius=2):
    """
    Measure line continuity for vertical or horizontal lines.

    For a vertical line:
        checks pixels from x-radius to x+radius for every y.

    For a horizontal line:
        checks pixels from y-radius to y+radius for every x.

    Args:
        binary: binary image, foreground white = 255.
        line: dict with "points" and "orientation".
        radius: number of pixels around the detected line.

    Returns:
        coverage between 0 and 1.
    """

    x1, y1, x2, y2 = line["points"]
    orientation = line["orientation"]

    h, w = binary.shape

    # Convert numpy int32 to regular Python int
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    hits = 0
    total = 0

    if orientation == "vertical":
        x = int(round((x1 + x2) / 2))

        y_start = min(y1, y2)
        y_end = max(y1, y2)

        for y in range(y_start, y_end + 1):

            x_min = max(0, x - radius)
            x_max = min(w, x + radius + 1)

            band = binary[y, x_min:x_max]

            if np.any(band > 0):
                hits += 1

            total += 1

    elif orientation == "horizontal":
        y = int(round((y1 + y2) / 2))

        x_start = min(x1, x2)
        x_end = max(x1, x2)

        for x in range(x_start, x_end + 1):

            y_min = max(0, y - radius)
            y_max = min(h, y + radius + 1)

            band = binary[y_min:y_max, x]

            if np.any(band > 0):
                hits += 1

            total += 1

    else:
        raise ValueError(
            "This function only handles vertical and horizontal lines."
        )

    if total == 0:
        return 0.0

    return hits / total


def visualize_connectivity(
    binary,
    corridor,
    track_in_corridor,
    labels,
    start_point,
    end_point,
    connected,
    scale=8
):
    """
    Visualize path connectivity checking.

    White: original foreground
    Green: corridor
    Red/blue: endpoints
    """

    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Corridor in green
    green_overlay = np.zeros_like(vis)
    green_overlay[corridor > 0] = (0, 255, 0)

    vis = cv2.addWeighted(vis, 0.8, green_overlay, 0.3, 0)

    # Track inside corridor in yellow
    vis[track_in_corridor > 0] = (0, 255, 255)

    sx, sy = start_point
    ex, ey = end_point

    cv2.circle(vis, (sx, sy), 3, (0, 0, 255), -1)
    cv2.circle(vis, (ex, ey), 3, (255, 0, 0), -1)

    text = "CONNECTED" if connected else "BROKEN"
    color = (0, 255, 0) if connected else (0, 0, 255)

    cv2.putText(
        vis,
        text,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1
    )

    # cv2.imshow(
    #     "Connectivity check",
    #     cv2.resize(
    #         vis,
    #         None,
    #         fx=scale,
    #         fy=scale,
    #         interpolation=cv2.INTER_NEAREST
    #     )
    # )
    # cv2.waitKey(10)

    plt.figure(figsize=(8, 8))

    plt.imshow(
        cv2.cvtColor(
            cv2.resize(vis, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Connectivity check")
    plt.axis("off")
    plt.show()


def find_nearest_label(labels, point, radius=4):
    """
    Find a non-zero connected-component label near a point.

    Args:
        labels: connected component label image.
        point: (x, y).
        radius: search radius around the point.

    Returns:
        component label or None.
    """

    x, y = point
    h, w = labels.shape

    x_min = max(0, x - radius)
    x_max = min(w, x + radius + 1)

    y_min = max(0, y - radius)
    y_max = min(h, y + radius + 1)

    roi = labels[y_min:y_max, x_min:x_max]

    labels_found = roi[roi > 0]

    if len(labels_found) == 0:
        return None

    # Return the most frequent non-zero label
    values, counts = np.unique(labels_found, return_counts=True)
    return int(values[np.argmax(counts)])


def check_line_connectivity(binary, line, radius=3, endpoint_radius=4, show=False):
    """
    Check whether a vertical or horizontal PCB track is connected from one end to the other.

    Args:
        binary: binary image, white foreground = 255.
        line: dict with:
            - "points": (x1, y1, x2, y2)
            - "orientation": "vertical" or "horizontal"
        radius: half-width of the corridor around the line.
        endpoint_radius: radius used to find white pixels near each endpoint.
        show: debug visualization.

    Returns:
        connected: bool
        info: dict with diagnostic values
    """

    x1, y1, x2, y2 = line["points"]
    orientation = line["orientation"]

    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    h, w = binary.shape

    # Ensure binary is 0/255
    _, bin_img = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    corridor = np.zeros_like(bin_img)

    if orientation == "vertical":
        x = int(round((x1 + x2) / 2))

        y_start = max(0, min(y1, y2))
        y_end = min(h - 1, max(y1, y2))

        x_min = max(0, x - radius)
        x_max = min(w - 1, x + radius)

        corridor[y_start:y_end + 1, x_min:x_max + 1] = 255

        start_point = (x, y_start)
        end_point = (x, y_end)

    elif orientation == "horizontal":
        y = int(round((y1 + y2) / 2))

        x_start = max(0, min(x1, x2))
        x_end = min(w - 1, max(x1, x2))

        y_min = max(0, y - radius)
        y_max = min(h - 1, y + radius)

        corridor[y_min:y_max + 1, x_start:x_end + 1] = 255

        start_point = (x_start, y)
        end_point = (x_end, y)

    else:
        raise ValueError("Only vertical and horizontal lines are handled here.")

    # Keep only foreground pixels inside the corridor
    track_in_corridor = cv2.bitwise_and(bin_img, corridor)

    # Calcul des longueurs de segment
    ys, xs = np.where(track_in_corridor > 0)

    if len(xs) == 0:
        connected_length = 0
    elif orientation == "vertical":
        connected_length = ys.max() - ys.min()
    elif orientation == "horizontal":
        connected_length = xs.max() - xs.min()


    # Connected components
    num_labels, labels = cv2.connectedComponents(track_in_corridor)

    start_label = find_nearest_label(
        labels,
        start_point,
        endpoint_radius
    )

    end_label = find_nearest_label(
        labels,
        end_point,
        endpoint_radius
    )

    connected = (
        start_label is not None
        and end_label is not None
        and start_label == end_label
        and start_label != 0
    )

    info = {
        "start_label": start_label,
        "end_label": end_label,
        "num_components": num_labels - 1,
        "start_point": start_point,
        "end_point": end_point,
        "length":connected_length
    }

    if show:
        visualize_connectivity(
            binary=bin_img,
            corridor=corridor,
            track_in_corridor=track_in_corridor,
            labels=labels,
            start_point=start_point,
            end_point=end_point,
            connected=connected
        )

    return connected, info


def make_expected_line(binary, line, margin=5):
    """
    Convert a Hough line segment into a full expected PCB line.

    For vertical lines:
        keep x, extend y from margin to image height - margin

    For horizontal lines:
        keep y, extend x from margin to image width - margin
    """

    h, w = binary.shape

    x1, y1, x2, y2 = line["points"]
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    orientation = line["orientation"]

    if orientation == "vertical":
        x = int(round((x1 + x2) / 2))

        return {
            "orientation": "vertical",
            "points": (
                x,
                margin,
                x,
                h - 1 - margin
            )
        }

    elif orientation == "horizontal":
        y = int(round((y1 + y2) / 2))

        return {
            "orientation": "horizontal",
            "points": (
                margin,
                y,
                w - 1 - margin,
                y
            )
        }

    else:
        return line


def is_plain_line(binary, line, threshold=0.75, show=False):
    # coverage = line_coverage(binary, line)
    # return coverage >= threshold, coverage

    expected_line = make_expected_line(binary, line, margin=5)
    connected, info = check_line_connectivity(binary,
                                            expected_line,
                                            radius=3,
                                            endpoint_radius=4,
                                            show=show)
    return connected, info


def circle_fill_ratio(binary, circle):
    """Measure foreground ratio inside detected circle."""
    x, y, r = circle["x"], circle["y"], circle["r"]

    mask = np.zeros_like(binary)
    cv2.circle(mask, (x, y), r, 255, -1)

    expected_pixels = mask > 0
    foreground_pixels = binary > 0

    if expected_pixels.sum() == 0:
        return 0.0

    fill_ratio = np.logical_and(expected_pixels, foreground_pixels).sum()
    fill_ratio /= expected_pixels.sum()

    return fill_ratio


def is_plain_circle(binary, circle, threshold=0.75):
    ratio = circle_fill_ratio(binary, circle)
    return ratio >= threshold, ratio

# ===============================



