"""Make the FRONT and BACK AprilTags for the sumo robot, and remember which is which.

Writes:
    tags/front_tag.png, tags/back_tag.png   one tag each, labeled
    tags/robot_tags_print.png               both on one page, at true size (300 dpi)
    robot_tags.json                         IDs, sizes, and where each tag sits on the robot

Print robot_tags_print.png at 100% scale ("actual size", not "fit to page") and
check the black square measures --size-cm. Stick the FRONT tag near the front of
the robot (the color sensor end) and the BACK tag near the back, both on the
robot's center line, each with its arrow pointing toward the robot's front.

Measure from the robot's center (midway between the wheels, front to back):
    --front-offset-cm  how far ahead of center the FRONT tag's center is
    --back-offset-cm   how far behind center the BACK tag's center is
    --length-cm / --width-cm  the robot's footprint (used for "how much of us is out")
    --front-height-cm / --back-height-cm
                       how high each tag is above the floor (the camera looks from the
                       side, so raised tags look shifted; this corrects it)

Already printed and stuck on? Change only the measurements with --update, which
keeps the tag IDs and size and doesn't touch the images:
    python HW5/make_tags.py --update --front-height-cm 7.5 --back-height-cm 4.5

Usage:
    python HW5/make_tags.py
    python HW5/make_tags.py --front 1 --back 2 --size-cm 5 --front-offset-cm 5 --back-offset-cm 5 --length-cm 16 --width-cm 12
"""

import argparse
import json

import cv2
import cv2.aruco as aruco
import numpy as np

import config

DPI = 300
FAMILY = "tag36h11"
DICTIONARY = aruco.DICT_APRILTAG_36h11


def cm_to_px(cm):
    return int(round(cm / 2.54 * DPI))


def labeled_tag(dictionary, tag_id, size_cm, label):
    """One tag at true size with a white quiet zone, an arrow toward the robot's front and a label."""
    side = cm_to_px(size_cm)
    tag = aruco.generateImageMarker(dictionary, tag_id, side)
    margin = max(cm_to_px(0.8), side // 4)          # white border the detector needs
    arrow_h = cm_to_px(1.2)
    text_h = cm_to_px(1.4)
    w = side + 2 * margin
    h = arrow_h + side + 2 * margin + text_h
    img = np.full((h, w), 255, np.uint8)
    img[arrow_h + margin:arrow_h + margin + side, margin:margin + side] = tag

    # Arrow above the tag: "this edge faces the robot's front".
    cx = w // 2
    tip, base = arrow_h // 6, arrow_h - arrow_h // 6
    cv2.arrowedLine(img, (cx, base + margin // 2), (cx, tip + margin // 4), 0,
                    thickness=max(3, side // 40), tipLength=0.4)

    thick = max(2, side // 80)
    lines = ((f"{label}  (id {tag_id})", side / 260, int(h - text_h * 0.55)),
             ("arrow points to robot front", side / 520, int(h - text_h * 0.15)))
    for text, scale, y in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(img, text, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick, cv2.LINE_AA)
    return img


def update_measurements(parser, args):
    """Change only the measurements the user typed, keeping everything else in robot_tags.json."""
    try:
        tags = json.loads(config.ROBOT_TAGS_PATH.read_text())
    except (OSError, ValueError):
        raise SystemExit(f"{config.ROBOT_TAGS_PATH.name} not found; run make_tags.py without --update first.")
    keys = {"front_offset_cm": "front_offset_cm", "back_offset_cm": "back_offset_cm",
            "length_cm": "robot_length_cm", "width_cm": "robot_width_cm",
            "front_height_cm": "front_tag_height_cm", "back_height_cm": "back_tag_height_cm"}
    changed = []
    for arg, key in keys.items():
        value = getattr(args, arg)
        if value != parser.get_default(arg) or key not in tags:
            tags[key] = value
            changed.append(f"{key}={value:g}")
    tags.pop("tag_height_cm", None)        # replaced by the per-tag heights
    config.ROBOT_TAGS_PATH.write_text(json.dumps(tags, indent=2))
    print(f"Updated {config.ROBOT_TAGS_PATH.name}: {', '.join(changed) or 'nothing changed'}")
    print(f"Tags unchanged: FRONT = id {tags['front_id']}, BACK = id {tags['back_id']}, {tags['tag_size_cm']:g} cm.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--front", type=int, default=1, help="tag ID for the FRONT of the robot (default 1)")
    p.add_argument("--back", type=int, default=2, help="tag ID for the BACK of the robot (default 2)")
    p.add_argument("--size-cm", type=float, default=5.0, help="black square side length (default 5 cm)")
    p.add_argument("--front-offset-cm", type=float, default=5.0, help="FRONT tag center, cm ahead of robot center")
    p.add_argument("--back-offset-cm", type=float, default=5.0, help="BACK tag center, cm behind robot center")
    p.add_argument("--length-cm", type=float, default=16.0, help="robot length, front to back")
    p.add_argument("--width-cm", type=float, default=12.0, help="robot width, side to side")
    p.add_argument("--front-height-cm", type=float, default=8.0, help="FRONT tag height above the floor")
    p.add_argument("--back-height-cm", type=float, default=8.0, help="BACK tag height above the floor")
    p.add_argument("--update", action="store_true",
                   help="only update the measurements given on the command line in robot_tags.json "
                        "(keeps the printed tags' IDs and size; no new images)")
    args = p.parse_args()
    if args.update:
        update_measurements(p, args)
        return
    if args.front == args.back:
        raise SystemExit("--front and --back must be different tag IDs.")

    dictionary = aruco.getPredefinedDictionary(DICTIONARY)
    out_dir = config.HERE / "tags"
    out_dir.mkdir(exist_ok=True)

    front = labeled_tag(dictionary, args.front, args.size_cm, "FRONT")
    back = labeled_tag(dictionary, args.back, args.size_cm, "BACK")
    cv2.imwrite(str(out_dir / "front_tag.png"), front)
    cv2.imwrite(str(out_dir / "back_tag.png"), back)

    # Letter page (8.5 x 11 in) with both tags side by side.
    page = np.full((int(11 * DPI), int(8.5 * DPI)), 255, np.uint8)
    gap = cm_to_px(1.5)
    total_w = front.shape[1] + gap + back.shape[1]
    x0, y0 = (page.shape[1] - total_w) // 2, cm_to_px(2.5)
    page[y0:y0 + front.shape[0], x0:x0 + front.shape[1]] = front
    x1 = x0 + front.shape[1] + gap
    page[y0:y0 + back.shape[0], x1:x1 + back.shape[1]] = back
    note = f"Print at 100% scale. Each black square should measure {args.size_cm:g} cm."
    cv2.putText(page, note, (x0, y0 + front.shape[0] + cm_to_px(1.2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "robot_tags_print.png"), page, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    tags = {
        "family": FAMILY,
        "front_id": args.front,
        "back_id": args.back,
        "tag_size_cm": args.size_cm,
        "front_offset_cm": args.front_offset_cm,
        "back_offset_cm": args.back_offset_cm,
        "robot_length_cm": args.length_cm,
        "robot_width_cm": args.width_cm,
        "front_tag_height_cm": args.front_height_cm,
        "back_tag_height_cm": args.back_height_cm,
    }
    config.ROBOT_TAGS_PATH.write_text(json.dumps(tags, indent=2))

    # Check the printed images decode back to the right IDs.
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(page)
    found = sorted(int(i) for i in ids.ravel()) if ids is not None else []
    print(f"FRONT = tag {args.front}, BACK = tag {args.back} ({args.size_cm:g} cm). Saved {config.ROBOT_TAGS_PATH.name}.")
    print(f"Images in {out_dir}. Self-check found tag IDs {found} on the print page.")
    if found != sorted([args.front, args.back]):
        print("Warning: the self-check did not find both tags. Try a larger --size-cm.")


if __name__ == "__main__":
    main()
