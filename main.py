import os
import sys
import argparse
import time
from PIL import Image
from source.bring_data import (
    center_and_maximize_object,
    center_and_maximize_objects_absolute,
    get_image_from_ptz_position,
    get_image_from_ptz_position_multiboxes,
    publish_images,
)
from source.object_detector import DetectorFactory
import logging


def get_argparser():
    parser = argparse.ArgumentParser("PTZ APP")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug level logging.",
    )
    parser.add_argument(
        "-ki",
        "--keepimages",
        action="store_true",
        help="Keep collected images in persistent folder for later use",
    )
    parser.add_argument(
        "-it",
        "--iterations",
        help="An integer with the number of iterations (PTZ rounds) to be run (default=5).",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-obj",
        "--objects",
        help="Objects to capture with the camera (comma-separated, e.g., 'person,car,dog')",
        type=str,
        default="person",
    )
    parser.add_argument(
        "-un",
        "--username",
        help="The username of the PTZ camera.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-pw",
        "--password",
        help="The password of the PTZ camera.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-ip", "--cameraip", help="The ip of the PTZ camera.", type=str, default=""
    )
    parser.add_argument(
        "-ps", "--panstep", help="The step of pan in degrees.", type=int, default=15
    )
    parser.add_argument(
        "-tv", "--tilt", help="The tilt value in degrees.", type=int, default=0
    )
    parser.add_argument("-zm", "--zoom", help="The zoom value.", type=int, default=1)
    parser.add_argument(
        "-m",
        "--model",
        help="Model to use (e.g., 'yolo11n', 'Florence-base')",
        type=str,
        default="yolo11n",
    )
    parser.add_argument(
        "-id",
        "--iterdelay",
        help="Delay in seconds between iterations (default=0.0)",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "-conf",
        "--confidence",
        help="Confidence threshold for detections (0-1, default=0.1)",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-multiple",
        "--multiple",
        help="Save multiple images for each discrete PTZ position (default=False)",
        action="store_true",
    )
    parser.add_argument(
        "--debug-detections",
        action="store_true",
        help="Publish images with detection boxes drawn for debugging",
    )

    return parser


def look_for_object(args):
    objects = [obj.strip().lower() for obj in args.objects.split(",")]
    pans = [angle for angle in range(0, 360, args.panstep)]
    tilts = [args.tilt for _ in range(len(pans))]
    zooms = [args.zoom for _ in range(len(pans))]

    try:
        detector = DetectorFactory.create_detector(args.model, args.objects)
    except ValueError as e:
        print(f"Error creating detector: {str(e)}")
        sys.exit(1)

    for iteration in range(args.iterations):
        iteration_start_time = time.time()

        for idx, (pan, tilt, zoom) in enumerate(zip(pans, tilts, zooms)):
            # Create increment ID based on initial PTZ position
            increment_id = f"scan_{pan:03d}_{tilt:03d}_{zoom:02d}"
            print(f"Trying PTZ: {pan} {tilt} {zoom} (ID: {increment_id})")

            if not args.multiple:
                image_path, detection = get_image_from_ptz_position(
                    args, objects, pan, tilt, zoom, detector, None, args.debug_detections, increment_id
                )
                if detection is None or detection["reward"] > (1 - args.confidence):
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)
                    continue

                label = detection["label"]
                bbox = detection["bbox"]
                reward = detection["reward"]
                confidence = 1 - reward

                print(f"Following {label} object (confidence: {confidence:.2f})")

                image = Image.open(image_path)
                center_and_maximize_object(args, bbox, image, reward, label, increment_id)
            else:
                # get multiple images for each detection
                image_path, detections = get_image_from_ptz_position_multiboxes(
                    args, objects, pan, tilt, zoom, detector, None, args.debug_detections, increment_id
                )
                
                if not detections:
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)
                    continue
                
                image = Image.open(image_path)
                center_and_maximize_objects_absolute(args, detections, image, increment_id)
                


            if os.path.exists(image_path):
                os.remove(image_path)

            # Publish images after each increment
            publish_images()

        iteration_time = time.time() - iteration_start_time
        if args.iterdelay > 0:
            remaining_delay = max(0, args.iterdelay - iteration_time)
            if remaining_delay > 0:
                print(f"Waiting {remaining_delay:.2f} seconds before next iteration...")
                time.sleep(remaining_delay)


def main():
    args = get_argparser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    look_for_object(args)


if __name__ == "__main__":
    main()
