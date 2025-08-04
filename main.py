import os
os.environ['HF_HOME'] = '/hf_cache'

import sys
import argparse
import time
import uuid
from source.bring_data import (
    center_and_maximize_object,
    get_image_from_ptz_position,
)
from source.object_detector import DetectorFactory
from source.mcp_client import MCPClient
import logging
from waggle.plugin import Plugin

def get_argparser():
    parser = argparse.ArgumentParser("PTZ Object Detection Client")
    parser.add_argument(
        "--mcp_server_url", type=str, default="http://localhost:8000",
        help="URL of the local Camera Hardware MCP Server."
    )
    parser.add_argument(
        "--boredom_tilt_step", type=float, default=5.0,
        help="How many degrees to tilt down if a full scan finds nothing."
    )
    parser.add_argument(
        "--prompt_context", type=str, default="In this outdoor nature scene",
        help="String providing context for the AI model's prompt."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug level logging."
    )
    parser.add_argument(
        "-it", "--iterations", type=int, default=10,
        help="Number of full 360-degree scans to run."
    )
    parser.add_argument(
        "-obj", "--objects", type=str, default="a bird;a cat;a dog;a horse;a sheep;a cow;a bear",
        help="Target objects or phrases, separated by semicolons (;)."
    )
    parser.add_argument(
        "-ps", "--panstep", type=int, default=15, help="The step of pan in degrees."
    )
    parser.add_argument(
        "-tv", "--tilt", type=int, default=0, help="The initial tilt value in degrees."
    )
    parser.add_argument("-zm", "--zoom", type=int, default=1, help="The initial zoom value.")
    parser.add_argument(
        "-m", "--model", type=str, default="Florence-base",
        help="Model to use (e.g., 'yolo11n', 'Florence-base')"
    )
    parser.add_argument(
        "-id", "--iterdelay", type=float, default=60.0,
        help="Delay in seconds between scans."
    )
    parser.add_argument(
        "-conf", "--confidence", type=float, default=0.1,
        help="Confidence threshold for detections (0-1)."
    )
    return parser


def look_for_object(args, plugin):
    mcp_client = MCPClient(args.mcp_server_url)
    objects = [phrase.strip() for phrase in args.objects.split(";")]
    
    try:
        detector = DetectorFactory.create_detector(args.model, objects, args.prompt_context)
    except Exception as e:
        print(f"Error creating detector: {e}")
        sys.exit(1)

    current_tilt = args.tilt

    for iteration in range(args.iterations):
        iteration_start_time = time.time()
        found_object_in_scan = False

        pans = [angle for angle in range(0, 360, args.panstep)]
        
        print(f"\n--- Starting Scan #{iteration + 1} at Tilt: {current_tilt} ---")

        for pan in pans:
            print(f"Trying PTZ: Pan={pan}, Tilt={current_tilt}, Zoom={args.zoom}")
            
            event_id = str(uuid.uuid4())

            image, detection = get_image_from_ptz_position(
                mcp_client, objects, pan, current_tilt, args.zoom, detector, plugin, event_id
            )

            if image is None:
                print("Could not retrieve image from camera server. Skipping position.")
                continue

            if detection and detection["reward"] <= (1 - args.confidence):
                found_object_in_scan = True
                label = detection["label"]
                bbox = detection["bbox"]
                reward = detection["reward"]
                confidence = 1 - reward

                print(f"Following {label} object (confidence: {confidence:.2f})")
                center_and_maximize_object(mcp_client, args, bbox, image, reward, label, plugin, event_id)

        if not found_object_in_scan:
            print("\nScan complete. No objects found. Adjusting tilt for next scan.")
            current_tilt -= args.boredom_tilt_step
            if current_tilt < -20:
                print("Tilt limit reached. Resetting to initial tilt.")
                current_tilt = args.tilt

        iteration_time = time.time() - iteration_start_time
        if args.iterdelay > 0:
            remaining_delay = max(0, args.iterdelay - iteration_time)
            if remaining_delay > 0:
                print(f"Waiting {remaining_delay:.2f} seconds before next scan...")
                time.sleep(remaining_delay)


def main():
    args = get_argparser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    with Plugin() as plugin:
        look_for_object(args, plugin)


if __name__ == "__main__":
    main()
