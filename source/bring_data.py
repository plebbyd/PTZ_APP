import os
import logging
import math
import time
import datetime
from PIL import Image
import io
from pathlib import Path
from source.object_detector import get_label_from_image_and_object
from source.mcp_client import MCPClient
import json
from waggle.plugin import Plugin

logger = logging.getLogger(__name__)

try:
    tmp_dir = Path("/imgs")
    tmp_dir.mkdir(exist_ok=True, mode=0o777)
except OSError:
    logger.warning("Could not create temporary image directory.")
    tmp_dir = Path("./imgs")
    tmp_dir.mkdir(exist_ok=True)


def get_image_from_ptz_position(
    mcp_client: MCPClient, object_list: list, pan: float, tilt: float, zoom: float, detector, plugin: Plugin, event_id: str
) -> tuple:
    """
    Moves the camera, gets an image, and runs detection, including PTZ coordinates in metadata.
    """
    if not mcp_client.move_absolute(pan, tilt, zoom):
        logger.error(f"Failed to move camera to P={pan}, T={tilt}, Z={zoom}")
        return None, None
    time.sleep(2)

    image_bytes = mcp_client.take_snapshot()
    if not image_bytes:
        logger.error("Failed to get snapshot from MCP server.")
        return None, None
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Failed to open image from bytes: {e}")
        return None, None

    detections, raw_results = get_label_from_image_and_object(image, object_list, detector)

    current_pos = {"pan": pan, "tilt": tilt, "zoom": zoom}
    
    if plugin and raw_results:
        plugin.publish("ptz.florence.raw_output", json.dumps(raw_results), meta={"event_id": event_id, "ptz": current_pos})
    
    if not detections:
        return image, None
    
    best_detection = min(detections, key=lambda x: x['reward'])
    LABEL = {
        'bbox': best_detection['bbox'],
        'label': best_detection['label'],
        'reward': best_detection['reward'],
    }
    
    if plugin:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"initial_{timestamp}.jpg"
        temp_path = tmp_dir / filename
        try:
            image.save(temp_path)
            plugin.upload_file(str(temp_path), meta={"event_id": event_id, "view": "before", "ptz": current_pos})
            os.remove(temp_path)
        except Exception as e:
            logger.error(f"Error publishing 'before' image: {e}")

    return image, LABEL


def center_and_maximize_object(
    mcp_client: MCPClient, args, bbox, image, reward, label, plugin: Plugin, event_id: str
):
    """
    Calculates movement to center and zoom, then publishes the final image with its PTZ coordinates.
    """
    x1, y1, x2, y2 = bbox
    image_width, image_height = image.size

    pos = mcp_client.get_position()
    if not pos:
        logger.error("Could not get current camera position to center object.")
        return
    
    _, _, current_zoom = pos
    
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    
    diff_x = image_center_x - bbox_center_x
    diff_y = image_center_y - bbox_center_y

    current_h_fov, current_v_fov = get_fov_from_zoom(current_zoom)
    
    pan_offset = -(diff_x / image_width) * current_h_fov
    tilt_offset = -(diff_y / image_height) * current_v_fov

    bbox_width = x2 - x1
    target_width = image_width * 0.8 
    zoom_ratio = target_width / bbox_width 
    new_zoom = current_zoom * zoom_ratio
    new_zoom = max(1, min(40, new_zoom))
    zoom_offset = new_zoom - current_zoom

    logger.info(f"Calculated relative move: Pan={pan_offset:.2f}, Tilt={tilt_offset:.2f}, Zoom={zoom_offset:.2f}")

    if not mcp_client.move_relative(pan=pan_offset, tilt=tilt_offset, zoom=zoom_offset):
        logger.error("MCP client failed to perform relative move.")
        return
    time.sleep(3)

    final_image_bytes = mcp_client.take_snapshot()
    if not final_image_bytes:
        logger.error("Failed to take final snapshot after centering.")
        return

    final_pos_tuple = mcp_client.get_position()
    final_pos_dict = {"pan": final_pos_tuple[0], "tilt": final_pos_tuple[1], "zoom": final_pos_tuple[2]} if final_pos_tuple else {}

    if plugin:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        confidence = 1 - reward
        filename = f"zoomed_{label}_conf{confidence:.2f}_{timestamp}.jpg"
        
        try:
            temp_path = tmp_dir / filename
            with open(temp_path, "wb") as f:
                f.write(final_image_bytes)
            
            logger.info(f"Publishing final 'after' image: {temp_path}")
            plugin.upload_file(str(temp_path), meta={"event_id": event_id, "view": "after", "ptz": final_pos_dict})
            os.remove(temp_path)
        except Exception as e:
            logger.error(f"Error saving or publishing final image: {e}")


def get_fov_from_zoom(zoom_level):
    min_focal_length, max_focal_length = 4.25, 170
    h_wide, h_tele = 65.66, 1.88
    v_wide, v_tele = 39.40, 1.09
    min_zoom, max_zoom = 1, 40

    zoom_level = max(min_zoom, min(max_zoom, zoom_level))
    
    h_fov = h_wide - ((zoom_level - min_zoom) / (max_zoom - min_zoom)) * (h_wide - h_tele)
    v_fov = v_wide - ((zoom_level - min_zoom) / (max_zoom - min_zoom)) * (v_wide - v_tele)

    return h_fov, v_fov
