import asyncio
import logging
import argparse
import base64
import sys
from typing import Dict

from mcp.server.fastmcp import FastMCP
from source.sunapi_control import CameraControl

# --- Globals ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
mcp = FastMCP("CameraHardwareServer")
camera_controller: CameraControl = None

# --- MCP Tools ---

@mcp.tool()
async def get_position() -> Dict[str, float]:
    """
    Retrieves the current Pan, Tilt, and Zoom position of the camera.
    Returns a dictionary with 'pan', 'tilt', and 'zoom' values.
    """
    if not camera_controller:
        return {"error": "Camera controller not initialized."}
    try:
        pan, tilt, zoom = camera_controller.requesting_cameras_position_information()
        logger.info(f"Position queried: Pan={pan}, Tilt={tilt}, Zoom={zoom}")
        return {"pan": pan, "tilt": tilt, "zoom": zoom}
    except Exception as e:
        logger.error(f"Failed to get camera position: {e}")
        return {"error": str(e)}

@mcp.tool()
async def move_absolute(pan: float, tilt: float, zoom: float) -> Dict[str, str]:
    """
    Moves the camera to an absolute Pan, Tilt, and Zoom position.
    """
    if not camera_controller:
        return {"error": "Camera controller not initialized."}
    try:
        logger.info(f"Moving to absolute position: Pan={pan}, Tilt={tilt}, Zoom={zoom}")
        camera_controller.absolute_control(pan=pan, tilt=tilt, zoom=zoom)
        return {"status": "success", "message": f"Move to P={pan}, T={tilt}, Z={zoom} initiated."}
    except Exception as e:
        logger.error(f"Failed to execute absolute move: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def move_relative(pan: float, tilt: float, zoom: float) -> Dict[str, str]:
    """
    Moves the camera by a relative Pan, Tilt, and Zoom amount.
    """
    if not camera_controller:
        return {"error": "Camera controller not initialized."}
    try:
        logger.info(f"Moving by relative amount: Pan={pan}, Tilt={tilt}, Zoom={zoom}")
        camera_controller.relative_control(pan=pan, tilt=tilt, zoom=zoom)
        return {"status": "success", "message": f"Relative move of P={pan}, T={tilt}, Z={zoom} initiated."}
    except Exception as e:
        logger.error(f"Failed to execute relative move: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def take_snapshot() -> Dict[str, str]:
    """
    Captures a snapshot from the camera and returns it as a base64 encoded string.
    """
    if not camera_controller:
        return {"error": "Camera controller not initialized."}
    try:
        logger.info("Taking snapshot.")
        image_bytes = camera_controller.snap_shot()
        if image_bytes:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            return {"status": "success", "image_base64": base64_image}
        else:
            logger.error("Failed to capture image bytes from camera.")
            return {"status": "error", "message": "Failed to capture image bytes."}
    except Exception as e:
        logger.error(f"Failed to take snapshot: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def stop_movement() -> Dict[str, str]:
    """
    Stops all ongoing Pan, Tilt, and Zoom movements.
    """
    if not camera_controller:
        return {"error": "Camera controller not initialized."}
    try:
        logger.info("Stopping all movement.")
        camera_controller.stop_control()
        return {"status": "success", "message": "Stop command sent."}
    except Exception as e:
        logger.error(f"Failed to stop movement: {e}")
        return {"status": "error", "message": str(e)}

# --- Startup ---

def get_argparser():
    parser = argparse.ArgumentParser(description="Local MCP Hardware Server for PTZ Camera")
    parser.add_argument("-un", "--username", help="The username for the PTZ camera.", type=str, required=True)
    parser.add_argument("-pw", "--password", help="The password for the PTZ camera.", type=str, required=True)
    parser.add_argument("-ip", "--cameraip", help="The IP address of the PTZ camera.", type=str, required=True)
    return parser

async def print_registered_tools():
    tools = await mcp.list_tools()
    print("="*50)
    print("🚀 Local Camera Hardware MCP Server Starting...")
    print("="*50)
    print(f"✅ Registered tools ({len(tools)}):")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")
    print(f"\n🌐 Server will be available at http://localhost:8000/mcp")
    print("="*50 + "\n")

def main():
    global camera_controller
    parser = get_argparser()
    args = parser.parse_args()

    try:
        camera_controller = CameraControl(ip=args.cameraip, user=args.username, password=args.password)
        logger.info(f"Successfully initialized camera controller for IP: {args.cameraip}")
    except Exception as e:
        logger.error(f"Fatal error initializing camera controller: {e}")
        sys.exit(1)

    try:
        asyncio.run(print_registered_tools())
        logger.info(f"Starting MCP server on port 8000...")
        mcp.run()
    except KeyboardInterrupt:
        print("\n Server stopped by user.")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
