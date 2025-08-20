#!/usr/bin/env python3
import asyncio
import logging
import sys
from typing import List

from mcp.server.fastmcp import FastMCP
from models import (
    SageConfig, PluginArguments, PluginSpec, SageJob, SelectorRequirements
)
from job_service import SageJobService

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP + JobService
mcp = FastMCP("PtzAppMCP")
sage_config = SageConfig()
job_service = SageJobService(sage_config)

# --- MCP Tool ---

@mcp.tool()
def submit_ptz_app_job(
    job_name: str,
    nodes: str,
    model: str = "yolo11n",
    iterations: int = 5,
    objects: str = "person",
    username: str = "camera",
    password: str = "0Bscura#",
    camera_ip: str = "130.202.23.153",
    pan_step: int = 15,
    tilt: int = 0,
    zoom: int = 1,
    confidence: float = 0.1,
    iter_delay: float = 60.0
) -> str:
    """
    Submits the PTZ detection job to specified SAGE nodes.
    """
    try:
        # Node list parsing
        node_list = [n.strip() for n in nodes.split(",") if n.strip()]
        if not node_list:
            raise ValueError("No valid nodes specified.")

        # Optional IP warning
        if not camera_ip.startswith("130.202."):
            logger.warning(f"⚠️ Using non-default camera IP: {camera_ip}")

        # Format arguments
        plugin_args_str = (
            f"model={model},iterations={iterations},objects={objects},"
            f"username={username},password={password},cameraip={camera_ip},"
            f"panstep={pan_step},tilt={tilt},zoom={zoom},"
            f"confidence={confidence},iterdelay={iter_delay}"
        )
        plugin_args = PluginArguments.from_string(plugin_args_str)

        # Define plugin spec
        plugin_spec = PluginSpec(
            name=job_name,
            image="plebbyd/ptzapp-yolo:0.1.14",
            args=plugin_args,
            selector=SelectorRequirements(gpu=True, camera=True)
        )

        # Build SAGE Job
        job = SageJob(
            name=job_name,
            nodes=node_list,
            plugins=[plugin_spec]
        )

        # Submit
        success, result = job_service.submit_job(job)
        return result

    except Exception as e:
        logger.error(f"❌ Failed to submit PTZ App job: {e}")
        return f"Error submitting job: {str(e)}"

# --- Startup ---

async def print_registered() -> None:
    tools = await mcp.list_tools()
    print("="*50)
    print("🚀 PTZ App MCP Server Starting...")
    print("="*50)
    print(f"✅ Registered tools ({len(tools)}):")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")
    print(f"\n🌐 Server: http://localhost:8000/mcp")
    print("="*50 + "\n")

def main() -> None:
    try:
        asyncio.run(print_registered())
        logger.info("Starting MCP server...")
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
