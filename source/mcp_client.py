import requests
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class MCPClient:
    """
    A client for interacting with the local Camera Hardware MCP Server.
    """
    def __init__(self, server_url: str):
        """
        Initializes the client.
        Args:
            server_url (str): The base URL of the MCP server (e.g., http://localhost:8000).
        """
        if not server_url.endswith('/'):
            server_url += '/'
        self.mcp_endpoint = f"{server_url}mcp"
        logger.info(f"MCP Client initialized for server: {self.mcp_endpoint}")

    def _make_request(self, tool_name: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Makes a POST request to a specific tool on the MCP server.
        Args:
            tool_name (str): The name of the tool to call.
            params (dict, optional): A dictionary of parameters for the tool. Defaults to None.
        Returns:
            dict or None: The JSON response from the server, or None if an error occurs.
        """
        payload = {
            "tool": tool_name,
            "params": params or {}
        }
        try:
            response = requests.post(self.mcp_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling tool '{tool_name}': {e}")
            return None

    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """
        Gets the current (pan, tilt, zoom) of the camera.
        Returns:
            A tuple (pan, tilt, zoom) or None on failure.
        """
        response = self._make_request("get_position")
        if response and response.get("pan") is not None:
            return response["pan"], response["tilt"], response["zoom"]
        logger.error(f"Failed to get position. Response: {response}")
        return None

    def move_absolute(self, pan: float, tilt: float, zoom: float) -> bool:
        """
        Moves the camera to an absolute position.
        Returns:
            True on success, False on failure.
        """
        params = {"pan": pan, "tilt": tilt, "zoom": zoom}
        response = self._make_request("move_absolute", params)
        return response and response.get("status") == "success"

    def move_relative(self, pan: float, tilt: float, zoom: float) -> bool:
        """
        Moves the camera by a relative amount.
        Returns:
            True on success, False on failure.
        """
        params = {"pan": pan, "tilt": tilt, "zoom": zoom}
        response = self._make_request("move_relative", params)
        return response and response.get("status") == "success"

    def take_snapshot(self) -> Optional[bytes]:
        """
        Requests a snapshot from the camera.
        Returns:
            The image data as bytes, or None on failure.
        """
        import base64
        response = self._make_request("take_snapshot")
        if response and response.get("status") == "success" and "image_base64" in response:
            try:
                return base64.b64decode(response["image_base64"])
            except (TypeError, base64.binascii.Error) as e:
                logger.error(f"Failed to decode base64 image: {e}")
                return None
        logger.error(f"Failed to take snapshot. Response: {response}")
        return None
