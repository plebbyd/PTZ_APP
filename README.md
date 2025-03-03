# PTZ APP

This is an application for sending images of specific objects autonomously using PTZ cameras.

## Build the container

```bash
sudo docker buildx build --platform=linux/amd64,linux/arm64/v8 -t your_docker_hub_user_name/ptzapp -f Dockerfile --push .
```

Then pull the container from dockerhub in the node:

```bash
sudo docker image pull your_docker_hub_user_name/ptzapp
```

## Run the container on a dell blade

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest -ki -it 5 -un camera_user_name -pw camera_password -ip camera_ip_address -obj person,car
```

## Run the container on a waggle node

```bash
sudo docker run -it --rm your_docker_hub_user_name/ptzapp:latest -ki -it 5 -un camera_user_name -pw camera_password -ip camera_ip_address -obj person,car
```

## Example with Florence model

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --model Florence-base --iterations 5 --username dario --password 'Why1Not@' --cameraip 130.202.23.92 --objects 'person,car'
```

## Using Different Object Detection Models

### YOLO (Default)
By default, the application uses the YOLO model (yolo11n) for object detection. Specify objects by name:

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --objects 'person,car,dog'
```

### Florence Models
When using Florence models, you have more powerful detection capabilities:

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --model Florence-base --objects 'person,car'
```

### Detecting All Objects with Florence

To detect all objects using Florence models, use the asterisk:

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --model Florence-base --objects '*'
```

**Note:** When using `'*'` with Florence models, the application runs in the `<OD>` task mode, which enables general object detection without filtering for specific classes. This can be useful for inventorying all objects in a scene but may produce more diverse results than when targeting specific objects.

## Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--model` | `-m` | Model to use (e.g., 'yolo11n', 'Florence-base') | yolo11n |
| `--iterations` | `-it` | Number of iterations (PTZ rounds) to run | 5 |
| `--username` | `-un` | PTZ camera username | "" |
| `--password` | `-pw` | PTZ camera password | "" |
| `--cameraip` | `-ip` | PTZ camera IP address | "" |
| `--objects` | `-obj` | Objects to detect (comma-separated or '*' for everything) | "person" |
| `--keepimages` | `-ki` | Keep collected images in persistent folder | False |
| `--panstep` | `-ps` | Step of pan in degrees | 15 |
| `--tilt` | `-tv` | Tilt value in degrees | 0 |
| `--zoom` | `-zm` | Zoom value | 1 |
| `--confidence` | `-conf` | Confidence threshold (0-1) | 0.1 |
| `--iterdelay` | `-id` | Minimum delay in seconds between iterations | 60.0 |
| `--debug` | | Enable debug level logging | False |
