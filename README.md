# Hand-Tracking-Using-Opencv

Real-time hand tracking using OpenCV and MediaPipe.

## About

This project detects and tracks hand landmarks from:

- live webcam feed
- a saved video file

It draws hand landmarks, highlights landmark point 0, and shows FPS in real time.

## Project Status

This repository is currently maintained and modified by Zeeshan.

It is an adapted project, and ongoing changes are being added in this version.

## Repository

GitHub: https://github.com/Zeeshan5932/Hand-Tracking-Using-Opencv

## Built With

- OpenCV
- MediaPipe
- Python

## Files

- app.py: Hand tracking from webcam
- Hand Tracking from Media .py: Hand tracking on video file and saves output video
- requirements.txt: Python dependencies

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Zeeshan5932/Hand-Tracking-Using-Opencv.git
cd Hand-Tracking-Using-Opencv
```

2. Create and activate a virtual environment (recommended):

```bash
conda create -n handtrack python=3.10 -y
conda activate handtrack
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run webcam mode:

```bash
python app.py
```

Run video-file mode:

```bash
python "Hand Tracking from Media .py"
```

Note: In Hand Tracking from Media .py, set video_path to your input video path before running.

## Contributing

Suggestions and improvements are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push branch
5. Open a pull request

## License

Distributed under the MIT License. See LICENSE for details.

## Contact

Zeeshan

Email: zeeshanoffical01@gmail.com
