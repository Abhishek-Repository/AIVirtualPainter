
# AIVirtualPainter

This project introduces a contactless virtual drawing tool controlled by hand gestures using the MediaPipe framework for hand tracking. The tool enables intuitive drawing and erasing functionalities without requiring a physical input device, enhancing accessibility for users in various fields. Our experiments demonstrate reliable gesture recognition and responsive performance across varied environments, with potential applications in digital art, education, and virtual reality.


## Requirements

This project requires the following Python libraries:
opencv-python~=4.10.0.84
numpy~=2.1.3
mediapipe~=0.10.18
opencv-contrib-python~=4.10.0.84



To install all the required libraries, simply run:

```bash
pip install -r requirements.txt
```

This will install the necessary dependencies listed in the `requirements.txt` file.

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/Abhishek-Repository/AIVirtualPainter.git
cd AIVirtualPainter
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your webcam is connected.
2. Run the main script:

```bash
python main.py
```

This will open a webcam feed, and the program will detect hands in the frame and display their positions. The confidence threshold for detection is set to 0.85, but this can be adjusted in the code.

## Troubleshooting

- **Camera not working:** Ensure the camera is properly connected and recognized by your system. Try running other webcam applications to verify.
- **Hand not detected:** Try adjusting the `detectionCon` parameter for better sensitivity or lighting conditions.

