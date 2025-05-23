# Object Detection

Repository for the object detection microservice of the SmartWorkMCD project 2024/25 UA

## Module Description

### Description

- Uses computer vision to detect objects at each workstation.
- Identifies components in the candy box assembly process to ensure correct
placement.
- Provides real-time feedback to the Workstation Brain module.

### Functionality

- Captures images via a camera at each workstation.
- Runs AI-based object tracking models optimized for Raspberry Pi 5 and Coral Dev
Board.
- Sends data to the Workstation Brain for decision-making.

### Responsible Members

- Main Responsible: Mateus Aleixo
- Team: Mateus Aleixo, Pedro Batista, Pedro Mendes, Hugo Veríssimo

## Running the App

To run the app locally, follow these steps:

1. **Clone the repository**:

    ```sh
    git clone https://github.com/SmartWorkMCD/Object_Detection.git
    cd Object_Detection
    ```

2. **Create a virtual environment and activate it**:

    ```sh
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    source .venv/bin/activate  # On macOS/Linux
    ```

3. **Install the dependencies**:

    ```sh
    pip install -r requirements.txt
    ```

4. **Run Neighbooring Components (The module migth require another module to be runnning)**

    ```sh
    docker compose up
    ```

5. **Run the app**:

    ```sh
    python3 app/main.py
    ```

## Running the Tests

To run the tests, follow these steps:

1. **Ensure the virtual environment is activated**:

    ```sh
    .venv\Scripts\activate  # On Windows
    source .venv/bin/activate  # On macOS/Linux
    ```

2. **Run the tests using pytest**:

    ```sh
    pytest
    ```

## Module Structure

The module is structured as follows:

```sh
Object_Detection/
├── app/
│   ├── classes/
│   └── ...
├── data/
│   ├── annotations/
│   ├── augmented_data/
│   │   ├── frames/
│   │   └── masks/
│   ├── frames/
│   ├── masks/
│   ├── split_data/
│   │   ├── images/
│   │   └── labels/
│   └── videos/
├── models/
├── tests/
├── ultralytics/ # YOLO results
├── ...
└── README.md
```

## MQTT Payload Format

The object detection module currently sends detection results as a flat dictionary (serialized as JSON or pickle) containing both YOLO and RF-DETR outputs. Each detection is represented with indexed keys for bounding box coordinates, scores, and class labels.

**Example flat dictionary payload (JSON):**
```json
{
    "timestamp": 1717267200.123,
    "yolo_0_x1": 120,
    "yolo_0_y1": 80,
    "yolo_0_x2": 200,
    "yolo_0_y2": 160,
    "yolo_0_score": 0.98,
    "yolo_0_class": "red",
    "yolo_1_x1": 50,
    "yolo_1_y1": 40,
    "yolo_1_x2": 300,
    "yolo_1_y2": 220,
    "yolo_1_score": 0.95,
    "yolo_1_class": "box",
    "rfdetr_0_x1": 130,
    "rfdetr_0_y1": 90,
    "rfdetr_0_x2": 210,
    "rfdetr_0_y2": 170,
    "rfdetr_0_confidence": 0.93,
    "rfdetr_0_class_id": 2
}
```

- **YOLO results** are prefixed with `yolo_{i}_` and include bounding box coordinates (`x1`, `y1`, `x2`, `y2`), detection score, and class label.
- **RF-DETR results** are prefixed with `rfdetr_{i}_` and include bounding box coordinates, confidence score, and class ID.

This flat structure allows easy parsing and supports both JSON and pickle serialization for efficient communication between modules.

## Contribution Guidelines

To ensure a smooth collaboration, please follow these guidelines:

1. **Do not push directly to the `main` branch**: All changes should be made through pull requests.
2. **Submit pull requests to the `development` branch**: Create a new branch for your feature or bug fix and submit a pull request to the `development` branch.
3. **Write clear commit messages**: Use descriptive commit messages that explain the purpose of the changes.
4. **Run tests before submitting a pull request**: Ensure that all tests pass and that your changes do not introduce any new issues.
5. **Follow the coding standards**: Maintain consistent coding style and adhere to the project's coding standards.

### Example Workflow

1. **Create a new branch**:

    ```sh
    git checkout -b feature/my-new-feature
    ```

2. **Make your changes and commit them**:

    ```sh
    git add .
    git commit -m "Add new feature"
    ```

3. **Push your branch to GitHub**:

    ```sh
    git push origin feature/my-new-feature
    ```

4. **Create a pull request**: Go to the GitHub repository and create a pull request to the `development` branch.

Thank you for contributing to the project!
