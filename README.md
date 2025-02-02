# Face Recognition Attendance System

This project is a **Face Recognition-based Attendance System** developed using **OpenCV**, **NumPy**, **Pickle**, and **scikit-learn (K-Nearest Neighbors)**. The system captures face data, trains a machine learning model, and records attendance in a CSV file.

## Features

- Real-time face detection and recognition.
- Stores and manages face data using **Pickle**.
- Records attendance with **timestamp** and **date** in CSV format.
- Supports adding new user face data.

## Technologies Used

- **Python 3.7+**
- **OpenCV**: For face detection and video capture.
- **NumPy**: For data manipulation.
- **Pickle**: For persistent storage of face data and user names.
- **scikit-learn (KNeighborsClassifier)**: For face classification.

## Project Structure

```
FaceRecognitionAttendanceSystem/
├── analysis.py       # Script to capture face data
├── attendance.py    # Main script to run the attendance system
├── data/                   # Directory to store face data and names
│   ├── face_data.pkl       # Pickled face data
│   └── names.pkl           # Pickled user names
└── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
```

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd FaceRecognitionAttendanceSystem
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Haar Cascade:**
   Ensure you have `haarcascade_frontalface_default.xml` in the project directory. You can download it from [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades).

5. **Run Face Data Capture:**

   ```bash
   python analyse.py
   ```

   Follow the on-screen instructions to capture 100 face samples for a user.

6. **Run the Attendance System:**

   ```bash
   python attendance.py
   ```

   Press **'o'** to record attendance and **'q'** to quit.

## CSV Attendance Output

Attendance records are saved in `Attendance/attendance_<date>.csv`. Each record includes:

- Name
- Time
- Date

## Requirements

The project requires the following dependencies:

```
opencv-python==4.5.5.64
numpy==1.21.4
epickledb==0.2
scikit-learn==1.0.2
```

## Example Usage



## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.

