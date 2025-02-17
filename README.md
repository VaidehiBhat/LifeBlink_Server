# LifeBlink

LifeBlink is an innovative smart eyewear solution integrating PPG (heart rate) and EOG (eye blink) sensors to monitor and process biometric data. Designed to assist individuals with speech impairments, LifeBlink translates blinks into predefined language commands, enabling non-verbal communication. The system collects data, processes it using machine learning, and provides insights such as blink count and heart rate. Based on blink detection, the ESP32 can trigger audio playback via a Bluetooth speaker or convert blinks into text or speech.

## Features

Smart Eyewear for Accessibility: Converts blinks into language-based communication.

PPG Heart Rate Monitoring: Detects heartbeats and calculates heart rate.

EOG Blink Detection: Counts blinks and determines eye movement patterns.

ESP32 Data Handling: Stores sensor data on an SD card and uploads it to a Flask API.

Flask API on Render: Processes sensor data and returns results.

Audio Feedback & Speech Output: Plays specific audio files or converts blinks to speech based on user input.

Test Case Examples: Demonstrates real-world usage with images.

## Directory Structure

LifeBlink/
│-- requirements.txt        # Dependencies for the Flask API
│-- server.py               # Flask API server
│-- src/                    # Arduino codes for ESP32
│-- validation/             # Validation scripts and ML models
│   ├── codes/              # Processing scripts
│   ├── dataset/            # Training datasets and ML code
│-- examples/               # Test case images

## Installation and Setup

### Prerequisites

ESP32 with SD card module

PPG and EOG sensors

Python 3.x

Render account for Flask API deployment

Arduino IDE for ESP32 programming

### Clone Repository

git clone https://github.com/your-username/LifeBlink.git
cd LifeBlink

### Install Dependencies

pip install -r requirements.txt

### Running the Flask API

python server.py

### ESP32 Code Deployment

Open the Arduino IDE.

Install required ESP32 board support and libraries.

Upload the code from the src/ directory.

## Usage

Power up the ESP32.

It collects data from PPG and EOG sensors.

Stores the data on an SD card and uploads it to the Flask API.

The Flask API processes the data and returns blink count & heart rate.

If the blink count matches a predefined threshold, the ESP32 plays audio via Bluetooth or converts the blinks into speech/text.

## Example Test Cases

Refer to the examples/ directory for images of real-world test scenarios.

## Potential Impact

LifeBlink is designed to empower individuals with speech impairments by offering a seamless way to communicate through eye blinks. This innovation can greatly enhance accessibility and independence for users with conditions such as ALS, paralysis, or speech disorders.

## Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request.



