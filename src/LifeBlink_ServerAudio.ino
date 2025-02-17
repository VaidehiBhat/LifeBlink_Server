#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <SD.h>
#include <SPI.h>
#include <WebServer.h> // Web server library
#include <time.h>
int globalBlinkCount = -1;  // Stores blink count

const char* ssid = "Lavanya vivo";
const char* password = "23452345";
const char* renderEOG_URL = "https://lifeblink-server.onrender.com/process_eog";
const char* renderPPG_URL = "https://lifeblink-server.onrender.com/process_ppg";
const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 19800;  // Change based on your timezone (e.g., for IST, use 19800)
const int   daylightOffset_sec = 0;
#define SD_CS 5  // Chip select pin for SD card module
#define EOG_SENSOR_PIN 34
#define PPG_SENSOR_PIN 35

WebServer server(80);
String logBuffer = ""; // Buffer to store logs

// Function to log messages to Serial Monitor and Webpage
void logPrint(String message) {
    Serial.println(message);
    logBuffer += message + "<br>"; // Append to buffer for webpage
}

// Serve Webpage with Logs
void handleRoot() {
    String html = "<html><head><meta http-equiv='refresh' content='2'></head><body>";
    html += "<h2>ESP32 Serial Logs</h2><pre>" + logBuffer + "</pre>";
    html += "<h3>Audio Playback</h3><a href='/play_audio'>Play Audio</a>";
    html += "</body></html>";
    
    server.send(200, "text/html", html);
}
void handleAudioPlayback(int blinkCount) {
    String audioFile;

    // Select the correct audio file based on blink count
    if (blinkCount == 1 || blinkCount == 2) {
        audioFile = "/audio/blink1_2.wav";
    } else if (blinkCount==3 || blinkCount == 4 || blinkCount == 5) {
        audioFile = "/audio/blink3_4_5.wav";
    } else if (globalBlinkCount==0) {
        audioFile = "/audio/blink0.wav";
    }
    else {
        audioFile = "/audio/blink_6.wav";
        return;
    }

    logPrint("Attempting to play audio: " + audioFile);

    // Open the file from SD card
    File file = SD.open(audioFile);
    if (!file) {
        logPrint("Error: File not found - " + audioFile);
        server.send(500, "text/plain", "Audio file not found on SD card");
        return;
    }

    // Send the audio file to the client
    server.streamFile(file, "audio/wav");
    file.close();

    logPrint("Successfully streamed audio file: " + audioFile);
}

void handleAudioRequest() {
    if (globalBlinkCount == -1) {
        server.send(500, "text/plain", "Error: Blink count not available!");
        return;
    }

    logPrint("Received audio request for stored blink count: " + String(globalBlinkCount));

    //handleAudioPlayback(globalBlinkCount);  // Play the correct audio
    String audioFile;

    // Select the correct audio file based on blink count
    if (globalBlinkCount == 1 || globalBlinkCount == 2) {
        audioFile = "/audio/blink1_2.wav";
    } else if (globalBlinkCount==3|| globalBlinkCount == 4 || globalBlinkCount == 5) {
        audioFile = "/audio/blink3_4_5.wav";
    } else if (globalBlinkCount==0) {
        audioFile = "/audio/blink0.wav";
    }
      else {
        audioFile = "/audio/blink_6.wav";
        //return;
    }

    logPrint("Attempting to play audio: " + audioFile);

    // Open the file from SD card
    File file = SD.open(audioFile);
    if (!file) {
        logPrint("Error: File not found - " + audioFile);
        server.send(500, "text/plain", "Audio file not found on SD card");
        return;
    }

    // Send the audio file to the client
    server.streamFile(file, "audio/wav");
    file.close();

    logPrint("Successfully streamed audio file: " + audioFile);

    server.send(200, "text/plain", "Playing audio for blink count: " + String(globalBlinkCount));
}
String getTimestamp() {
    struct tm timeinfo;
    if (!getLocalTime(&timeinfo)) {
        Serial.println("Failed to obtain time");
        return "00-00-00_00-00-00";  // Fallback timestamp
    }

    char buffer[20];
    strftime(buffer, sizeof(buffer), "%d-%m-%y_%H-%M-%S", &timeinfo);
    return String(buffer);
}


void setup() {
    Serial.begin(115200);
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        logPrint("Connecting to WiFi...");
    }
    logPrint("Connected to WiFi!");
    logPrint("ESP32 IP Address: " + WiFi.localIP().toString());

    if (!SD.begin(SD_CS)) {
        logPrint("SD Card initialization failed!");
        return;
    }
    logPrint("SD Card initialized.");

    // Start Web Server
    server.on("/", handleRoot);
    server.on("/play_audio", HTTP_GET, handleAudioRequest);
    server.begin();
    logPrint("Web Server started! Access logs at: http://" + WiFi.localIP().toString());
    logPrint("Starting Data Collection in 3...");
    delay(1000);
    logPrint("...2...");
    delay(1000);
    logPrint("...1...");
    delay(1000);
    logPrint("Start!");
    // Generate unique filenames based on time
    String filenameEOG = "/sensor1_" + getTimestamp() + ".csv";
    String filenamePPG = "/sensor2_" + getTimestamp() + ".csv";

    // Collect EOG Data
    logPrint("Collecting EOG data...");
    String eogData = collectEOGData(EOG_SENSOR_PIN, 8000);
    saveToSD(filenameEOG, eogData);
    int blinkCount = sendDataToRender(renderEOG_URL, filenameEOG);
    logPrint("Blink Count: " + String(blinkCount));
    globalBlinkCount=blinkCount;
    //handleAudioPlayback(blinkCount);


    // Wait 10 seconds
    delay(10000);

    // Collect PPG Data
    logPrint("Collecting PPG data...");
    String ppgData = collectPPGData(PPG_SENSOR_PIN, 20000);
    saveToSD(filenamePPG, ppgData);
    int heartRate = sendDataToRender(renderPPG_URL, filenamePPG);

    logPrint("Heart Rate: " + String(heartRate));

    logPrint("Process complete.");
}

void loop() {
    server.handleClient(); // Handle web server requests
}

// Existing functions remain unchanged

float EOGFilter(float input) {
    float output = input;
    {
        static float z1, z2;
        float x = output - 0.02977423*z1 - 0.04296318*z2;
        output = 0.09797471*x + 0.19594942*z1 + 0.09797471*z2;
        z2 = z1;
        z1 = x;
    }
    {
        static float z1, z2;
        float x = output - 0.08383952*z1 - 0.46067709*z2;
        output = 1.00000000*x + 2.00000000*z1 + 1.00000000*z2;
        z2 = z1;
        z1 = x;
    }
    {
        static float z1, z2;
        float x = output - -1.92167271*z1 - 0.92347975*z2;
        output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
        z2 = z1;
        z1 = x;
    }
    {
        static float z1, z2;
        float x = output - -1.96758891*z1 - 0.96933514*z2;
        output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
        z2 = z1;
        z1 = x;
    }
    return output;
}

String collectEOGData(int sensorPin, int duration) {
    String data = "timestamp,value\n";
    unsigned long startTime = millis();
    static unsigned long past = 0;
    static long timer = 0;

    while (millis() - startTime < duration) {
        unsigned long present = micros();
        unsigned long interval = present - past;
        past = present;
        timer -= interval;

        if (timer < 0) {
            timer += 1000000 / 75;
            float rawValue = analogRead(sensorPin);
            float filteredValue = EOGFilter(rawValue);
            data += String(millis()) + "," + String(filteredValue) + "\n";
        }
    }
    return data;
}

String collectPPGData(int sensorPin, int duration) {
    String data = "timestamp,value\n";
    unsigned long startTime = millis();

    while (millis() - startTime < duration) {
        int value = analogRead(sensorPin);
        data += String(millis()) + "," + String(value) + "\n";
        delay(10);
    }
    return data;
}

void saveToSD(String filename, String data) {
    File file = SD.open(filename, FILE_WRITE);
    if (file) {
        file.print(data);
        file.close();
        logPrint("Data saved to: " + filename);
    } else {
        logPrint("Error saving file: " + filename);
    }
}

int sendDataToRender(String serverURL, String filename) {
    File file = SD.open(filename);
    if (!file) {
        logPrint("Failed to open file: " + filename);
        return -1;
    }

    String jsonPayload = "{\"values\":[";
    bool first = true;
    
    while (file.available()) {
        String line = file.readStringUntil('\n');
        if (line.startsWith("timestamp")) continue;  // Skip headers

        int commaIndex = line.indexOf(',');
        if (commaIndex != -1) {
            String value = line.substring(commaIndex + 1);
            value.trim(); // Remove any extra spaces or newlines
            if (!first) jsonPayload += ",";
            jsonPayload += value;
            first = false;
        }
    }
    file.close();

    jsonPayload += "]}";  // Close JSON array

    // If the payload is empty, prevent sending
    if (jsonPayload == "{\"values\":[]}") {
        logPrint("No valid data to send.");
        return -1;
    }

    logPrint("Sending JSON: " + jsonPayload);  // Log the full payload

    // HTTP Request
    HTTPClient http;
    http.begin(serverURL);
    http.addHeader("Content-Type", "application/json");

    int httpResponseCode = http.POST(jsonPayload);
    int result = -1;

    if (httpResponseCode > 0) {
        String response = http.getString();
        //logPrint("Raw Response: " + response);

        // Parse JSON response
        DynamicJsonDocument doc(1024);
        DeserializationError error = deserializeJson(doc, response);
        
        if (error) {
            logPrint("JSON Parsing Failed!");
            return -1;
        }

        if (doc.containsKey("blink_count")) {
            result = doc["blink_count"].as<int>();
        } else if (doc.containsKey("heart_rate")) {
            result = doc["heart_rate"].as<int>();
        } else {
            logPrint("Unexpected JSON format!");
            return -1;
        }
    } else {
        logPrint("HTTP Request Failed, Code: " + String(httpResponseCode));
    }

    http.end();
    return result;
}
