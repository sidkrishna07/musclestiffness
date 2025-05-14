#include <ArduinoBLE.h>
#include <Nano33BLE_System.h>
#include <Arduino_LSM9DS1.h>

// Device & BLE UUIDs
const char* DEVICE_NAME   = "MStiffSens";
const char* SERVICE_UUID  = "0000FFFF-0000-1000-8000-00805F9B34FB";
const char* CHAR_UUID     = "0000FFFE-0000-1000-8000-00805F9B34FB";

// BLE service & characteristic (10 float values)
BLEService stiffnessService(SERVICE_UUID);
BLECharacteristic featureChar(
  CHAR_UUID,
  BLERead   | BLENotify,
  10 * sizeof(float)
);

// Vibration motor pin & timing
const int    MOTOR_PIN       = 9;
const unsigned long INTERVAL = 5000;   // ms between vibrations
const unsigned long DURATION = 1000;   // ms vibration on

// Feature buffer (10 values)
const int   NUM_FEATURES = 10;
float       features[NUM_FEATURES];

// State
unsigned long lastMotorTime = 0;
bool          motorActive   = false;
unsigned long motorStart    = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial && millis() < 5000);

  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU");
    while (1);
  }

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Failed to initialize BLE");
    while (1);
  }

  // Motor pin
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);

  // Configure BLE
  BLE.setDeviceName(DEVICE_NAME);
  BLE.setLocalName(DEVICE_NAME);
  BLE.setAdvertisedService(stiffnessService);

  stiffnessService.addCharacteristic(featureChar);
  BLE.addService(stiffnessService);

  // Seed with zeros
  featureChar.writeValue((byte*)features, sizeof(features));

  BLE.advertise();
  Serial.println("Advertising as " + String(DEVICE_NAME));
}

void loop() {
  // Wait for a central to connect
  BLEDevice central = BLE.central();
  if (!central) return;

  Serial.print("Connected: ");
  Serial.println(central.address());

  while (central.connected()) {
    unsigned long now = millis();

    // Start motor at intervals
    if (!motorActive && now - lastMotorTime >= INTERVAL) {
      motorActive = true;
      motorStart  = now;
      digitalWrite(MOTOR_PIN, HIGH);
      Serial.println("Motor ON");
    }

    // Stop motor after DURATION
    if (motorActive && now - motorStart >= DURATION) {
      motorActive   = false;
      digitalWrite(MOTOR_PIN, LOW);
      lastMotorTime = now;
      Serial.println("Motor OFF");

      // Once vibration ends, collect & send features
      if (featureChar.subscribed()) {
        collectAndSendFeatures();
      }
    }

    delay(10);
  }

  Serial.print("Disconnected: ");
  Serial.println(central.address());
  BLE.advertise();
}

void collectAndSendFeatures() {
  // Sample 10 acceleration magnitudes
  for (int i = 0; i < NUM_FEATURES; i++) {
    while (!IMU.accelerationAvailable());
    float x, y, z;
    IMU.readAcceleration(x, y, z);
    features[i] = sqrt(x*x + y*y + z*z);
  }

  // Send 10 floats over BLE
  featureChar.writeValue((byte*)features, sizeof(features));

  // Debug print
  Serial.println("Sent features:");
  for (int i = 0; i < NUM_FEATURES; i++) {
    Serial.println(features[i], 6);
  }
}