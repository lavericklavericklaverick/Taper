#include <Arduino.h>
#include <AccelStepper.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ============================================================================
// HARDWARE PINOUT
// ============================================================================
constexpr uint8_t PIN_EN   = 21;
constexpr uint8_t PIN_DIR  = 22;
constexpr uint8_t PIN_PUL  = 23;

// Interface type 1 = External Driver (PUL/DIR pins)
AccelStepper stepper(AccelStepper::DRIVER, PIN_PUL, PIN_DIR);

// ============================================================================
// BLE CONFIGURATION (Nordic UART Service)
// ============================================================================
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

BLEServer *pServer = nullptr;
BLECharacteristic *pTxCharacteristic = nullptr;

bool deviceConnected = false;
volatile bool shouldStep = false;

// Microsteps to turn each time 'R' is received (use negative to invert direction)
const long STEPS_INCREMENT = 100;

// ============================================================================
// BLE CALLBACKS
// ============================================================================
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
  }
  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    BLEDevice::startAdvertising(); // Allow phone to reconnect
  }
};

class RxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pChar) override {
    String rxValue = pChar->getValue();
    if (rxValue.length() > 0) {
      char cmd = rxValue[0];

      // 'R' received from phone to trigger a step
      if (cmd == 'R' && !shouldStep) {
        stepper.move(STEPS_INCREMENT);
        shouldStep = true;
      }
    }
  }
};

void setup() {
  Serial.begin(115200);

  // Enable Pin (Active LOW on most drivers like A4988 / TB6600)
  pinMode(PIN_EN, OUTPUT);
  digitalWrite(PIN_EN, LOW);

  // Stepper motion configuration
  stepper.setMaxSpeed(2000.0);
  stepper.setAcceleration(4000.0);

  // Initialize BLE
  BLEDevice::init("ESP32C6_Stepper");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  // TX Characteristic (Notifies phone when step finishes)
  pTxCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID_TX,
                        BLECharacteristic::PROPERTY_NOTIFY
                      );
  pTxCharacteristic->addDescriptor(new BLE2902());

  // RX Characteristic (Receives commands from phone)
  BLECharacteristic *pRxCharacteristic = pService->createCharacteristic(
                                           CHARACTERISTIC_UUID_RX,
                                           BLECharacteristic::PROPERTY_WRITE
                                         );
  pRxCharacteristic->setCallbacks(new RxCallbacks());

  pService->start();
  BLEDevice::startAdvertising();

  Serial.println("[+] BLE Stepper Controller Initialized.");
}

void loop() {
  // Drive the stepper smoothly with AccelStepper
  if (stepper.distanceToGo() != 0) {
    stepper.run();
  } else if (shouldStep) {
    // Stepping finished: notify the Pixel 6a to capture the next frame
    shouldStep = false;

    if (deviceConnected && pTxCharacteristic != nullptr) {
      uint8_t ack[] = {'D', 'O', 'N', 'E', '\n'};
      pTxCharacteristic->setValue(ack, 5);
      pTxCharacteristic->notify();
    }
  }
}