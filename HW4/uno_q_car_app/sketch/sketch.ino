// Microcontroller side of the UNO Q minifig car.
//
// Two motors on a Cytron Maker Drive. Each motor has two inputs (A/B):
//   A = PWM, B = LOW  -> forward
//   A = LOW, B = PWM  -> backward
//   both LOW          -> coast (stop)
//
// The Linux side (python/main.py) calls over the Bridge:
//   set_speed(speed)  -> speed -100..100 (% of full power) for both motors
// If no command arrives for COMMAND_TIMEOUT_MS, the motors stop, so the car
// halts if the computer program, Wi-Fi, or MQTT broker drops out.
//
// Wiring: UNO Q GND -> Maker Drive GND (common ground is required).

#include <Arduino_RouterBridge.h>

// PWM-capable pins wired to the Maker Drive inputs. Change to match your wiring.
const int M1A_PIN = 3;
const int M1B_PIN = 5;
const int M2A_PIN = 6;
const int M2B_PIN = 9;

// Motors mounted mirror-image on a car spin opposite ways for the same
// command; set one of these true so both wheels drive the car the same way.
const bool INVERT_M1 = false;
const bool INVERT_M2 = true;

const unsigned long COMMAND_TIMEOUT_MS = 500;

// Written by the Bridge handler, read by loop(). The handler only records the
// request; loop() does all the pin writes.
volatile int targetSpeed = 0;
volatile bool dirty = true;
volatile unsigned long lastCommandMs = 0;

void set_speed(int speed) {
  if (speed > 100) speed = 100;
  if (speed < -100) speed = -100;
  targetSpeed = speed;
  lastCommandMs = millis();
  dirty = true;
}

void driveMotor(int pinA, int pinB, int speed, bool invert) {
  if (invert) speed = -speed;
  int pwm = map(abs(speed), 0, 100, 0, 255);
  if (speed > 0) {
    analogWrite(pinA, pwm);
    analogWrite(pinB, 0);
  } else if (speed < 0) {
    analogWrite(pinA, 0);
    analogWrite(pinB, pwm);
  } else {
    analogWrite(pinA, 0);
    analogWrite(pinB, 0);
  }
}

void setup() {
  pinMode(M1A_PIN, OUTPUT);
  pinMode(M1B_PIN, OUTPUT);
  pinMode(M2A_PIN, OUTPUT);
  pinMode(M2B_PIN, OUTPUT);
  driveMotor(M1A_PIN, M1B_PIN, 0, INVERT_M1);
  driveMotor(M2A_PIN, M2B_PIN, 0, INVERT_M2);

  Bridge.begin();
  Bridge.provide("set_speed", set_speed);
}

void loop() {
  if (targetSpeed != 0 && millis() - lastCommandMs > COMMAND_TIMEOUT_MS) {
    targetSpeed = 0;
    dirty = true;
  }

  if (dirty) {
    dirty = false;
    int speed = targetSpeed;
    driveMotor(M1A_PIN, M1B_PIN, speed, INVERT_M1);
    driveMotor(M2A_PIN, M2B_PIN, speed, INVERT_M2);
  }

  delay(10);
}
