// Microcontroller side of the UNO Q minifig car.
//
// Two motors on a Cytron Maker Drive. Each motor has two inputs (A/B):
//   A = PWM, B = LOW  -> forward
//   A = LOW, B = PWM  -> backward
//   both LOW          -> coast (stop)
//
// The Linux side (python/main.py) calls over the Bridge:
//   set_speed(speed)  -> speed -100..100 (% of full power) for both motors
// Both motors always get the same PWM duty: there is a single speed, and the
// duty is computed once and written to both motors together.
// If no command arrives for COMMAND_TIMEOUT_MS, the motors stop, so the car
// halts if the computer program, Wi-Fi, or MQTT broker drops out.
//
// Wiring: UNO Q GND -> Maker Drive GND (common ground is required).

#include <Arduino_RouterBridge.h>

// PWM-capable pins wired to the Maker Drive inputs (all four are 500 Hz PWM
// on the UNO Q). Change to match your wiring.
const int M1A_PIN = 3;
const int M1B_PIN = 5;
const int M2A_PIN = 6;
const int M2B_PIN = 9;

// Direction only (never speed). Motors mounted mirror-image on a car spin
// opposite ways for the same command, so one is inverted. Flip BOTH to
// reverse which way the car drives; flip ONE if the wheels fight each other.
const bool INVERT_M1 = true;
const bool INVERT_M2 = false;

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

// Set one motor's direction with the shared duty `pwm` (0 = stop).
void writeMotor(int pinA, int pinB, bool forward, int pwm) {
  analogWrite(pinA, forward ? pwm : 0);
  analogWrite(pinB, forward ? 0 : pwm);
}

// Drive both motors at the same speed. The duty is computed once, so the two
// motors can't receive different speeds.
void driveBoth(int speed) {
  int pwm = map(abs(speed), 0, 100, 0, 255);
  bool forward = speed >= 0;
  writeMotor(M1A_PIN, M1B_PIN, forward != INVERT_M1, pwm);
  writeMotor(M2A_PIN, M2B_PIN, forward != INVERT_M2, pwm);
}

void setup() {
  pinMode(M1A_PIN, OUTPUT);
  pinMode(M1B_PIN, OUTPUT);
  pinMode(M2A_PIN, OUTPUT);
  pinMode(M2B_PIN, OUTPUT);
  driveBoth(0);

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
    driveBoth(targetSpeed);
  }

  delay(10);
}
