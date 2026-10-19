// Microcontroller side of the HW5 sumo shake controller.
//
// Reads an MPU-6050 at 200 Hz over I2C and turns accelerometer + gyro into one
// "shake intensity" from 0 (still) to 1 (shaken hard). The Linux side
// (python/main.py) polls it over the Bridge and publishes it over MQTT.
//
// Wiring (GY-521 breakout): VCC -> 3.3V, GND -> GND, SDA -> SDA, SCL -> SCL
// (the header's I2C pins = Wire). AD0 unconnected/low -> address 0x68.
//
// Fusion of the two sensors:
//   - gravity estimate: a complementary filter. The gyro rotates the estimate
//     between samples and the accelerometer slowly pulls it back, so tilting or
//     turning the board isn't counted as shaking.
//   - dynamic accel = |accel - gravity|, rotational shake = |gyro|
//   - raw = rms(dynamic accel) + rms(gyro) / GYRO_PER_G, over ~0.2 s
//   - envelope: fast attack, slow release, so the robot doesn't flip between
//     attack and defense between strokes of a shake
//
// Bridge:
//   get_shake() -> float 0..1, or -1 if the MPU-6050 isn't answering

#include <Arduino_RouterBridge.h>
#include <Wire.h>

const uint8_t MPU_ADDR = 0x68;
const unsigned long SAMPLE_US = 5000;        // 200 Hz
const float ACCEL_LSB_PER_G = 4096.0f;       // +-8 g range
const float GYRO_LSB_PER_DPS = 32.8f;        // +-1000 deg/s range

const float GRAVITY_BLEND = 0.02f;           // how fast the accelerometer corrects the gravity estimate
const float RMS_TAU_S = 0.2f;
const float GYRO_PER_G = 500.0f;             // 500 deg/s of rotation counts like 1 g of shaking
const float REST_LEVEL = 0.06f;              // noise / hand tremor below this = still
const float SHAKE_MAX = 1.8f;                // raw level that counts as full aggression (tune by hand)
const float ATTACK_TAU_S = 0.10f;
const float RELEASE_TAU_S = 0.4f;            // was 1.0: stopping the shake now shows within ~0.5 s

volatile float shakeOut = 0.0f;              // read by the Bridge handler
volatile bool sensorOk = false;

float gx_est = 0, gy_est = 0, gz_est = 1;     // gravity estimate (g, sensor frame)
float msAccel = 0, msGyro = 0, envelope = 0;
unsigned long lastSampleUs = 0;
unsigned long lastInitTryMs = 0;
int failCount = 0;

float get_shake() {
  return sensorOk ? shakeOut : -1.0f;
}

bool writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool mpuInit() {
  bool ok = writeReg(0x6B, 0x01);            // PWR_MGMT_1: wake, PLL with gyro X clock
  delay(10);
  ok = ok && writeReg(0x1A, 0x03);           // CONFIG: DLPF ~44 Hz
  ok = ok && writeReg(0x19, 0x04);           // SMPLRT_DIV: 1 kHz / 5 = 200 Hz
  ok = ok && writeReg(0x1B, 0x10);           // GYRO_CONFIG: +-1000 deg/s
  ok = ok && writeReg(0x1C, 0x10);           // ACCEL_CONFIG: +-8 g
  return ok;
}

// Burst-read accel (g) and gyro (deg/s). False on an I2C error.
bool mpuRead(float &ax, float &ay, float &az, float &wx, float &wy, float &wz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                          // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 14) != 14) return false;
  int16_t v[7];
  for (int i = 0; i < 7; i++) {
    uint8_t hi = Wire.read();
    uint8_t lo = Wire.read();
    v[i] = (int16_t)((hi << 8) | lo);
  }
  ax = v[0] / ACCEL_LSB_PER_G;  ay = v[1] / ACCEL_LSB_PER_G;  az = v[2] / ACCEL_LSB_PER_G;
  // v[3] is temperature
  wx = v[4] / GYRO_LSB_PER_DPS; wy = v[5] / GYRO_LSB_PER_DPS; wz = v[6] / GYRO_LSB_PER_DPS;
  return true;
}

void update(float dt, float ax, float ay, float az, float wx, float wy, float wz) {
  // Rotate the gravity estimate with the gyro: dg/dt = -w x g (sensor frame), w in rad/s.
  const float k = dt * 0.0174533f;
  float rx = wx * k, ry = wy * k, rz = wz * k;
  float nx = gx_est - (ry * gz_est - rz * gy_est);
  float ny = gy_est - (rz * gx_est - rx * gz_est);
  float nz = gz_est - (rx * gy_est - ry * gx_est);
  // Then pull it gently toward the accelerometer and keep it at 1 g.
  nx += GRAVITY_BLEND * (ax - nx);
  ny += GRAVITY_BLEND * (ay - ny);
  nz += GRAVITY_BLEND * (az - nz);
  float n = sqrtf(nx * nx + ny * ny + nz * nz);
  if (n > 1e-3f) { gx_est = nx / n; gy_est = ny / n; gz_est = nz / n; }

  float dx = ax - gx_est, dy = ay - gy_est, dz = az - gz_est;
  float dyn2 = dx * dx + dy * dy + dz * dz;           // g^2
  float rot2 = wx * wx + wy * wy + wz * wz;           // (deg/s)^2
  float a = dt / RMS_TAU_S;
  msAccel += a * (dyn2 - msAccel);
  msGyro += a * (rot2 - msGyro);

  float raw = sqrtf(msAccel) + sqrtf(msGyro) / GYRO_PER_G;
  float tau = raw > envelope ? ATTACK_TAU_S : RELEASE_TAU_S;
  envelope += (dt / tau) * (raw - envelope);

  float s = (envelope - REST_LEVEL) / (SHAKE_MAX - REST_LEVEL);
  shakeOut = s < 0 ? 0 : (s > 1 ? 1 : s);
}

void setup() {
  Wire.begin();
  Wire.setClock(400000);
  sensorOk = mpuInit();
  lastInitTryMs = millis();
  lastSampleUs = micros();

  Bridge.begin();
  Bridge.provide("get_shake", get_shake);
}

void loop() {
  if (!sensorOk) {
    // Not answering (unplugged?): retry once a second.
    if (millis() - lastInitTryMs > 1000) {
      lastInitTryMs = millis();
      sensorOk = mpuInit();
      failCount = 0;
    }
    delay(5);
    return;
  }

  unsigned long now = micros();
  if (now - lastSampleUs < SAMPLE_US) return;
  float dt = (now - lastSampleUs) * 1e-6f;
  lastSampleUs = now;
  if (dt > 0.05f) dt = 0.05f;

  float ax, ay, az, wx, wy, wz;
  if (!mpuRead(ax, ay, az, wx, wy, wz)) {
    if (++failCount > 10) sensorOk = false;
    return;
  }
  failCount = 0;
  update(dt, ax, ay, az, wx, wy, wz);
}
