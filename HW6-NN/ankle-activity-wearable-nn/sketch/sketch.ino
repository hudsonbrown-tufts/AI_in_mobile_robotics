// Microcontroller side of the HW6 ankle activity wearable.
//
// Shared by Part 1 and Part 2. The sketch never classifies anything; it
//   1. reads the MPU-6050 at 200 Hz and packs it into 50 Hz sample rows
//      (python/main.py pulls them over the Bridge), and
//   2. scrolls the word python/main.py sends (STAND / WALK / RUN / BIKE) across
//      the 13x8 LED matrix.
//
// Wiring (GY-521 breakout, same as HW5): VCC -> 3.3V, GND -> GND,
// SDA -> SDA, SCL -> SCL (the header's I2C pins = Wire). AD0 low -> 0x68.
//
// One sample row = 12 int16 values, from 4 consecutive 200 Hz readings:
//   [0..5]  mean ax, ay, az, gx, gy, gz   (raw counts)  -> used by PART 2
//   [6..8]  max  ax, ay, az               (raw counts)  -> used by PART 1
//   [9..11] min  ax, ay, az               (raw counts)  -> used by PART 1
// PART 1 needs the max/min because a heel strike lasts only ~10-20 ms, and
// averaging it into a 50 Hz sample would shrink the peak it measures.
//
// Bridge:
//   get_samples() -> String. "E" if the MPU-6050 isn't answering. Otherwise
//                    3 chars sequence number of the first row, then up to
//                    MAX_ROWS_PER_REPLY rows of 12 values. Every value is
//                    (int16 + 32768) written as 3 base64 chars (6 bits each),
//                    so one reply stays under the 256-byte RPC buffer.
//   set_display(word, mode) -> scroll `word` (A-Z, 0-9, space, '-');
//                    mode 1 or 2 lights 1 or 2 dots in the bottom-left corner.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>

// ---------------------------------------------------------------- IMU --------
const uint8_t MPU_ADDR = 0x68;
const unsigned long READ_US = 5000;     // 200 Hz
const int READS_PER_ROW = 4;            // -> 50 Hz rows
const int VALUES_PER_ROW = 12;
const int ROW_BUFFER = 128;             // 2.5 s of rows if python falls behind
const int MAX_ROWS_PER_REPLY = 6;       // 3 + 6*12*3 = 219 chars per reply

bool sensorOk = false;
unsigned long lastReadUs = 0;
unsigned long lastInitTryMs = 0;
int failCount = 0;

int32_t sum[6];
int16_t accMax[3], accMin[3];
int readsInRow = 0;

int16_t rows[ROW_BUFFER][VALUES_PER_ROW];
int rowHead = 0;                         // index of the oldest row
int rowCount = 0;
uint16_t nextSeq = 0;                    // sequence number of the next row pushed

// ---------------------------------------------------------------- matrix -----
const int MATRIX_COLS = 13;
const int MATRIX_ROWS = 8;
const unsigned long SCROLL_MS = 75;     // ~13 columns/s
const int GAP_COLS = 5;                  // blank columns between repeats
const unsigned long LINK_TIMEOUT_MS = 3000;

Arduino_LED_Matrix matrix;
uint8_t frame[MATRIX_ROWS * MATRIX_COLS];

const int MAX_TEXT = 16;
char wantedText[MAX_TEXT + 1] = "HW6";   // last word from python
char shownText[MAX_TEXT + 1] = "";       // word currently on the strip
int displayMode = 0;
uint8_t strip[MAX_TEXT * 6 + GAP_COLS];  // one byte per column, bit 0 = top row
int stripLen = 0;
int scrollPos = 0;
unsigned long lastScrollMs = 0;
unsigned long lastBridgeMs = 0;

// 5x7 font, one byte per column, bit 0 = top row (classic glcdfont shapes).
const uint8_t FONT_DIGITS[10][5] = {
  {0x3E,0x51,0x49,0x45,0x3E}, {0x00,0x42,0x7F,0x40,0x00}, {0x42,0x61,0x51,0x49,0x46},
  {0x21,0x41,0x45,0x4B,0x31}, {0x18,0x14,0x12,0x7F,0x10}, {0x27,0x45,0x45,0x45,0x39},
  {0x3C,0x4A,0x49,0x49,0x30}, {0x01,0x71,0x09,0x05,0x03}, {0x36,0x49,0x49,0x49,0x36},
  {0x06,0x49,0x49,0x29,0x1E}};
const uint8_t FONT_LETTERS[26][5] = {
  {0x7E,0x11,0x11,0x11,0x7E}, {0x7F,0x49,0x49,0x49,0x36}, {0x3E,0x41,0x41,0x41,0x22},
  {0x7F,0x41,0x41,0x22,0x1C}, {0x7F,0x49,0x49,0x49,0x41}, {0x7F,0x09,0x09,0x09,0x01},
  {0x3E,0x41,0x49,0x49,0x7A}, {0x7F,0x08,0x08,0x08,0x7F}, {0x00,0x41,0x7F,0x41,0x00},
  {0x20,0x40,0x41,0x3F,0x01}, {0x7F,0x08,0x14,0x22,0x41}, {0x7F,0x40,0x40,0x40,0x40},
  {0x7F,0x02,0x0C,0x02,0x7F}, {0x7F,0x04,0x08,0x10,0x7F}, {0x3E,0x41,0x41,0x41,0x3E},
  {0x7F,0x09,0x09,0x09,0x06}, {0x3E,0x41,0x51,0x21,0x5E}, {0x7F,0x09,0x19,0x29,0x46},
  {0x46,0x49,0x49,0x49,0x31}, {0x01,0x01,0x7F,0x01,0x01}, {0x3F,0x40,0x40,0x40,0x3F},
  {0x1F,0x20,0x40,0x20,0x1F}, {0x3F,0x40,0x38,0x40,0x3F}, {0x63,0x14,0x08,0x14,0x63},
  {0x07,0x08,0x70,0x08,0x07}, {0x61,0x51,0x49,0x45,0x43}};
const uint8_t FONT_DASH[5] = {0x08,0x08,0x08,0x08,0x08};
const uint8_t FONT_BLANK[5] = {0, 0, 0, 0, 0};

const uint8_t* glyph(char c) {
  if (c >= 'a' && c <= 'z') c -= 32;
  if (c >= 'A' && c <= 'Z') return FONT_LETTERS[c - 'A'];
  if (c >= '0' && c <= '9') return FONT_DIGITS[c - '0'];
  if (c == '-') return FONT_DASH;
  return FONT_BLANK;
}

// ---------------------------------------------------------------- Bridge -----
const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

char* put3(char* p, uint16_t u) {
  p[0] = B64[(u >> 12) & 0x0F];
  p[1] = B64[(u >> 6) & 0x3F];
  p[2] = B64[u & 0x3F];
  return p + 3;
}

String get_samples() {
  lastBridgeMs = millis();
  if (!sensorOk) return String("E");
  char out[3 + MAX_ROWS_PER_REPLY * VALUES_PER_ROW * 3 + 1];
  char* p = put3(out, (uint16_t)(nextSeq - rowCount));
  int n = rowCount < MAX_ROWS_PER_REPLY ? rowCount : MAX_ROWS_PER_REPLY;
  for (int i = 0; i < n; i++) {
    for (int v = 0; v < VALUES_PER_ROW; v++) {
      p = put3(p, (uint16_t)((int32_t)rows[rowHead][v] + 32768));
    }
    rowHead = (rowHead + 1) % ROW_BUFFER;
    rowCount--;
  }
  *p = '\0';
  return String(out);
}

void set_display(String word, int mode) {
  lastBridgeMs = millis();
  word.toCharArray(wantedText, MAX_TEXT + 1);
  displayMode = mode;
}

// ---------------------------------------------------------------- IMU code ---
bool writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool mpuInit() {
  bool ok = writeReg(0x6B, 0x01);        // PWR_MGMT_1: wake, PLL with gyro X clock
  delay(10);
  ok = ok && writeReg(0x1A, 0x02);       // CONFIG: DLPF ~94 Hz (keeps heel-strike peaks)
  ok = ok && writeReg(0x19, 0x04);       // SMPLRT_DIV: 1 kHz / 5 = 200 Hz
  ok = ok && writeReg(0x1B, 0x18);       // GYRO_CONFIG: +-2000 deg/s (16.4 counts per deg/s)
  ok = ok && writeReg(0x1C, 0x18);       // ACCEL_CONFIG: +-16 g (2048 counts per g), running passes 8 g
  return ok;
}

// Burst-read accel and gyro as raw counts. False on an I2C error.
bool mpuRead(int16_t v[6]) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                      // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 14) != 14) return false;
  int16_t r[7];
  for (int i = 0; i < 7; i++) {
    uint8_t hi = Wire.read();
    uint8_t lo = Wire.read();
    r[i] = (int16_t)((hi << 8) | lo);
  }
  v[0] = r[0]; v[1] = r[1]; v[2] = r[2];  // r[3] is temperature
  v[3] = r[4]; v[4] = r[5]; v[5] = r[6];
  return true;
}

void pushRow() {
  if (rowCount == ROW_BUFFER) {          // python fell behind: drop the oldest
    rowHead = (rowHead + 1) % ROW_BUFFER;
    rowCount--;
  }
  int16_t* row = rows[(rowHead + rowCount) % ROW_BUFFER];
  for (int i = 0; i < 6; i++) row[i] = (int16_t)(sum[i] / READS_PER_ROW);
  for (int i = 0; i < 3; i++) { row[6 + i] = accMax[i]; row[9 + i] = accMin[i]; }
  rowCount++;
  nextSeq++;
}

void addReading(const int16_t v[6]) {
  if (readsInRow == 0) {
    for (int i = 0; i < 6; i++) sum[i] = 0;
    for (int i = 0; i < 3; i++) { accMax[i] = v[i]; accMin[i] = v[i]; }
  }
  for (int i = 0; i < 6; i++) sum[i] += v[i];
  for (int i = 0; i < 3; i++) {
    if (v[i] > accMax[i]) accMax[i] = v[i];
    if (v[i] < accMin[i]) accMin[i] = v[i];
  }
  if (++readsInRow == READS_PER_ROW) {
    readsInRow = 0;
    pushRow();
  }
}

void readImu() {
  if (!sensorOk) {
    // Not answering (unplugged?): retry once a second.
    if (millis() - lastInitTryMs > 1000) {
      lastInitTryMs = millis();
      sensorOk = mpuInit();
      failCount = 0;
      readsInRow = 0;
      lastReadUs = micros();
    }
    return;
  }
  unsigned long now = micros();
  if (now - lastReadUs < READ_US) return;
  // Keep a steady 200 Hz; if we fell far behind, resync instead of bursting.
  lastReadUs = (now - lastReadUs > 4 * READ_US) ? now : lastReadUs + READ_US;
  int16_t v[6];
  if (!mpuRead(v)) {
    if (++failCount > 10) sensorOk = false;
    return;
  }
  failCount = 0;
  addReading(v);
}

// ---------------------------------------------------------------- matrix code
void buildStrip(const char* text) {
  stripLen = 0;
  for (int i = 0; text[i] && i < MAX_TEXT; i++) {
    const uint8_t* g = glyph(text[i]);
    for (int c = 0; c < 5; c++) strip[stripLen++] = g[c];
    strip[stripLen++] = 0;               // 1 blank column between letters
  }
  for (int c = 0; c < GAP_COLS; c++) strip[stripLen++] = 0;
  scrollPos = stripLen - 1;              // first letter starts at column 1, readable at once
}

void updateMatrix() {
  const char* text = wantedText;
  if (!sensorOk) text = "NO IMU";
  else if (millis() - lastBridgeMs > LINK_TIMEOUT_MS) text = "WAIT";   // python app not running

  bool changed = strcmp(text, shownText) != 0;
  if (changed) {
    strncpy(shownText, text, MAX_TEXT);
    shownText[MAX_TEXT] = '\0';
    buildStrip(shownText);
  } else if (millis() - lastScrollMs < SCROLL_MS) {
    return;
  } else {
    scrollPos = (scrollPos + 1) % stripLen;
  }
  lastScrollMs = millis();

  memset(frame, 0, sizeof(frame));
  for (int c = 0; c < MATRIX_COLS; c++) {
    uint8_t bits = strip[(scrollPos + c) % stripLen];
    for (int r = 0; r < 7; r++) {
      if (bits & (1 << r)) frame[r * MATRIX_COLS + c] = 7;
    }
  }
  // Bottom row: which part is running (1 dot = Part 1, 2 dots = Part 2).
  if (displayMode >= 1) frame[7 * MATRIX_COLS + 0] = 7;
  if (displayMode >= 2) frame[7 * MATRIX_COLS + 2] = 7;
  matrix.draw(frame);
}

// ---------------------------------------------------------------- main -------
void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);            // brightness values 0-7
  matrix.clear();

  Wire.begin();
  Wire.setClock(400000);
  sensorOk = mpuInit();
  lastInitTryMs = millis();
  lastReadUs = micros();

  Bridge.begin();
  // provide_safe: handlers run in this loop's thread, so they never race
  // with readImu() over the row buffer.
  Bridge.provide_safe("get_samples", get_samples);
  Bridge.provide_safe("set_display", set_display);
}

void loop() {
  readImu();
  updateMatrix();
}
