// Microcontroller side of the UNO Q minifig tracker.
//
// The Linux side (python/main.py) calls these over the Bridge:
//   set_pixel(col, row) -> light exactly one pixel on the 13x8 matrix
//   clear_matrix()      -> turn every pixel off
// If no update arrives for STALE_TIMEOUT_MS, the matrix clears itself so a
// stale pixel doesn't stay lit when the computer stops sending.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

const int MATRIX_COLS = 13;
const int MATRIX_ROWS = 8;
const unsigned long STALE_TIMEOUT_MS = 5000;

Arduino_LED_Matrix matrix;
uint8_t frame[MATRIX_ROWS * MATRIX_COLS];

// Written by the Bridge handlers, read by loop(). The handlers only record
// the request; loop() does all the drawing so the matrix is touched from
// one place.
volatile int targetCol = -1;  // -1 = nothing lit
volatile int targetRow = -1;
volatile bool dirty = true;
volatile unsigned long lastUpdateMs = 0;

void set_pixel(int col, int row) {
  if (col < 0 || col >= MATRIX_COLS || row < 0 || row >= MATRIX_ROWS) return;
  targetCol = col;
  targetRow = row;
  lastUpdateMs = millis();
  dirty = true;
}

void clear_matrix() {
  targetCol = -1;
  targetRow = -1;
  lastUpdateMs = millis();
  dirty = true;
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // brightness values 0-7
  matrix.clear();
  Bridge.begin();
  Bridge.provide("set_pixel", set_pixel);
  Bridge.provide("clear_matrix", clear_matrix);
}

void loop() {
  if (targetCol >= 0 && millis() - lastUpdateMs > STALE_TIMEOUT_MS) {
    targetCol = -1;
    targetRow = -1;
    dirty = true;
  }

  if (dirty) {
    dirty = false;
    memset(frame, 0, sizeof(frame));
    if (targetCol >= 0) {
      frame[targetRow * MATRIX_COLS + targetCol] = 7;  // full brightness (0-7)
    }
    matrix.draw(frame);
  }

  delay(10);
}
