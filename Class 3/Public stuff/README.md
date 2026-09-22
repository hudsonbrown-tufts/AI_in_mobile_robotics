# Public stuff

The public-facing part of the ME193 Robotics course code (Fall 2026) — the
subset of the class repo that's meant to be shared.

## What's here

- [useful libraries/](useful%20libraries/) — the shared `lelib.py` wrapper
  around `legoeducation`, plus a `main.py` template and a hardware smoke
  test. Start here if you're writing your own code for the LEGO Education
  hardware.
- [Kernels/](Kernels/) — a single-page, no-install tool for exploring image
  convolution: load an image, threshold it, then edit a 3×3 kernel and hit
  "Convolve" to see the effect. Open `index.html` directly in a browser.
- [Controls/](Controls/) — a live PD-control demo: a Single Motor acts as a
  hand-turned dial, and a Double Motor drives to match its position. Kp/Kd
  sliders let you watch overshoot and oscillation change in real time.
