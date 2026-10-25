# Embedded-Readiness Resource Budget (Estimated)

> **Disclosure:** All figures below are analytical embedded resource estimates for deploying the 1.0 s fast-path pipeline on an **STM32F407 / STM32G4 (Cortex-M4F @ 168 MHz)** or **ESP32-S3 (@ 240 MHz)** microcontroller. Host Python execution time per 1-second window was measured at **2.74 ms** by `run_all.py`.

## 1. Memory Footprint (SRAM & Flash)

| Buffer / Component | Calculation | Size (KB) | Target Limit |
|---|---|---|---|
| **Acoustic Ring Buffer** (16 kS/s, 1.0 s, `int16`) | `16,000 * 2 B` | 32.0 KB | |
| **Hall 6-Channel Buffer** (10 kS/s, 1.0 s, decimated 10x to 1 kS/s for fe & sidebands, `int16`) | `6 * 1,000 * 2 B` | 12.0 KB | |
| **FFT / Hilbert Scratchpad** (4096-pt real FFT in-place `float32`) | `4,096 * 4 B * 2` | 32.0 KB | |
| **Random Forest / Quantized Classifier** (80 trees, depth 7, `int8`/struct nodes) | `~1,800 nodes * 12 B` | 21.6 KB (Flash) | <= 100 KB |
| **Total Active SRAM** | Buffers + Scratchpad + Stack | **~82.0 KB** | **<= 256 KB** |
| **Total Flash (Firmware + CMSIS-DSP + Classifier)** | Code + Constants + Model | **~96.0 KB** | **<= 1,024 KB** |

## 2. Compute Latency per 1.0 s Window (Cortex-M4F @ 168 MHz)

| DSP Stage | Operations | Est. Cycles | Est. Time (ms) |
|---|---|---|---|
| **Hall Phasor Projection & Spatial-DFT** | 6 ch x 1,000 MACs + 6-pt DFT | ~45,000 | 0.27 ms |
| **Acoustic Band-Pass + Envelope + 16k FFT** | 4th-order IIR + Radix-2 RFFT | ~2,900,000 | 17.26 ms |
| **Order Axis Cubic Interpolation** | 2,000 order bins | ~120,000 | 0.71 ms |
| **Feature Extraction + Classifier Inference** | 11 features + 80 trees (depth 7) | ~15,000 | 0.09 ms |
| **Total Compute Latency per Window** | | **~3.08M cycles** | **~18.33 ms (< 2% CPU duty)** |
