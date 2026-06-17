# Workflow Preferences

- **Run experiments in tmux** — no timeouts, can check progress async
- **Render small batches for testing** — e.g. 6x6 grid (36 images), not 10x10 (100)
- **Show results as a grid image** — stitch cells into a single PNG and open it
- **Use Cycles with 8 samples + OIDN denoiser** — fast (~3s/frame) and good enough quality
- **Use Metal GPU** on macOS for Cycles
- **Don't save .blend files** — just render images to consume
- **Pack textures** if saving blends (avoids purple missing textures)
- **No mesh smoothing on robot** — keep flat shading, blocky sim look
