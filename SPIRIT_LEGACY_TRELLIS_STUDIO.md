# Spirit Legacy TRELLIS Studio

This adds a production-oriented browser GUI on top of the existing TRELLIS.2 ROCm fork without replacing `app.py`.

## Features

- Front, Back, Left, Right, Top, Bottom, Extra View 1, Extra View 2 file selectors.
- Source filenames do not matter; selected files are copied into the job folder and their original filenames are recorded.
- One image works as a single-image job. Two or more images use multi-image conditioning.
- Multi-image fusion modes: `multidiffusion` (averages all image predictions at each sampler step) and `stochastic` (cycles through images, lighter compute).
- Presets: Phase 8 Safe, Fast Preview, Game Asset, High-Res Master (Experimental), Ultra 1536 (Experimental), Custom.
- Advanced sampler, decimation, texture-size, token and seed controls.
- Every run saves `request.json`, `job.json`, `metadata.json`, `worker.log`, the normalized input copies, and `model.glb`.
- The generated GLB and metadata are returned through the browser.

## ROCm / R9700 behavior

The worker preserves the known-good Phase-8 environment: SDPA attention, `flex_gemm` sparse convolution, gfx1201, ROCm 7.14 paths, low-VRAM model movement, and releasing inference models before o-voxel GLB postprocessing.

Input images must be RGBA PNG files with meaningful transparency. This deliberately avoids the BiRefNet path that previously failed on this ROCm setup.

`Phase 8 Safe` is the verified 512 baseline. The 1024 and 1536 presets use the corresponding TRELLIS.2 cascade models but are marked experimental until validated on this exact R9700 installation.

## Pull and run on the TRELLIS machine

```bash
cd /home/foster/TRELLIS.2-ROCm
git pull origin rocm
chmod +x tools/start_spirit_legacy_gui.sh tools/spirit_legacy_worker.py tools/spirit_legacy_gui.py
./tools/start_spirit_legacy_gui.sh
```

Open:

```text
http://100.125.111.71:7860
```

## Windows launcher

`Start-TRELLIS-Studio.bat` connects to `100.125.111.71` over SSH, starts the web GUI if it is not already running, and opens the browser.

## Output layout

```text
/home/foster/trellis2-outputs/studio/<category>/<asset>_<timestamp>_<seed>/
```

Each asset run retains its inputs and exact settings for later reproduction.

## View semantics

The view labels are retained for organization and reproducibility. The multi-image conditioning itself operates on the supplied image set, so TRELLIS does not require the source files to be named `front.png`, `back.png`, and so on. Top, Bottom, and extra 3/4-angle images can participate in the same fusion path.
