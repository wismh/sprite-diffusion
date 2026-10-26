# Sprite diffusion

A denoising diffusion model that draws 16x16 RGB sprites, trained on an NVIDIA GPU (CUDA), with a small app on the [Wind engine](https://github.com/wismh/wind-engine) to manage datasets and models and to watch a model train.

The network is a U-Net (16x16, 8x8, 4x4) with residual blocks, GroupNorm, SiLU and a sinusoidal time embedding; the noise follows the cosine schedule; new sprites come from DDIM on an exponential moving average of the weights. Everything, kernels included, is written here: no cuDNN, no cuBLAS.

## The app

- **Datasets.** Pick a folder of 16x16 pictures (PNG, BMP, TGA, JPG; subfolders are read too) and import it under a name. A pixel with alpha of 128 or more keeps its color, any other becomes magenta, so the sprites are opaque and the background marks what was transparent. The tab lists the datasets and shows their sprites, page by page.
- **Models.** Make a model: a name, a size (Small 32 channels, Medium 64, Large 96) and the dataset it learns from. Saved models are listed with their size, steps and dataset; open one to train it further, or delete it.
- **Generate.** Draws a number of sprites (1000, say) with a saved model and writes them as 16x16 PNG files in a new folder. The background can be made transparent (pixels close to magenta).
- **Filters.** Loads the marked model and shows what is inside it, one stage of the network at a time, a picture per channel: its answers to a sprite at a chosen noise level, its weights (the 3x3 filters), and its dream, the sprite it answers to hardest (found in the background by gradient ascent). Picking a channel shows all three, larger.
- **Process.** How sprites come out of noise (the steps of sampling, as the network's guess of the sprite or as the noisy input), the sprite of the dataset each result is closest to, and a check that draws 256 sprites and counts how many are closer to a dataset sprite than one in ten dataset sprites are to their nearest neighbour (about 10% for a model that does not copy).
- **Help.** A reference in short topics, from what a diffusion model is and how a sprite comes out of noise to the U-Net, training, the tabs of the program and why sprites come out as a mess, with formulas and two figures.
- **Two languages.** The window and the reference are in English and Ukrainian (the EN and UA buttons at the right of the tabs); the choice is remembered.
- **Train.** Start, pause, resume and stop the open model. The tab shows the step, the loss (a chart on a logarithmic scale) and a grid of sprites it draws now, refreshed every 250 steps or on demand. Training runs on a thread of its own and saves a checkpoint every 500 steps and when it stops; the model that was open comes back at the next start.

Files are in the per-user data directory (`%APPDATA%\wismh\diffusion`): `locale.txt` (the language), `datasets\*.spds`, `models\*.ckpt` (weights, the moving average, Adam's moments and the random generator: training goes on exactly where it stopped), `last_model.txt`.

## Build

Needs the Wind SDK (the version in `wind_project.toml`), the CUDA Toolkit and MSVC. The Wind editor builds and runs the game: open this folder as a project and press Play. For a standalone build:

```
cmake -S . -B build-export -G "Ninja Multi-Config" -DCMAKE_CONFIGURATION_TYPES=DebugGame;Release -DCMAKE_PREFIX_PATH=<path to the Wind SDK> -DWIND_EXPORT=ON
cmake --build build-export --config Release
```

`bin\Release\diffusion.exe` with its `assets` folder is the game; it needs only the NVIDIA driver.

## Tests and tools

```
cmake --preset tests -DCMAKE_PREFIX_PATH=<path to the Wind SDK>
cmake --build build-tests --config Release
ctest --test-dir build-tests -C Release
```

The tests (GoogleTest, taken from the Wind SDK; they need the GPU) cover the kernels (against CPU references and finite differences), the layers, the whole network, checkpoints, the trainer, the dataset and model files, the importer, the generator, the background jobs, the sessions of an open and of an inspected model, the numbers behind the Filters, Process and Train tabs (channel strengths, the Process grid, the loss chart axis, the small pictures) and the saved language. They are off by default so that building in the editor does not compile them.

Tools, built with the tests: `train_demo` (train on generated sprites and write sample grids), `bench` (time of a training step), `import_dataset` (the import of the Datasets tab from the command line).

## Layout

```
src/cuda/     The model: tensors, GEMM and convolution kernels, layers, U-Net, diffusion, trainer, checkpoints.
src/data/     Sprites in and out: datasets, importer, stores of datasets and models, PNG writer, the background job that the
              importer and the other long tasks run in.
src/inspect/  What is done with a trained model: drawing sprites to files, the copy check, the nearest sprite of the
              dataset, and the numbers and small pictures that the Filters, Process and Train tabs show.
src/view/     Painters and view models of the UI (a list of names, the paints).
src/          The app: game, session (an open model), inspection (a model loaded for looking into), the panels of the tabs.
assets/       UI (XML and CSS), the help topics and the messages in two languages (generated, see tools/).
tests/        Unit tests.
tools/        Command-line tools, and the Python scripts that write the messages (gen_locale.py) and the help (help/gen_help.py).
```

`src/cuda`, `src/data` and `src/inspect` are static libraries with no engine types in them; the game module and the tests link the same code. `src/session.cpp`, `src/inspect_session.cpp` and `src/loc/locale_prefs.cpp` have none either, and the tests compile them from there. The panels only turn what these give into text, pictures and buttons: a long task is a `data::BackgroundJob` (cancel, finished, one result with `cancelled` and `error`), and the number crunching of a tab is a plain function in `inspect/`.