# weight-space-galaxy10

A convolutional network for [Galaxy10 DECaLS](https://www.astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5) whose 475,161 weights were all written by Claude, with no gradient step anywhere. Claude designed galaxy-morphology measurements as conv weights and set every other number from closed-form statistics (quantiles, means, covariances, LDA/QDA) of activations returned by a budgeted harness.

It is an experiment in the spirit of "Sparks of Weight-Space Programming" (Takehi et al., MIT CSAIL, 2026), moved from editing a language model to building an image classifier from scratch.

| Model (≤ 500k parameters) | Test cross-entropy | Test accuracy |
| --- | --- | --- |
| Class prior | 2.220 | 14.9% |
| **Programmed CNN (v10)** | **0.804** | **73.7%** |
| Trained CNN (AdamW, 30 epochs) | 0.489 | 82.4% |

The programmed network closes 82% of the cross-entropy gap between guessing class frequencies and a gradient-trained CNN of the same size. The test split was scored once, after the design was frozen.

**[Read the technical report →](REPORT.md)** It covers the architecture, the ReLU constructions, all 151 measurements, the closed-form QDA head, the version history, the per-class comparison and the limitations.

## How it works

- **A fixed `nn.Sequential`.** The 21 layers use only whitelisted modules (Conv2d, Linear, ReLU, pooling, Flatten), so all behaviour lives in the weights.
- **Measurements.** A bank of 76 filters (light, colour, band-pass, derivatives, bar kernels) feeds 430 channels. Each channel is one weighted integral over a 27 × 27 map centred on the galaxy. Together they compute 151 morphology measurements: asymmetry, Fourier modes, log-spiral amplitudes, colour profiles, isophotal areas and more.
- **Classifier.** A piecewise-linear log of each feature, then a regularised QDA, all expressed with ReLU staircases.
- **The harness.** [`galaxy/wsp.py`](galaxy/wsp.py) is the only code that runs a network. It enforces the rules: no gradients or loss fitting, fewer than 500,000 parameters, 50 passes. Val is visible only as summary scores, and test only through a single `final`.

## Layout

| Path | What it is |
| --- | --- |
| [`REPORT.md`](REPORT.md) | the technical report |
| [`galaxy/wsp.py`](galaxy/wsp.py) | the harness: `check`, `acts`, `val`, `final`, `status` |
| [`galaxy/common.py`](galaxy/common.py), [`galaxy/prepare.py`](galaxy/prepare.py) | paths, class names, the challenge notebook's split, the image cache |
| [`galaxy/baselines.py`](galaxy/baselines.py) | class-prior and trained-CNN baselines |
| [`galaxy/editor/`](galaxy/editor) | Claude's weight builders, `v1.py` … `v10.py`; each version patches or extends the previous one |
| [`galaxy/runs/r1/log.jsonl`](galaxy/runs/r1/log.jsonl) | every pass of the run (network hash, scores) and the final |
| [`galaxy/runs/r1/submission/`](galaxy/runs/r1/submission) | the submitted network: `arch.json` + `submission.pt` |
| `galaxy/runs/r1/acts/pass047_layer8.npz`, `pass048_layer14.npz` | the two activation passes v10's knots and head were computed from |
| [`galaxy/runs/baseline_cnn/`](galaxy/runs/baseline_cnn), [`baseline_prior/`](galaxy/runs/baseline_prior) | baseline scores, training log and the CNN's weights |

## Rebuild the submitted network (no data needed)

The editor scripts only need numpy and torch. These three commands rebuild v10 from the included activations and produce weights identical to `submission.pt`:

```bash
pip install -r requirements.txt
python galaxy/editor/v10.py build
python galaxy/editor/v10.py knots galaxy/runs/r1/acts/pass047_layer8.npz
python galaxy/editor/v10.py qda galaxy/runs/r1/acts/pass048_layer14.npz --alpha 0.5 --temp 0.65 --out v10_qda_a05_t065
```

The networks are written to `galaxy/nets/`, which git ignores.

## Run the harness

Scoring a network, or starting a fresh run, needs the dataset: 2.7 GB, about 6 GB once cached.

```bash
mkdir -p galaxy10_data
curl -L -o galaxy10_data/Galaxy10_DECals.h5 https://www.astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5
python galaxy/prepare.py                                  # image cache + the notebook's train/val/test split
python galaxy/wsp.py check galaxy/nets/v10_qda_a05_t065   # free: architecture and parameter count
python galaxy/wsp.py val galaxy/nets/v10_qda_a05_t065 --run mine   # 1 pass of a new run's 50
```

Set `GALAXY_DATA_DIR` to keep the data elsewhere. Run `r1` is closed (its `final` has been used), so new work goes in a new `--run`.

Tested with Python 3.13, torch 2.15, numpy 2.5 and h5py 3.16 on an Apple M4 (MPS); the harness falls back to the CPU.

## Data

Galaxy10 DECaLS (Henry Leung and Jo Bovy) combines images from the [DESI Legacy Imaging Surveys](https://www.legacysurvey.org/) with morphology labels from [Galaxy Zoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/) volunteers. The images are not included in this repository.

## License

The code, report and network weights are released under the [MIT License](LICENSE). The Galaxy10 DECaLS images are not part of this repository and keep their own terms.
