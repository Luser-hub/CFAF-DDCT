# CFAF-DDCT: Channel-Feature Attention Fusion based Dual-domain Cooperative Transfer for Unilateral Upper Limb Fine Motor MI-BCI Decoding

Implementation of Channel-Feature Attention Fusion based Dual-domain Cooperative Transfer for three-class EEG decoding.

## Features

- Sliding-window absolute Pearson channel scoring and hard voting
- Group and individual channel-weight fusion
- Top-K channel selection and channel-weighted feature extraction
- Confidence-based source sample selection
- Temporal convolution and channel-feature attention fusion
- ICL, MMD, LMMD, CORAL, ADVER, and classification-only training
- Synthetic source and target domain data generator

## Project Structure

```text
CFAF-DDCT-main/
  main.py
  synthetic_data.py
  cfaf_ddct/
    config.py
    data.py
    losses.py
    model.py
    train.py
  tests/
    test_channel_scoring.py
    test_model.py
    test_source_selection.py
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Data Generation

```bash
python synthetic_data.py --output-dir outputs
```

The generated file contains 15 source domains and one target domain. EEG arrays use the shape `[trials, channels, time]`.

## Training

```bash
python main.py --loss icl --epochs 200 --data-dir outputs
```

Available losses:

```text
none, icl, mmd, lmmd, coral, adver
```

Common options:

```text
--epochs {180,200}
--gamma FLOAT
--seed INTEGER
--device {cpu,cuda}
--data-dir PATH
--output-dir PATH
--no-normalize-transfer-features
```

Training progress is printed for every epoch. Metrics, selected channels, channel weights, source sample indices, and loss histories are saved to `result.json`.

## Tests

```bash
python -m unittest discover -s tests -v
```
