# Dataset

The raw audio dataset is **not** included in this repository due to its large size (~32 GB).

## Download Instructions

This project uses the **ASVspoof 2019** dataset (Logical Access partition), which contains genuine and spoofed speech samples for anti-spoofing research.

### Option 1: Kaggle (Recommended)

1. Install the Kaggle CLI: `pip install kaggle`
2. Set up your `kaggle.json` API token (see [Kaggle docs](https://www.kaggle.com/docs/api))
3. Run:
   ```bash
   kaggle datasets download -d awsaf49/asvpoof-2019-dataset
   unzip asvpoof-2019-dataset.zip -d dataset/
   ```

### Option 2: Official ASVspoof Website

Download directly from [https://www.asvspoof.org/](https://www.asvspoof.org/)

---

## Required Directory Structure

After downloading, organize the data as follows so the `AudioDeepfakeDataset` class can find it:

```
dataset/
├── client_0/
│   ├── real/     ← bonafide audio files (.wav)
│   └── fake/     ← spoofed audio files (.wav)
├── client_1/
│   ├── real/
│   └── fake/
└── client_2/
    ├── real/
    └── fake/
```

Each `client_N/` subdirectory represents the private local dataset of one federated client. For a single-machine demo, you can use a single `dataset/real/` and `dataset/fake/` folder — the code will fall back to this automatically.

---

> **Privacy Note:** In a real deployment, these files would reside on separate physical edge devices and would **never** leave the local machine. Only model parameter updates are transmitted to the aggregation server.
