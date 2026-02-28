![Bird-CNN Banner](docs/assets/banner.png)
![License](https://img.shields.io/badge/License-MIT-blue.svg) ![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

A custom built Convolutional Neural Network designed to identify bird species directly in the browser.

**The current supported species are:**
  * American Robin
  * Bald Eagle
  * Blue Jay
    More to come!

**Performance**
  Thanks to an optimized CuPy backend, the training engine can achieve massive throughput:
  **Images:** 600
  **Batch Size:** 64
  **Total Operations:** 60000 images processed
  **Total Time:** 67.06 seconds
  **Throughput:** 895 images/sec

**Technical Deep Dive**
  The model architecture is a 4-layer CNN built from scratch. (no framework like TensorFlow/PyTorch used)
  Is this better? Likely not, but I still learned a lot.

  1. Feature Extraction
    Uses custom kernels to detect edges, textures, and colors.

  2. Activation
    ReLU (Rectified Linear Unit) filters out noise by passing only positive features.

  3. Distribution
     Softmax converts raw logits into human-readable probabilities.


**Project Structure**
  * `training/main.py`: The GPU accelerated training script.
  * `docs/`: The web-facing application.
  * `docs/model/model_weights.json`: The pre-trained "brain" of the operation.
  * `data/`: Placeholder folder for data organization.

**How to Run**
  **1. Web:** Simply go to PLACEHOLDER URL in your browser.
  **2. Training/Local:** Install dependencies (listed in `requirements.txt`
    - Place images in `data/[Species Name]/`
    - Run `training/main.py`
