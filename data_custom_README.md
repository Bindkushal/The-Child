# data/custom/ — Your Own Handwritten Samples

Put your own handwritten letter and digit photos here.
They automatically mix into training alongside EMNIST.

---

## Folder Structure

```
data/custom/
├── A/
│   ├── my_a_1.jpg
│   ├── my_a_2.jpg
│   └── ...
├── B/
│   └── my_b_1.png
├── C/
...
├── Z/
├── 0/
├── 1/
...
└── 9/
```

One folder per character. Name the files anything you like.

---

## How to Take Good Photos

- White paper, dark pen or marker
- Letter large and centered, filling most of the frame
- Good lighting — no shadows across the letter
- Photo taken straight on, not at an angle
- Crop close to the letter before saving

---

## Supported Formats

jpg, jpeg, png, bmp

---

## What Happens Automatically

1. Image is converted to grayscale
2. Colors are inverted (white letter on black background — matches EMNIST)
3. Resized to 28x28 pixels
4. Normalized to match EMNIST scale
5. Mixed into training batch alongside EMNIST samples

Your samples are weighted equally to EMNIST samples,
so even a few photos per letter make a real difference.

---

## Adding via Colab

Use `collect_data.py` to upload and label photos directly
from your phone inside the Colab notebook.
