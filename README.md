# Chest X-Ray Anomaly Detector

A small Jupyter app that classifies a chest X-ray as normal or pneumonia using a fastai image classifier.

Personal deep learning practice project. This repo contains the finished model and the notebook that serves it; the training notebook is not included.

## What it does

`production.ipynb` loads the exported model `export.pkl` and shows a simple widget UI: upload a scan, click "Classify", and it displays a thumbnail with the predicted label and its probability.

## How it works

- Model: a fastai (v2) `cnn_learner` built on a ResNet-18 backbone, exported with `learn.export()`. The call arguments recorded inside `export.pkl` show it was trained with `fine_tune`, using random resized crops and flips as training augmentation.
- Classes: `NORMAL` and `PNEUMONIA`. The labels come from the parent folder name of each image, and the image file names stored in the export match the Kaggle "Chest X-Ray Images (Pneumonia)" dataset.
- Inference code (in the notebook):
  1. `load_learner('export.pkl')`
  2. `ipywidgets` `FileUpload` and `Button` controls
  3. on click, the uploaded bytes are opened with `PILImage.create`, passed to `learn_inf.predict`, and the result is printed as `Prediction: <class>; Probability: <value>`.
- Accuracy and error rate are not recorded in the notebook, so no performance figures are given here.

## Status

The saved notebook has cells for the upload button, the classify button and the click handler, but the cells that define the output area (`out_pl`) and the prediction label (`lbl_pred`) are missing from it. Add these before the final cell:

```python
out_pl = widgets.Output()
lbl_pred = widgets.Label()
```

## Tech

Python 3, fastai v2, PyTorch (ResNet-18), ipywidgets, Jupyter.

## Run it

```
pip install fastai jupyter ipywidgets
jupyter notebook production.ipynb
```

Run the cells from the top with `export.pkl` in the same folder (about 47 MB). `load_learner` unpickles the file, so only use an `export.pkl` you trust. If the widgets do not render in classic Jupyter, enable them with `jupyter nbextension enable --py widgetsnbextension`.

For research and learning only; this is not a medical tool.
