# SpamGuard AI

A single-page SMS/email spam detection application built with Streamlit and a
saved Scikit-learn text-classification pipeline.

## Files

- `app.py` — Streamlit user interface and prediction workflow.
- `spam_detection.py` — training script using the UCI SMS Spam Collection.
- `spam_detection_model.pkl` — trained TF-IDF + Multinomial Naive Bayes pipeline.
- `requirements.txt` — Python dependencies.

## Run the application

Use Python 3.10 or newer.

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, you can install and run without activating:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep `app.py` and `spam_detection_model.pkl` in the same folder.

## Model

The included model is a saved Scikit-learn Pipeline with TF-IDF text
vectorization and Multinomial Naive Bayes. The application loads this existing
model and uses `predict()` and `predict_proba()` for live user-entered messages.
It does not retrain the model when the app starts.

If you intentionally need to train a new model, run:

```bash
python spam_detection.py
```

The training script downloads the UCI SMS Spam Collection dataset and writes
`spam_detection_model.pkl` in its working directory. This replaces the
existing model file, so keep a backup if you want to preserve the current model.

## Notes

- `ham` is displayed as `NOT SPAM`.
- Model confidence is the larger of the two class probabilities.
- Probabilities are model outputs, not guarantees of safety.
- The training script requires an internet connection to download the dataset.
