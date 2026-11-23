# IMDB Sentiment Analysis with a Vanilla RNN

This project trains a PyTorch Vanilla RNN on the IMDB movie-review dataset and
provides a Streamlit app for predicting the sentiment of a review.

## Project files

```text
RNN/
├── Dataset/
│   └── IMDB Dataset.csv
├── Model/
│   └── imdb_rnn_model.pth
├── app.py
├── helper.py
├── model.py
├── preprocessing.py
├── requirements.txt
└── train.py
```

The two original notebooks are retained as exploratory/legacy material. The
Python scripts above are the repeatable training and prediction workflow.

## Setup (Windows PowerShell)

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The tokenizer uses spaCy's built-in English tokenizer; no separate language
model download is needed.

## Train and evaluate

Ensure `Dataset\IMDB Dataset.csv` contains the `review` and `sentiment`
columns, then run:

```powershell
python train.py
```

Training creates a stratified training, validation, and test split. It builds
the vocabulary from training reviews only, selects the best checkpoint by
validation accuracy, and reports final test loss and accuracy. The checkpoint
is written to `Model\imdb_rnn_model.pth`.

## Run the prediction app

The project includes a trained checkpoint at `Model\imdb_rnn_model.pth`. Start
the app with:

```powershell
streamlit run app.py
```

Paste a movie review into the page and select **Analyze sentiment**. The app
uses the checkpoint's vocabulary and model dimensions, and displays the
predicted class with its model score. To refresh the model, run `python
train.py` first.

## Preprocessing and model

Reviews have HTML tags and URLs removed, chat abbreviations expanded, and are
lowercased before spaCy tokenization. The prediction app uses the same process
as training. The model uses a 100-dimensional embedding, a 128-unit vanilla
RNN, and a sigmoid probability derived from its binary-classification logit.
