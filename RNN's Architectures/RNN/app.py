from pathlib import Path

import streamlit as st
import torch

from model import RNN
from preprocessing import tokenize_review


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "Model" / "imdb_rnn_model.pth"

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {CHECKPOINT_PATH}. "
            "Run `python train.py` to train and save it."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    required_keys = {
        "model_state_dict",
        "vocab_size",
        "embedding_dim",
        "hidden_dim",
        "max_length",
        "word2idx",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            "Model checkpoint is missing required fields: "
            + ", ".join(sorted(missing_keys))
        )

    model = RNN(
        checkpoint["vocab_size"],
        checkpoint["embedding_dim"],
        checkpoint["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["word2idx"], checkpoint["max_length"], device


st.title("🎬 IMDB Movie Review Sentiment")
st.write(
    "Paste a movie review to estimate whether its sentiment is positive or negative."
)

try:
    model, word2idx, max_length, device = load_model()
except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
    st.error(str(error))
    st.stop()

with st.form("sentiment_form"):
    review = st.text_area(
        "Movie review",
        placeholder="This movie was beautifully written...",
        height=200,
        max_chars=20_000,
    )
    submitted = st.form_submit_button("Analyze sentiment", type="primary")

if submitted:
    if not review.strip():
        st.warning("Enter a review before analyzing it.")
    else:
        tokens = tokenize_review(review)
        token_ids = [
            word2idx.get(token, word2idx["<UNK>"])
            for token in tokens[:max_length]
        ]

        if not token_ids:
            st.warning("The review did not contain any tokens the model can analyze.")
        else:
            inputs = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
            lengths = torch.tensor([len(token_ids)], dtype=torch.long)

            with torch.inference_mode():
                probability_positive = torch.sigmoid(
                    model(inputs, lengths)
                ).item()

            if probability_positive >= 0.5:
                sentiment = "Positive"
                confidence = probability_positive
            else:
                sentiment = "Negative"
                confidence = 1.0 - probability_positive

            st.subheader(f"Predicted sentiment: {sentiment}")
            st.metric("Model score for this prediction", f"{confidence:.1%}")
            st.progress(confidence)
            st.caption(
                "This is the model's score, not a calibrated probability. "
                "Predictions may be less reliable for sarcasm, mixed reviews, "
                "or text unlike the IMDB training data."
            )
