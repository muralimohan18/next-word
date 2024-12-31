import os
import base64  # Import base64 for encoding images
import streamlit as st
import torch
import string
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, DistilBertTokenizer, DistilBertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel

# Inject custom CSS to set the background image and text color
def set_background(image_path):
    """Sets a background image for the Streamlit app."""
    if os.path.isfile(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url('data:image/jpeg;base64,{encoded_string}');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            .stTextArea textarea {{
                color: #1c1c1c;  /* Dark black color */
            }}
            .stTextArea label {{
                color: #1c1c1c;  /* Dark black color for label */
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error(f"Background image not found at: {image_path}")

# Define the function to load the model
@st.cache_resource
def load_model(model_name):
    """Load the tokenizer and model based on the selected model name."""
    try:
        if model_name == 'BERT':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        elif model_name == 'RoBERTa':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaForMaskedLM.from_pretrained('roberta-base')
        elif model_name == 'DistilBERT':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        elif model_name == 'GPT-2':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            st.warning(f"Model {model_name} not supported. Using BERT as default.")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')

        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to decode the predictions from the model
def decode(tokenizer, pred_idx, top_clean):
    """Decode the top predictions into readable text."""
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])

# Function to encode the input text
def encode(tokenizer, text_sentence, add_special_tokens=True):
    """Encode the input sentence with the tokenizer."""
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

# Function to get all predictions
def get_all_predictions(text_sentence, top_k, top_clean=5, tokenizer=None, model=None):
    """Get top-k predictions for the given sentence."""
    input_ids, mask_idx = encode(tokenizer, text_sentence)
    with torch.no_grad():
        predict = model(input_ids)[0]
    bert = decode(tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    return {'predictions': bert}

# Function to handle predictions for end-of-sentence input
def get_prediction_eos(input_text, top_k, tokenizer, model):
    """Append <mask> token to the input text and get predictions."""
    try:
        input_text += ' <mask>'
        res = get_all_predictions(input_text, top_k, tokenizer=tokenizer, model=model)
        return res
    except Exception as error:
        st.error(f"Error during prediction: {error}")
        return None

# Main Streamlit app logic
st.title("Next Word Prediction using Pretrained Models")

# Set background image
background_image_path = r"img.jpeg"  # Update this path to your image
set_background(background_image_path)

st.sidebar.text("Next Word Prediction")

# Sidebar options
top_k = st.sidebar.slider("How many words do you need?", 1, 25, 5)  # Adjust the range and default value as needed
model_name = st.sidebar.selectbox(
    label='Select Model to Apply', 
    options=['BERT', 'RoBERTa', 'DistilBERT', 'GPT-2'], 
    index=0, 
    key="model_name"
)

# Load the selected model and tokenizer
tokenizer, model = load_model(model_name)

if tokenizer and model:
    # Display the label in bold and dark black for the text area
    st.markdown("**Enter your text here:**")  # Making the label bold

    # Get user input
    input_text = st.text_area("", placeholder="Type a sentence and click outside to get predictions.")

    if input_text:
        # Display a spinner while loading predictions
        with st.spinner("Generating predictions..."):
            res = get_prediction_eos(input_text, top_k, tokenizer=tokenizer, model=model)

        if res:
            # Display predictions
            answer = res['predictions'].split("\n")
            answer_as_string = "    ".join(answer)
            st.text_area("Predicted List is Here", answer_as_string, key="predicted_list")
else:
    st.error("Unable to load the model. Please check the model name or configurations.")
