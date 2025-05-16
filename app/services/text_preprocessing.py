import pandas as pd
import re
import html
import warnings
import random
import nltk.tokenize
from unidecode import unidecode
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# NLTK downloads
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# Google Cloud Translate setup
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =config.PATH_GOOGLE_APPLICATION_CREDENTIALS
#translator = translate.Client()


def clean_BERT_mBERT(text):
    """Nettoie le texte en supprimant les balises HTML, entités, accents et caractères spéciaux."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = html.unescape(text)
    text = re.sub(r"[^a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def translate_column(df, num_workers=10):
    df["txt"] = df["txt"].fillna("").astype(str)
    texts = df["txt"].tolist()
    translated_results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(translate_text, text): i for i, text in enumerate(texts)}
        for future in as_completed(futures):
            translated_results[futures[future]] = future.result()
    df["txt"] = [translated_results[i] for i in sorted(translated_results)]
    return df

def clean_translation(text):
    """Nettoie le texte en supprimant les accents, les majuscules, les nombres isolés et les caractères spéciaux."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_duplicates(df):
    duplicate_mask = df.duplicated(subset='txt', keep='first')
    df = df[~duplicate_mask]
    df = df.assign(count=df.groupby("txt")["prdtypecode"].transform(lambda x: x.value_counts()))
    df = df.drop_duplicates("txt", keep="first").drop(columns="count")
    df = df.drop_duplicates(subset=["txt", "prdtypecode"], keep="first")
    return df

def preprocess_txt(df):
    df["txt"] = df["designation"].fillna("").astype(str) + " " + df["description"].fillna("")
    df["txt"] = df["txt"].apply(clean_BERT_mBERT)
    # df = translate_column(df)  # optionnel si non nécessaire
    return df["txt"].apply(clean_translation)


def get_random_row_pandas_sample(df):
    if df.empty:
        return pd.DataFrame()  # Return an empty DataFrame
    else:
        random_index = random.randint(0, len(df) - 1)
        return df.iloc[[random_index]] # Return a DataFrame containing the row
