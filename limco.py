import re
import unicodedata as ud
from collections import Counter
from typing import Optional, Union

import fire
import ginza
import jaconv
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

Num = Union[int, float]

# Regex patterns for remove_extra_spaces()
BLOCKS_JA = "".join(
    (
        "\u4E00-\u9FFF",  # CJK UNIFIED IDEOGRAPHS
        "\u3040-\u309F",  # HIRAGANA
        "\u30A0-\u30FF",  # KATAKANA
        "\u3000-\u303F",  # CJK SYMBOLS AND PUNCTUATION
        "\uFF00-\uFFEF",  # HALFWIDTH AND FULLWIDTH FORMS
    )
)
BLOCK_LT = "\u0000-\u007F"
PTN_SPC_BW_JA = re.compile(f"([{BLOCKS_JA}]) +?([{BLOCKS_JA}])")
PTN_SPC_LEFT_JA = re.compile(f"([{BLOCK_LT}]) +?([{BLOCKS_JA}])")
PTN_SPC_RIGHT_JA = re.compile(f"([{BLOCKS_JA}]) +?([{BLOCK_LT}])")

NLP = spacy.load("ja_ginza")


def remove_extra_spaces(text: str) -> str:
    # Taken from https://gist.github.com/hideaki-t/198898f44aab078ed1a1#file-normalize_neologd-py-L17
    for ptn in [PTN_SPC_BW_JA, PTN_SPC_LEFT_JA, PTN_SPC_RIGHT_JA]:
        text = ptn.sub(r"\1\2", text)
    return text


def normalise(text: str, preserve_newlines=False) -> str:
    """Normalise Japanese text.

    Args:
        text (str): Japanese text to normalise.
        preserve_newlines (bool, optional):
            Whether to preserve newlines.
            Enable if the input is formatted like 'sentence-per-line'
            (e.g. when the original sentence do not end with punctuation).
            Defaults to False.
    """
    if not preserve_newlines:
        text = text.replace("\r", "")
        text = text.replace("\n", "")

    # Zenkaku alphabets, numbers, and signs to Hankaku
    # Hankaku Kana to Zenkaku
    text = jaconv.normalize(text, "NFKC")

    text = remove_extra_spaces(text)

    return text


def count_charcat(text: str) -> dict[str, int]:
    """Count number of characters in each Japanese character category."""
    c = Counter([ud.name(char).split()[0] for char in text])
    return {"hiragana": c["HIRAGANA"], "katakana": c["KATAKANA"], "kanji": c["CJK"]}


def count_conversations(text: str) -> dict[str, int]:
    """Count number of conversations in Japanese text."""
    return {
        "single": len(re.findall(r"「.+?」", text)),
        "double": len(re.findall(r"『.+?』", text)),
    }


def describe_sentence_lengths(doc: spacy.tokens.Doc) -> dict[str, Num]:
    """Calculate descriptive stats of sentence lengths (char counts)."""
    sentlens = [len(sent.text) for sent in doc.sents]
    return {
        "sentlen_mean": np.mean(sentlens),
        "sentlen_std": np.std(sentlens, ddof=1),
        "sentlen_min": np.min(sentlens),
        "sentlen_q1": np.quantile(sentlens, 0.25),
        "sentlen_med": np.median(sentlens),
        "sentlen_q3": np.quantile(sentlens, 0.75),
        "sentlen_max": np.max(sentlens),
    }


def measure_pos(
    doc: spacy.tokens.Doc,
    stopwords: list[str],
    awd: dict[str, float],
    jiwc: Optional[pd.DataFrame],
) -> dict[str, Num]:
    res = {}

    all_tokens = []
    total_tokens = 0
    total_verbs = 0
    total_nouns = 0
    total_adjs = 0
    total_advs = 0
    total_dets = 0
    for sent in doc.sents:
        for token in sent:
            if token.lemma_ in stopwords:
                continue
            all_tokens.append(token.lemma_)
            total_tokens += 1
            if token.pos_ == "VERB":
                total_verbs += 1
            elif token.pos_ == "NOUN":
                total_nouns += 1
            elif token.pos_ == "ADJ":
                total_adjs += 1
            elif token.pos_ == "ADV":
                total_advs += 1
            elif token.pos_ == "DET":
                total_dets += 1

    # CONTENT WORDS RATIO
    total_content_words = total_verbs + total_nouns + total_adjs
    res["cwr"] = np.divide(total_content_words, total_tokens)
    # NOTE: skip FUNCTION WORDS RATIO since it's equiv to 1 - CWR

    # Modifying words and verb ratio (MVR)
    total_modifying_words = total_adjs + total_advs + total_dets
    res["mvr"] = np.divide(total_modifying_words, total_verbs)

    # Named Entity Ratio
    res["pct_ne"] = np.divide(len(list(doc.ents)), total_tokens)

    res.update(calc_ttrs(all_tokens))
    if awd:
        res.update(score_abstractness(all_tokens, awd))
    if jiwc:
        res.update(score_jiwc(all_tokens, jiwc))

    return res


def calc_ttrs(tokens: list[str]) -> dict[str, Num]:
    cnt = Counter(tokens)
    Vn = len(cnt)
    logVn = np.log(Vn)
    N = np.sum(list(cnt.values()))
    logN = np.log(N)
    # TODO: implement frequency-wise TTR variants
    return {
        "ttr_orig": np.divide(Vn, N),  # original plain TTR: not robust to the length
        "ttr_guiraud_r": np.divide(Vn, np.sqrt(N)),  # Guiraud's R
        "ttr_herdan_ch": np.divide(logVn, logN),  # Herdan's C_H
        "ttr_rubet_k": np.divide(logVn, np.log(logN)),  # Rubet's k
        "ttr_maas_a2": np.divide((logN - logVn), (logN**2)),  # Maas's a^2
        "ttr_tuldava_ln": np.divide(
            (1 - (Vn**2)), ((Vn**2) * logN)
        ),  # Tuldava's LN
        "ttr_brunet_w": np.float_power(N, np.float_power(Vn, 0.172)),  # Brunet's W
        "ttr_dugast_u": np.divide((logN**2), (logN - logVn)),  # Dugast's U
    }


def score_abstractness(tokens: list[str], awd: dict[str, float]) -> dict[str, float]:
    scores = [awd.get(token, 0.0) for token in tokens]
    return {
        "abst_top5_mean": np.mean(sorted(scores, reverse=True)[:5]),
        "abst_max": max(scores),
    }


def score_jiwc(tokens: list[str], df_jiwc: pd.DataFrame) -> dict[str, float]:
    """calculate JIWC sentiment scores.

    Sentiment scores are normalized by the sum of all scores (i.e. softmax).
    """
    jiwc_words = list(set(tokens) & set(df_jiwc.index))
    jiwc_vals = df_jiwc.loc[jiwc_words].sum()
    return (
        (jiwc_vals / jiwc_vals.sum())
        .rename(
            {
                "Sad": "jiwc_sadness",
                "Anx": "jiwc_anxiety",
                "Anger": "jiwc_anger",
                "Hate": "jiwc_hatrid",
                "Trustful": "jiwc_trust",
                "S": "jiwc_surprise",
                "Happy": "jiwc_happiness",
            }
        )
        .to_dict()
    )


def count_taigendome(doc: spacy.tokens.Doc) -> int:
    """Count Japanese 体言止め (taigen-dome) sentences in a text."""
    pos_per_sent = [[token.pos_ for token in sent] for sent in doc.sents]
    return sum(is_taigendome(sent) for sent in pos_per_sent)


def is_taigendome(pos_list: list[str]) -> bool:
    """Check if a sentence is 体言止め (taigen-dome)."""
    if len(pos_list) < 2:
        return False

    if pos_list[-1] != "PUNCT":
        return pos_list[-1] == "NOUN"
    else:
        return pos_list[-2] == "NOUN"


# TODO: 荒牧先生の潜在語彙量も
# def calc_potentialvocab(text: str) -> float:
#     raise NotImplementedError


def describe_max_sent_depths(doc: spacy.tokens.Doc) -> dict[str, float]:
    """Describe the maximum depths of the sentences."""
    max_depths = [count_max_sent_depth(sent) for sent in doc.sents]
    return {
        "sentdepths_mean": np.mean(max_depths),
        "sentdepths_std": np.std(max_depths, ddof=1),
        "sentdepths_min": np.min(max_depths),
        "sentdepths_q1": np.quantile(max_depths, 0.25),
        "sentdepths_med": np.median(max_depths),
        "sentdepths_q3": np.quantile(max_depths, 0.75),
        "sentdepths_max": np.max(max_depths),
    }


def count_max_sent_depth(sent: spacy.tokens.Span) -> int:
    """Count the maximum number of depth of the sentence."""
    deps = {token.i: token.head.i for token in sent}
    depths = {i: _count_depth(0, i, deps) for i in deps}
    return max(depths.values())


def _count_depth(mem: int, tgt: int, deps: dict[int, int]) -> int:
    """A helper to count the maximum number of depth of the sentence."""
    if deps[tgt] == tgt:
        return mem
    else:
        return 1 + _count_depth(mem, deps[tgt], deps)


def describe_sent_chunks(doc: spacy.tokens.Doc) -> dict[str, float]:
    """Describe the number of chunks per sentence."""
    num_chunks = [len(list(ginza.bunsetu_spans(sent))) for sent in doc.sents]
    return {
        "sentchunks_mean": np.mean(num_chunks),
        "sentchunks_std": np.std(num_chunks, ddof=1),
        "sentchunks_min": np.min(num_chunks),
        "sentchunks_q1": np.quantile(num_chunks, 0.25),
        "sentchunks_med": np.median(num_chunks),
        "sentchunks_q3": np.quantile(num_chunks, 0.75),
        "sentchunks_max": np.max(num_chunks),
    }


def describe_chunk_tokens(doc: spacy.tokens.Doc) -> dict[str, float]:
    """Describe the number of tokens per chunk."""
    num_tokens = [len(span) for sent in doc.sents for span in ginza.bunsetu_spans(sent)]
    return {
        "chunktokens_mean": np.mean(num_tokens),
        "chunktokens_std": np.std(num_tokens, ddof=1),
        "chunktokens_min": np.min(num_tokens),
        "chunktokens_q1": np.quantile(num_tokens, 0.25),
        "chunktokens_med": np.median(num_tokens),
        "chunktokens_q3": np.quantile(num_tokens, 0.75),
        "chunktokens_max": np.max(num_tokens),
    }


def calculate_all(
    text: str, stopwords: list[str], awd: dict[str, float], jiwc: Optional[pd.DataFrame]
) -> dict[str, Num]:
    res = {}
    text = normalise(text)

    # calculatable without spacy
    num_chars = len(text)
    num_charcats = count_charcat(text)
    res["pct_hiragana"] = np.divide(num_charcats["hiragana"], num_chars)
    res["pct_katakana"] = np.divide(num_charcats["katakana"], num_chars)
    res["pct_kanji"] = np.divide(num_charcats["kanji"], num_chars)
    num_convs = count_conversations(text)

    # calculation requiring spacy
    doc = NLP(text)

    res["num_sents"] = len(list(doc.sents))
    res["pct_convs"] = np.divide(sum(num_convs.values()), res["num_sents"])
    res["pct_taigen"] = np.divide(count_taigendome(doc), res["num_sents"])
    res.update(describe_sentence_lengths(doc))
    res.update(measure_pos(doc, stopwords, awd, jiwc))

    # 文節 processing
    res.update(describe_max_sent_depths(doc))
    res.update(describe_sent_chunks(doc))
    res.update(describe_chunk_tokens(doc))

    return res


def from_df(
    df: pd.DataFrame,
    col: str,
    stopwords: Optional[list[str]] = None,
    awds: Optional[dict[str, float]] = None,
    df_jiwc: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    assert col in df.columns, f"{col} is not found in the input data"

    tqdm.pandas(total=len(df))
    return df.progress_apply(
        lambda row: calculate_all(row[col], stopwords, awds, df_jiwc),
        result_type="expand",
        axis=1,
    )


def from_file(
    fname: str, col: str, sw: str = None, awd: str = None, jiwc: str = None
) -> None:
    """Linguistic Measure Collection

    Calculate all linguistic measures for each text in the given file and write the results to an extended CSV file named `<input_file_path>.limco.csv`.

    Args:
        fname (str): File path to the input file. CSV and Excel files are supported.
        col (str): Column name of the text to be analysed.
        sw (str, optional): Path to stopwords, where words should be listed one per line. Defaults to None.
        awd (str, optional): Path to Abstruct Word Dictionary published by SocioCom. Defaults to None.
        jiwc (str, optional): Path to JIWC published by SocioCom. Defaults to None.

    Raises:
        ValueError: arise when you specify an unsupported file format.
    """
    if fname.endswith(".csv"):
        df = pd.read_csv(fname)
    elif fname.endswith(".xls") or fname.endswith(".xlsx"):
        df = pd.read_excel(fname)
    else:
        raise ValueError("Unsupported input format: please use CSV or Excel data")

    if sw:
        with open(sw, "r") as f:
            stopwords = [line.strip() for line in f]
    else:
        stopwords = []

    if awd:
        with open(awd, "r") as f:
            rows = [line.strip().split("\t") for line in f]
            awds = {word: float(score) for word, score, _, _ in rows}
    else:
        awds = {}

    if jiwc:
        df_jiwc = pd.read_csv(jiwc, index_col=1).drop(columns="Unnamed: 0")
    else:
        df_jiwc = None

    df_limco = from_df(df, col, stopwords, awds, df_jiwc)

    pd.concat(
        [df, df_limco],
        axis=1,
    ).to_csv(f"{fname}.limco.csv", index=False)


def main():
    fire.Fire(from_file)


if __name__ == "__main__":
    main()
