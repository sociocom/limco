from collections import Counter
import os
import re
from typing import List, Tuple, Dict, Union
import unicodedata as ud

import fire
from natto import MeCab
import pandas as pd
import numpy as np

Num = Union[int, float]

# TODO: efficiency
# preprocess mecab tokenisation first for library usage

# TODO: make these resources assignable by the user
# TODO: make these resources optional
# DELIM_SENT = re.compile(r"(?:[。？！\?\!]+|[。？！\?\!]?[」』])")  # FIXME: 文じゃないカギカッコが取れちゃう
NM = MeCab()  # NOTE: assume IPADIC
NMN = MeCab("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
with open(os.path.join(os.path.dirname(__file__), "./data/stopwords_jp.txt"), "r") as f:
    STOPWORDS_JP = [line.strip() for line in f]
STOPPOS_JP = ["形容動詞語幹", "副詞可能", "代名詞", "ナイ形容詞語幹", "特殊", "数", "接尾", "非自立"]
with open(os.path.expanduser("./data/AWD-J_EX.txt"), "r") as f:
    rows = [line.strip().split("\t") for line in f]
    AWD = {word: score for word, score, _, _ in rows}
DF_jiwc = pd.read_csv(os.path.expanduser("./data/2017-11-JIWC.csv"), index_col=1).drop(
    columns="Unnamed: 0"
)


def measure_sents(text: str) -> np.ndarray:
    """Show descriptive stats of sentence length.

    input text should be one sentence per line.
    """
    # sents = DELIM_SENT.split(text)
    if "\r" in text:
        sents = text.split("\r\n")
    else:
        sents = text.split("\n")
    lens_char = np.array([len(sent) for sent in sents])
    return np.array(
        [
            len(lens_char),
            np.mean(lens_char),
            np.std(lens_char, ddof=1),
            np.min(lens_char),
            np.quantile(lens_char, 0.25),
            np.median(lens_char),
            np.quantile(lens_char, 0.75),
            np.max(lens_char),
        ]
    )


def count_conversations(text: str) -> float:
    # 会話文の割合
    text = re.sub(r"\s", " ", text)
    singles = re.findall(r"「.+?」", text)
    doubles = re.findall(r"『.+?』", text)
    lens_single = [len(single) for single in singles]
    lens_double = [len(double) for double in doubles]
    return np.divide(sum(lens_single) + sum(lens_double), len(text))


def count_charcat(text: str) -> np.ndarray:
    text = re.sub(r"\s", " ", text)
    c = Counter([ud.name(char).split()[0] for char in text])
    counts = np.array([c["HIRAGANA"], c["KATAKANA"], c["CJK"]])
    return np.divide(counts, len(text))


def measure_pos(text: str) -> np.ndarray:
    tokens = [
        (n.surface, n.feature.split(","))
        for n in NM.parse(text, as_nodes=True)
        if not n.is_eos()
    ]
    # print(tokens)

    # VERB RELATED MEASURES
    verbs = [token for token in tokens if token[1][0] == "動詞"]
    # TODO: 助動詞との連語も含める？
    # lens_verb = [len(verb) for verb in verbs]

    # CONTENT WORDS RATIO
    nouns = [token for token in tokens if token[1][0] == "名詞"]
    adjcs = [token for token in tokens if token[1][0] == "形容詞"]
    content_words = verbs + nouns + adjcs
    cwr_simple = np.divide(len(content_words), len(tokens))
    cwr_advance = np.divide(
        len(
            [
                token
                for token in content_words
                if (token[1][1] not in STOPPOS_JP) and (token[0] not in STOPWORDS_JP)
            ]
        ),
        len(tokens),
    )

    # NOTE: skip FUNCTION WORDS RATIO since it's equiv to 1 - CWR

    # Modifying words and verb ratio (MVR)
    advbs = [token for token in tokens if token[1][0] == "副詞"]
    padjs = [token for token in tokens if token[1][0] == "連体詞"]
    mvr = np.divide(len(adjcs + advbs + padjs), len(verbs))

    # NER
    ners = [token for token in tokens if token[1][1] == "固有名詞"]
    nerr = np.divide(len(ners), len(tokens))

    # TTR
    ttrs = calc_ttrs(tokens)

    return np.concatenate(
        (
            np.array(
                [
                    # np.mean(lens_verb),
                    # np.std(lens_verb),
                    # np.min(lens_verb),
                    # np.quantile(lens_verb, 0.25),
                    # np.median(lens_verb),
                    # np.quantile(lens_verb, 0.75),
                    # np.max(lens_verb),
                    cwr_simple,
                    cwr_advance,
                    mvr,
                    nerr,
                ]
            ),
            ttrs,
        )
    )


def measure_abst(text: str) -> np.ndarray:
    # AWD uses neologd
    tokens = [
        (n.surface, n.feature.split(","))
        for n in NMN.parse(text, as_nodes=True)
        if not n.is_eos()
    ]
    # print(tokens)
    scores = [
        float(AWD.get(token[0] if token[1][6] == "*" else token[1][6], 0))
        for token in tokens
    ]
    # print(scores)

    # top k=5 mean
    return np.array([np.mean(sorted(scores, reverse=True)[:5]), max(scores)])


def detect_bunmatsu(text: str) -> float:
    if "\r" in text:
        sents = text.split("\r\n")
    else:
        sents = text.split("\n")

    # 体言止め
    taigen = 0
    for sent in sents:
        tokens = [
            (n.surface, n.feature.split(","))
            for n in NM.parse(text, as_nodes=True)
            if not n.is_eos()
        ]
        taigen += 1 if tokens[-2][1] == "名詞" else 0
    ratio_taigen = np.divide(taigen, len(sents))

    # TODO: what else?

    return ratio_taigen


def calc_ttrs(tokens: List[Tuple[str, List[str]]]) -> np.ndarray:
    cnt = Counter([token[1][6] for token in tokens])
    Vn = len(cnt)
    logVn = np.log(Vn)
    N = np.sum(list(cnt.values()))
    logN = np.log(N)
    # TODO: implement frequency-wise TTR variants
    return np.array(
        [
            np.divide(Vn, N),  # original plain TTR: not robust to the length
            np.divide(Vn, np.sqrt(N)),  # Guiraud's R
            np.divide(logVn, logN),  # Herdan's C_H
            np.divide(logVn, np.log(logN)),  # Rubet's k
            np.divide((logN - logVn), (logN ** 2)),  # Maas's a^2
            np.divide((1 - (Vn ** 2)), ((Vn ** 2) * logN)),  # Tuldava's LN
            np.float_power(N, np.float_power(Vn, 0.172)),  # Brunet's W
            np.divide((logN ** 2), (logN - logVn)),  # Dugast's U
        ]
    )


def calc_potentialvocab(text: str) -> float:
    # 荒牧先生の潜在語彙量も
    raise NotImplementedError


def calc_jiwc(text: str) -> np.ndarray:
    tokens = [
        (n.surface, n.feature.split(","))
        for n in NM.parse(text, as_nodes=True)
        if not n.is_eos()
    ]
    jiwc_words = set(
        [token[0] if token[1][6] == "*" else token[1][6] for token in tokens]
    ) & set(DF_jiwc.index)
    jiwc_vals = DF_jiwc.loc[jiwc_words].sum()
    return np.divide(jiwc_vals, jiwc_vals.sum())
    # Sad Anx Anger Hate Trustful S Happy


def apply_all(text: str) -> Dict[str, Num]:
    try:
        all_res = np.concatenate(
            (
                measure_sents(text),
                [count_conversations(text)],
                count_charcat(text),
                measure_pos(text),
                measure_abst(text),
                [detect_bunmatsu(text)],
                calc_jiwc(text),
            )
        )
    except:
        print(text)
        raise
    headers = [
        "num_sent",
        "mean_sent_len",
        "std_sent_len",
        "min_sent_len",
        "q1_sent_len",
        "median_sent_len",
        "q3_sent_len",
        "max_sent_len",
        "num_conv",
        "pct_hiragana",
        "pct_katakana",
        "pct_kanji",
        "cwr_simple",
        "cwr_advance",
        "mvr",
        "pct_ner",
        "ttr_plain",
        "guiraud_r",
        "herdan_ch",
        "rubet_k",
        "maas_a2",
        "tuldava_ln",
        "brunet_w",
        "dugast_u",
        "top5_mean_abst",
        "max_abst",
        "ratio_taigendome",
        "jiwc_sadness",
        "jiwc_anxiety",
        "jiwc_anger",
        "jiwc_hatrid",
        "jiwc_trust",
        "jiwc_surprise",
        "jiwc_happiness",
    ]
    return dict(zip(headers, all_res))


def apply_file(fname, col):
    if fname.endswith(".csv"):
        df = pd.read_csv(fname)
    elif fname.endswith(".xls") or fname.endswith(".xlsx"):
        df = pd.read_excel(fname)
    else:
        raise ValueError("Unsupported input format: please use CSV or Excel data")

    assert col in df.columns, f"{col} is not found in the input data"

    pd.concat(
        [df, df.apply(lambda row: apply_all(row[col]), result_type="expand", axis=1)],
        axis=1,
    ).to_csv(f"{fname}.measured.csv", index=False)


if __name__ == "__main__":
    fire.Fire(apply_file)
