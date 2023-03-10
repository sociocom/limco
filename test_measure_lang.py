import pandas as pd
import pytest

import measure_lang as ml

TEXT = """ここは駅から程よい距離にある日本の住宅街である。
「びっくり！」
この街は市内でも最も少子高齢化が進んでいる街として有名。
その現状として小学校が合併され、二校になったり、
街の南の中学校のクラスは1クラスになりそうなところである。
しかも、街の北にある中学校でさえも2クラスになろうとしている。『結構小さい街なんだね。』
だが、いつもこの街にあるスーパーで毎年行われる納涼祭はとても盛り上がり、この街だけではおさまらず、他のところから来ている人も多数いる。
"""
DOC = ml.NLP(ml.normalise(TEXT))


def test_count_charcat():
    text = "あれとコレと竜巻．"
    assert ml.count_charcat(text) == {"hiragana": 4, "katakana": 2, "kanji": 2}


def test_count_conversations():
    assert ml.count_conversations(TEXT) == {"single": 1, "double": 1}


def test_describe_sentence_lengths():
    assert list(ml.describe_sentence_lengths(DOC).values()) == pytest.approx(
        [31.857143, 21.341888, 7.0, 18.5, 28.0, 42.0, 67.0]
    )


def test_calc_ttrs():
    # FIXME: 数式との一致を確認する
    assert ml.calc_ttrs(
        ["今日", "明日", "月曜日", "明るい", "明るい", "今日"]
    ).values() == pytest.approx(
        [
            0.6666666666666666,
            1.6329931618554523,
            0.7737056144690831,
            2.377055766815052,
            0.1262973012936895,
            -0.5232287123917944,
            9.72041321151472,
            7.917825557290553,
        ]
    )


def test_score_abstractness():
    awd = {
        "程よい": 5.0,
        "少子高齢化": 1.0,
        "スーパー": 2.0,
        "小さい": 3.2,
        "中学校": 2.0,
        "納涼祭": 1.0,
    }
    assert list(
        ml.score_abstractness(list(awd.keys()) + ["明日", "今日"], awd).values()
    ) == pytest.approx([2.64, 5.0])


def test_score_jiwc():
    df_jiwc = pd.DataFrame(
        data=[
            ["明日", 0.1, 0.0, 0.0, 0.2, 0.3, 0.0, 0.02],
            ["今日", 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01],
            ["感謝", 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.67],
        ],
        columns=["word", "Sad", "Anx", "Anger", "hate", "Trustful", "S", "Happy"],
    ).set_index("word")
    res = list(ml.score_jiwc(["明日", "感謝", "大きい"], df_jiwc).values())
    assert res == pytest.approx(
        [0.04784689, 0.0, 0.0, 0.09569377, 0.526316, 0.0, 0.3301435]
    )


def test_count_taigendome():
    assert ml.count_taigendome(DOC) == 1
