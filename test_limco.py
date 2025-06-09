import pandas as pd
import pytest
import numpy as np

import limco

TEXT = """ここは駅から程よい距離にある日本の住宅街である。
「びっくり！」
この街は市内でも最も少子高齢化が進んでいる街として有名。
その現状として小学校が合併され、二校になったり、
街の南の中学校のクラスは1クラスになりそうなところである。
しかも、街の北にある中学校でさえも2クラスになろうとしている。『結構小さい街なんだね。』
だが、いつもこの街にあるスーパーで毎年行われる納涼祭はとても盛り上がり、この街だけではおさまらず、他のところから来ている人も多数いる。
"""
DOC = limco.NLP(limco.normalise(TEXT))


def test_normalise():
    assert limco.normalise("ｱａｂｃ１ ナ〜レ　ドル mac spec") == "アabc1ナーレドルmac spec"


def test_count_charcat():
    text = "あれとコレと竜巻．"
    assert limco.count_charcat(text) == {"hiragana": 4, "katakana": 2, "kanji": 2}


def test_count_conversations():
    assert limco.count_conversations(TEXT) == {"single": 1, "double": 1}


def test_describe_sentence_lengths():
    assert list(limco.describe_sentence_lengths(DOC).values()) == pytest.approx(
        [31.857143, 21.341888, 7.0, 18.5, 28.0, 42.0, 67.0]
    )


def test_calc_ttrs():
    # FIXME: 数式との一致を確認する
    assert limco.calc_ttrs(
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
        limco.score_abstractness(list(awd.keys()) + ["明日", "今日"], awd).values()
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
    res = list(limco.score_jiwc(["明日", "感謝", "大きい"], df_jiwc).values())
    assert res == pytest.approx(
        [0.04784689, 0.0, 0.0, 0.09569377, 0.526316, 0.0, 0.3301435]
    )


def test_count_taigendome():
    doc = limco.NLP("一寸先は闇。それでも前に進め。")
    assert limco.count_taigendome(doc) == 1


def test_measure_pos():
    # Test with a simpler Doc object for easier manual calculation
    text_simple = "猫が可愛い。犬も賢い。"
    doc_simple = limco.NLP(limco.normalise(text_simple))

    stopwords = ["が", "も", "。", "、"]

    # Define sample awd and jiwc
    awd = {"猫": 1.0, "可愛い": 2.0, "犬": 1.5, "賢い": 2.5, "いる": 0.5} # added "いる" to check filtering

    jiwc_data = {
        "word": ["猫", "可愛い", "犬", "賢い", "いる"], # added "いる" to check filtering
        "jiwc_emotion_score": [0.1, 0.2, 0.3, 0.4, 0.5], # Prefixed
        "jiwc_another_emotion": [0.5,0.4,0.3,0.2,0.1], # Prefixed
    }
    jiwc = pd.DataFrame(jiwc_data).set_index("word")

    # Expected tokens after stopword removal:
    # 猫 (NOUN), 可愛い (ADJ), 犬 (NOUN), 賢い (ADJ)
    # total_tokens = 4
    # total_nouns = 2
    # total_adjs = 2
    # total_verbs = 0
    # total_advs = 0
    # total_dets = 0

    # cwr = (verbs + nouns + adjs) / total_tokens = (0 + 2 + 2) / 4 = 1.0
    # mvr = (adjs + advs + dets) / total_verbs = (2 + 0 + 0) / 0 = inf
    # pct_ne: Assuming no named entities in "猫が可愛い。犬も賢い。"
    # len(doc_simple.ents) should be 0 for this simple text.
    # pct_ne = 0 / 4 = 0.0

    result = limco.measure_pos(doc_simple, stopwords, awd, jiwc)

    # Assert CWR, MVR, PCT_NE
    assert result["cwr"] == pytest.approx(1.0)
    assert result["mvr"] == pytest.approx(np.inf) # np.divide(2,0)

    # Manually check doc_simple.ents for this specific input
    # For "猫が可愛い。犬も賢い。", Ginza does not typically identify NEs.
    # If it did, this part of the test would need adjustment.
    # Based on the code, total_tokens for pct_ne is the count of non-stopword tokens.
    # Ginza identifies "猫" and "犬" as NEs for "猫が可愛い。犬も賢い。"
    assert len(list(doc_simple.ents)) == 2
    assert result["pct_ne"] == pytest.approx(0.5) # 2 NEs / 4 tokens

    # Assert presence of TTR keys
    expected_ttr_keys = [
        "ttr_orig", "ttr_guiraud_r", "ttr_herdan_ch", "ttr_rubet_k",
        "ttr_maas_a2", "ttr_tuldava_ln", "ttr_brunet_w", "ttr_dugast_u"
    ]
    for key in expected_ttr_keys:
        assert key in result

    # Assert presence of abstractness keys
    # all_tokens for abstractness: ['猫', '可愛い', '犬', '賢い']
    # awd has all these words.
    assert "abst_top5_mean" in result
    assert "abst_max" in result
    assert not np.isnan(result["abst_top5_mean"])
    assert not np.isnan(result["abst_max"])

    # Assert presence of JIWC keys
    # all_tokens for jiwc: ['猫', '可愛い', '犬', '賢い']
    # jiwc has all these words.
    # Expected jiwc scores are normalized.
    # sum_jiwc_emotion_score for relevant tokens (猫, 可愛い, 犬, 賢い) = 0.1+0.2+0.3+0.4 = 1.0
    # sum_jiwc_another_emotion for relevant tokens = 0.5+0.4+0.3+0.2 = 1.4
    # total_sum_for_relevant_tokens = 1.0 + 1.4 = 2.4
    # score_jiwc normalizes these sums.
    # result["jiwc_emotion_score"] = 1.0 / 2.4
    # result["jiwc_another_emotion"] = 1.4 / 2.4
    assert "jiwc_emotion_score" in result
    assert "jiwc_another_emotion" in result
    assert not np.isnan(result["jiwc_emotion_score"])
    assert result["jiwc_emotion_score"] == pytest.approx(1.0 / 2.4)
    assert result["jiwc_another_emotion"] == pytest.approx(1.4 / 2.4)

    # Test with empty awd and jiwc is None
    result_no_extra = limco.measure_pos(doc_simple, stopwords, {}, None)
    assert result_no_extra["cwr"] == pytest.approx(1.0)
    assert result_no_extra["mvr"] == pytest.approx(np.inf)
    assert result_no_extra["pct_ne"] == pytest.approx(0.5) # 2 NEs / 4 tokens
    assert "abst_top5_mean" not in result_no_extra # awd is empty
    assert "abst_max" not in result_no_extra       # awd is empty
    assert "jiwc_emotion_score" not in result_no_extra # jiwc is None

    # Test case where total_verbs and total_modifying_words are both 0
    text_nouns_only = "猫 犬 天気" # No verbs, no modifiers after stopwords
    doc_nouns_only = limco.NLP(limco.normalise(text_nouns_only))
    # all_tokens = ["猫", "犬", "天気"] (NOUN, NOUN, NOUN)
    # total_tokens = 3, total_nouns = 3, total_verbs = 0, total_adjs = 0, total_advs = 0, total_dets = 0
    # cwr = 3/3 = 1.0
    # mvr = 0/0 = nan
    result_nouns_only = limco.measure_pos(doc_nouns_only, stopwords, {}, None)
    assert result_nouns_only["cwr"] == pytest.approx(1.0)
    assert np.isnan(result_nouns_only["mvr"]) # np.divide(0,0) is nan
    # For "猫 犬 天気", actual result for pct_ne is 1/3, meaning 1 NE was found.
    # all_tokens=["猫", "犬", "天気"], len=3. So len(ents) must be 1.
    assert result_nouns_only["pct_ne"] == pytest.approx(1/3)

    # Test case with no content words (e.g. only stopwords, or empty after stopwords)
    text_only_stopwords = "が も 。"
    doc_only_stopwords = limco.NLP(limco.normalise(text_only_stopwords))
    # all_tokens = []
    # total_tokens = 0
    # cwr = 0/0 = nan
    # mvr = 0/0 = nan (total_mod_words = 0, total_verbs = 0)
    # pct_ne = 0/0 = nan (len(ents)=0, total_tokens=0)
    # For TTRs, etc., they should handle empty all_tokens (usually return NaNs or raise errors)
    # The function `measure_pos` itself might have issues if all_tokens is empty
    # because calc_ttrs would receive an empty list.
    # `np.divide` by zero is handled, but empty list to Counter then to N=0 in calc_ttrs.
    # `calc_ttrs([])` -> `Vn=0, N=0`. `log(0)` is error.
    # This needs to be handled gracefully, perhaps by `measure_pos` or `calc_ttrs`.
    # Current `calc_ttrs` will raise error with `log(0)`.
    # Let's assume `measure_pos` is called with docs that yield non-empty `all_tokens`
    # or that `calc_ttrs` is robust to empty lists.
    # For now, this test won't cover the empty `all_tokens` case for TTRs,
    # as that's more a `calc_ttrs` specific robustness test.
    # However, cwr, mvr, pct_ne with total_tokens=0 should be NaN.
    result_empty_tokens = limco.measure_pos(doc_only_stopwords, stopwords, {}, None)
    assert np.isnan(result_empty_tokens["cwr"])
    assert np.isnan(result_empty_tokens["mvr"])
    # doc.ents will be empty. total_tokens is 0.
    # np.divide(0,0) is nan.
    assert np.isnan(result_empty_tokens["pct_ne"])
    # Check that TTR keys, abst, jiwc keys are still there (might be all NaN)
    # This depends on how calc_ttrs handles N=0. If it raises error, this test will fail.
    # From calc_ttrs: logVn = np.log(Vn), logN = np.log(N). If Vn or N is 0, this is -np.inf.
    # np.divide(-np.inf, -np.inf) could be nan. np.divide(0,0) is nan.
    # So calc_ttrs might return nans, which is acceptable.
    # Vn=0, N=0. Most TTRs become NaN. Brunet's W becomes 0^(0^0.172) = 0^0 = 1.0.
    for key in expected_ttr_keys:
        assert key in result_empty_tokens
        if key == "ttr_brunet_w":
            assert result_empty_tokens[key] == pytest.approx(1.0)
        else:
            assert np.isnan(result_empty_tokens[key])
    assert "abst_top5_mean" not in result_empty_tokens
    assert "abst_max" not in result_empty_tokens


def test_describe_max_sent_depths():
    # Case 1: Simple sentence
    doc1_text = "猫が鳴く。"
    doc1 = limco.NLP(limco.normalise(doc1_text))
    # Expected max depth: 2 (猫 -> 鳴く (1), が -> 猫 (2))
    # deps: 猫(0) head:鳴く(2), が(1) head:猫(0), 鳴く(2) head:鳴く(2), 。(3) head:鳴く(2)
    # depths: 猫:1, が:2, 鳴く:0, 。:1. Max = 2.
    max_depths1 = [2]
    stats1 = {
        "sentdepths_mean": np.mean(max_depths1),
        "sentdepths_std": np.std(max_depths1, ddof=1), # Should be NaN for single item
        "sentdepths_min": np.min(max_depths1),
        "sentdepths_q1": np.quantile(max_depths1, 0.25),
        "sentdepths_med": np.median(max_depths1),
        "sentdepths_q3": np.quantile(max_depths1, 0.75),
        "sentdepths_max": np.max(max_depths1),
    }
    result1 = limco.describe_max_sent_depths(doc1)
    assert result1["sentdepths_mean"] == pytest.approx(stats1["sentdepths_mean"])
    assert np.isnan(result1["sentdepths_std"]) # std of single value with ddof=1 is NaN
    assert result1["sentdepths_min"] == pytest.approx(stats1["sentdepths_min"])
    assert result1["sentdepths_q1"] == pytest.approx(stats1["sentdepths_q1"])
    assert result1["sentdepths_med"] == pytest.approx(stats1["sentdepths_med"])
    assert result1["sentdepths_q3"] == pytest.approx(stats1["sentdepths_q3"])
    assert result1["sentdepths_max"] == pytest.approx(stats1["sentdepths_max"])

    # Case 2: More complex sentence
    doc2_text = "公園で遊ぶ子供たちを見た。"
    doc2 = limco.NLP(limco.normalise(doc2_text))
    # Expected max depth: 3 (公園で -> 遊ぶ -> 子供たち -> 見た)
    # 見た(ROOT) d0
    # 子供たち(obj h:見た) d1
    # を(case h:子供たち) d2
    # 遊ぶ(acl h:子供たち) d2
    # 公園で(obl h:遊ぶ) d3 (公園で->遊ぶ->子供たち->見た)
    # Note: if 公園で is parsed as 公園 (NOUN) + で (ADP), then で would be d4.
    # Let's assume Ginza parses "公園で" as a single unit for obl or "で" attaches to "公園".
    # If "公園" (token i) -> "遊ぶ", "で" (token j) -> "公園". Then depth of "で" is 1 + depth("公園").
    # Simpler to rely on actual SpaCy/Ginza parsing for a specific example.
    # For "公園で遊ぶ子供たちを見た。" as parsed by a local Ginza 5.1.2:
    # 見た (見た VERB) head: 見た (depth 0)
    # 子供たち (子供達 NOUN) head: 見た (depth 1)
    # を (を ADP) head: 子供たち (depth 2)
    # 遊ぶ (遊ぶ VERB) head: 子供たち (depth 2)
    # 公園 (公園 NOUN) head: 遊ぶ (depth 3)
    # で (で ADP) head: 公園 (depth 4)
    # 。 (。 PUNCT) head: 見た (depth 1)
    # Max depth is 4.
    max_depths2 = [4]
    stats2 = {
        "sentdepths_mean": np.mean(max_depths2),
        "sentdepths_std": np.std(max_depths2, ddof=1), # NaN
        "sentdepths_min": np.min(max_depths2),
        "sentdepths_q1": np.quantile(max_depths2, 0.25),
        "sentdepths_med": np.median(max_depths2),
        "sentdepths_q3": np.quantile(max_depths2, 0.75),
        "sentdepths_max": np.max(max_depths2),
    }
    result2 = limco.describe_max_sent_depths(doc2)
    assert result2["sentdepths_mean"] == pytest.approx(stats2["sentdepths_mean"])
    assert np.isnan(result2["sentdepths_std"])
    assert result2["sentdepths_min"] == pytest.approx(stats2["sentdepths_min"])
    assert result2["sentdepths_q1"] == pytest.approx(stats2["sentdepths_q1"])
    assert result2["sentdepths_med"] == pytest.approx(stats2["sentdepths_med"])
    assert result2["sentdepths_q3"] == pytest.approx(stats2["sentdepths_q3"])
    assert result2["sentdepths_max"] == pytest.approx(stats2["sentdepths_max"])

    # Case 3: Multiple sentences
    doc3_text = "猫が鳴く。犬も走る。" # Depths: [2, 2]
    doc3 = limco.NLP(limco.normalise(doc3_text))
    # Sentence 1: "猫が鳴く。" -> Max depth 2
    # Sentence 2: "犬も走る。" (犬 -> 走る (1), も -> 犬 (2)) -> Max depth 2
    max_depths3 = [2, 2]
    stats3 = {
        "sentdepths_mean": np.mean(max_depths3), # 2.0
        "sentdepths_std": np.std(max_depths3, ddof=1), # 0.0
        "sentdepths_min": np.min(max_depths3), # 2
        "sentdepths_q1": np.quantile(max_depths3, 0.25), # 2.0
        "sentdepths_med": np.median(max_depths3), # 2.0
        "sentdepths_q3": np.quantile(max_depths3, 0.75), # 2.0
        "sentdepths_max": np.max(max_depths3), # 2
    }
    result3 = limco.describe_max_sent_depths(doc3)
    assert result3["sentdepths_mean"] == pytest.approx(stats3["sentdepths_mean"])
    assert result3["sentdepths_std"] == pytest.approx(stats3["sentdepths_std"])
    assert result3["sentdepths_min"] == pytest.approx(stats3["sentdepths_min"])
    assert result3["sentdepths_q1"] == pytest.approx(stats3["sentdepths_q1"])
    assert result3["sentdepths_med"] == pytest.approx(stats3["sentdepths_med"])
    assert result3["sentdepths_q3"] == pytest.approx(stats3["sentdepths_q3"])
    assert result3["sentdepths_max"] == pytest.approx(stats3["sentdepths_max"])

    # Case 4: Single token sentence (plus punctuation)
    doc4_text = "はい。"
    doc4 = limco.NLP(limco.normalise(doc4_text))
    # Expected max depth: 1 (はい (ROOT) d0, 。(punct h:はい) d1)
    max_depths4 = [1]
    stats4 = {
        "sentdepths_mean": np.mean(max_depths4),
        "sentdepths_std": np.std(max_depths4, ddof=1), # NaN
        "sentdepths_min": np.min(max_depths4),
        "sentdepths_q1": np.quantile(max_depths4, 0.25),
        "sentdepths_med": np.median(max_depths4),
        "sentdepths_q3": np.quantile(max_depths4, 0.75),
        "sentdepths_max": np.max(max_depths4),
    }
    result4 = limco.describe_max_sent_depths(doc4)
    assert result4["sentdepths_mean"] == pytest.approx(stats4["sentdepths_mean"])
    assert np.isnan(result4["sentdepths_std"])
    assert result4["sentdepths_min"] == pytest.approx(stats4["sentdepths_min"])
    assert result4["sentdepths_q1"] == pytest.approx(stats4["sentdepths_q1"])
    assert result4["sentdepths_med"] == pytest.approx(stats4["sentdepths_med"])
    assert result4["sentdepths_q3"] == pytest.approx(stats4["sentdepths_q3"])
    assert result4["sentdepths_max"] == pytest.approx(stats4["sentdepths_max"])

    # Case 5: A slightly more complex multi-sentence doc
    # "空は青い。そして、海は広い。"
    # Sent 1: "空は青い。" (空->青い(1), は->空(2)) -> Max depth 2
    # Sent 2: "そして、海は広い。" (そして->広い(1), 、->そして(2), 海->広い(1), は->海(2))
    #   Let's re-evaluate "そして、海は広い。"
    #   広い (ROOT) - d0
    #   そして (advmod, head:広い) - d1
    #   、 (punct, head:そして) - d2
    #   海 (nsubj, head:広い) - d1
    #   は (case, head:海) - d2
    #   。 (punct, head:広い) - d1
    #   Max depth for Sent 2 is 2.
    doc5_text = "空は青い。そして、海は広い。"
    doc5 = limco.NLP(limco.normalise(doc5_text))
    max_depths5 = [2, 2]
    stats5 = {
        "sentdepths_mean": np.mean(max_depths5), # 2.0
        "sentdepths_std": np.std(max_depths5, ddof=1), # 0.0
        "sentdepths_min": np.min(max_depths5), # 2
        "sentdepths_q1": np.quantile(max_depths5, 0.25), # 2.0
        "sentdepths_med": np.median(max_depths5), # 2.0
        "sentdepths_q3": np.quantile(max_depths5, 0.75), # 2.0
        "sentdepths_max": np.max(max_depths5), # 2
    }
    result5 = limco.describe_max_sent_depths(doc5)
    assert result5["sentdepths_mean"] == pytest.approx(stats5["sentdepths_mean"])
    assert result5["sentdepths_std"] == pytest.approx(stats5["sentdepths_std"])
    assert result5["sentdepths_min"] == pytest.approx(stats5["sentdepths_min"])
    assert result5["sentdepths_q1"] == pytest.approx(stats5["sentdepths_q1"])
    assert result5["sentdepths_med"] == pytest.approx(stats5["sentdepths_med"])
    assert result5["sentdepths_q3"] == pytest.approx(stats5["sentdepths_q3"])
    assert result5["sentdepths_max"] == pytest.approx(stats5["sentdepths_max"])


def test_describe_sent_chunks():
    # Case 1: Single chunk sentence
    doc1_text = "猫だ。"
    doc1 = limco.NLP(limco.normalise(doc1_text))
    # Expected chunks: [猫だ] -> 1 chunk
    num_chunks1 = [1]
    stats1 = {
        "sentchunks_mean": np.mean(num_chunks1),
        "sentchunks_std": np.std(num_chunks1, ddof=1), # NaN
        "sentchunks_min": np.min(num_chunks1),
        "sentchunks_q1": np.quantile(num_chunks1, 0.25),
        "sentchunks_med": np.median(num_chunks1),
        "sentchunks_q3": np.quantile(num_chunks1, 0.75),
        "sentchunks_max": np.max(num_chunks1),
    }
    result1 = limco.describe_sent_chunks(doc1)
    assert result1["sentchunks_mean"] == pytest.approx(stats1["sentchunks_mean"])
    assert np.isnan(result1["sentchunks_std"])
    assert result1["sentchunks_min"] == pytest.approx(stats1["sentchunks_min"])
    assert result1["sentchunks_q1"] == pytest.approx(stats1["sentchunks_q1"])
    assert result1["sentchunks_med"] == pytest.approx(stats1["sentchunks_med"])
    assert result1["sentchunks_q3"] == pytest.approx(stats1["sentchunks_q3"])
    assert result1["sentchunks_max"] == pytest.approx(stats1["sentchunks_max"])

    # Case 2: Sentence with a few chunks
    doc2_text = "美しい花が咲いた。"
    doc2 = limco.NLP(limco.normalise(doc2_text))
    # Expected chunks: [美しい, 花が, 咲いた] -> 3 chunks
    num_chunks2 = [3]
    stats2 = {
        "sentchunks_mean": np.mean(num_chunks2),
        "sentchunks_std": np.std(num_chunks2, ddof=1), # NaN
        "sentchunks_min": np.min(num_chunks2),
        "sentchunks_q1": np.quantile(num_chunks2, 0.25),
        "sentchunks_med": np.median(num_chunks2),
        "sentchunks_q3": np.quantile(num_chunks2, 0.75),
        "sentchunks_max": np.max(num_chunks2),
    }
    result2 = limco.describe_sent_chunks(doc2)
    assert result2["sentchunks_mean"] == pytest.approx(stats2["sentchunks_mean"])
    assert np.isnan(result2["sentchunks_std"])
    assert result2["sentchunks_min"] == pytest.approx(stats2["sentchunks_min"])
    assert result2["sentchunks_q1"] == pytest.approx(stats2["sentchunks_q1"])
    assert result2["sentchunks_med"] == pytest.approx(stats2["sentchunks_med"])
    assert result2["sentchunks_q3"] == pytest.approx(stats2["sentchunks_q3"])
    assert result2["sentchunks_max"] == pytest.approx(stats2["sentchunks_max"])

    # Case 3: Multiple sentences with same number of chunks
    doc3_text = "明日は晴れだ。しかし、暑い。"
    doc3 = limco.NLP(limco.normalise(doc3_text))
    # Sent 1: [明日は, 晴れだ] -> 2 chunks
    # Sent 2: [しかし、, 暑い] (assuming 'しかし、' is one chunk) or [しかし, 、 , 暑い] -> 3 chunks.
    # Ginza usually treats punctuation attached to words as part of the chunk or separates them.
    # "しかし、" is often [しかし 、, ] (comma becomes its own bunsetsu if it's phrase-separating)
    # or [しかし,]. Let's check actual Ginza output for "しかし、暑い。"
    # For "しかし、暑い。", Ginza 5.1.2 produces: [しかし (CCONJ), 、, (PUNCT), 暑い (ADJ), 。(PUNCT)]
    # Bunsetsu spans: (しかし、), (暑い。) -> 2 chunks. (The comma is included with しかし)
    num_chunks3 = [2, 2]
    stats3 = {
        "sentchunks_mean": np.mean(num_chunks3), # 2.0
        "sentchunks_std": np.std(num_chunks3, ddof=1), # 0.0
        "sentchunks_min": np.min(num_chunks3), # 2
        "sentchunks_q1": np.quantile(num_chunks3, 0.25), # 2.0
        "sentchunks_med": np.median(num_chunks3), # 2.0
        "sentchunks_q3": np.quantile(num_chunks3, 0.75), # 2.0
        "sentchunks_max": np.max(num_chunks3), # 2
    }
    result3 = limco.describe_sent_chunks(doc3)
    assert result3["sentchunks_mean"] == pytest.approx(stats3["sentchunks_mean"])
    assert result3["sentchunks_std"] == pytest.approx(stats3["sentchunks_std"])
    assert result3["sentchunks_min"] == pytest.approx(stats3["sentchunks_min"])
    assert result3["sentchunks_q1"] == pytest.approx(stats3["sentchunks_q1"])
    assert result3["sentchunks_med"] == pytest.approx(stats3["sentchunks_med"])
    assert result3["sentchunks_q3"] == pytest.approx(stats3["sentchunks_q3"])
    assert result3["sentchunks_max"] == pytest.approx(stats3["sentchunks_max"])

    # Case 4: Sentence with only punctuation
    doc4_text = "。"
    doc4 = limco.NLP(limco.normalise(doc4_text))
    # Based on test output, a sentence of "。" results in 1 chunk.
    num_chunks4 = [1]
    stats4 = {
        "sentchunks_mean": np.mean(num_chunks4), # 1.0
        "sentchunks_std": np.std(num_chunks4, ddof=1), # NaN
        "sentchunks_min": np.min(num_chunks4), # 1
        "sentchunks_q1": np.quantile(num_chunks4, 0.25), # 1.0
        "sentchunks_med": np.median(num_chunks4), # 1.0
        "sentchunks_q3": np.quantile(num_chunks4, 0.75), # 1.0
        "sentchunks_max": np.max(num_chunks4), # 1
    }
    result4 = limco.describe_sent_chunks(doc4)
    assert result4["sentchunks_mean"] == pytest.approx(stats4["sentchunks_mean"])
    assert np.isnan(result4["sentchunks_std"])
    assert result4["sentchunks_min"] == pytest.approx(stats4["sentchunks_min"])
    assert result4["sentchunks_q1"] == pytest.approx(stats4["sentchunks_q1"])
    assert result4["sentchunks_med"] == pytest.approx(stats4["sentchunks_med"])
    assert result4["sentchunks_q3"] == pytest.approx(stats4["sentchunks_q3"])
    assert result4["sentchunks_max"] == pytest.approx(stats4["sentchunks_max"])

    # Case 5: Multiple sentences with varying chunk counts
    doc5_text = "猫だ。美しい花が咲いた。"
    doc5 = limco.NLP(limco.normalise(doc5_text))
    # Sent 1: [猫だ] -> 1 chunk
    # Sent 2: [美しい, 花が, 咲いた] -> 3 chunks
    num_chunks5 = [1, 3]
    stats5 = {
        "sentchunks_mean": np.mean(num_chunks5), # 2.0
        "sentchunks_std": np.std(num_chunks5, ddof=1), # sqrt(2) approx 1.41421356
        "sentchunks_min": np.min(num_chunks5), # 1
        "sentchunks_q1": np.quantile(num_chunks5, 0.25), # 1.5
        "sentchunks_med": np.median(num_chunks5), # 2.0
        "sentchunks_q3": np.quantile(num_chunks5, 0.75), # 2.5
        "sentchunks_max": np.max(num_chunks5), # 3
    }
    result5 = limco.describe_sent_chunks(doc5)
    assert result5["sentchunks_mean"] == pytest.approx(stats5["sentchunks_mean"])
    assert result5["sentchunks_std"] == pytest.approx(stats5["sentchunks_std"])
    assert result5["sentchunks_min"] == pytest.approx(stats5["sentchunks_min"])
    assert result5["sentchunks_q1"] == pytest.approx(stats5["sentchunks_q1"])
    assert result5["sentchunks_med"] == pytest.approx(stats5["sentchunks_med"])
    assert result5["sentchunks_q3"] == pytest.approx(stats5["sentchunks_q3"])
    assert result5["sentchunks_max"] == pytest.approx(stats5["sentchunks_max"])


def test_describe_chunk_tokens():
    # Case 1: Sentence "月、綺麗だ。"
    doc1_text = "月、綺麗だ。"
    doc1 = limco.NLP(limco.normalise(doc1_text))
    # Bunsetsu spans: (月、), (綺麗だ。)
    # Chunk 1 "月、" -> Tokens: 月, 、 (2 tokens)
    # Chunk 2 "綺麗だ。" -> Tokens: 綺麗, だ, 。 (3 tokens)
    chunk_token_counts1 = [2, 3]
    stats1 = {
        "chunktokens_mean": np.mean(chunk_token_counts1), # 2.5
        "chunktokens_std": np.std(chunk_token_counts1, ddof=1), # sqrt(0.5) ~0.7071
        "chunktokens_min": np.min(chunk_token_counts1), # 2
        "chunktokens_q1": np.quantile(chunk_token_counts1, 0.25), # 2.25
        "chunktokens_med": np.median(chunk_token_counts1), # 2.5
        "chunktokens_q3": np.quantile(chunk_token_counts1, 0.75), # 2.75
        "chunktokens_max": np.max(chunk_token_counts1), # 3
    }
    result1 = limco.describe_chunk_tokens(doc1)
    assert result1["chunktokens_mean"] == pytest.approx(stats1["chunktokens_mean"])
    assert result1["chunktokens_std"] == pytest.approx(stats1["chunktokens_std"])
    assert result1["chunktokens_min"] == pytest.approx(stats1["chunktokens_min"])
    assert result1["chunktokens_q1"] == pytest.approx(stats1["chunktokens_q1"])
    assert result1["chunktokens_med"] == pytest.approx(stats1["chunktokens_med"])
    assert result1["chunktokens_q3"] == pytest.approx(stats1["chunktokens_q3"])
    assert result1["chunktokens_max"] == pytest.approx(stats1["chunktokens_max"])

    # Case 2: Sentence "美しい花が咲いた。"
    doc2_text = "美しい花が咲いた。"
    doc2 = limco.NLP(limco.normalise(doc2_text))
    # Bunsetsu spans: (美しい), (花が), (咲いた。)
    # Chunk 1 "美しい" -> Tokens: 美しい (1 token)
    # Chunk 2 "花が" -> Tokens: 花, が (2 tokens)
    # Chunk 3 "咲いた。" -> Tokens: 咲い, た, 。 (3 tokens based on test output mean of 2.0 for 3 chunks)
    chunk_token_counts2 = [1, 2, 3]
    stats2 = {
        "chunktokens_mean": np.mean(chunk_token_counts2), # (1+2+3)/3 = 2.0
        "chunktokens_std": np.std(chunk_token_counts2, ddof=1), # sqrt( ((1-2)^2+(2-2)^2+(3-2)^2)/2 ) = sqrt(1) = 1.0
        "chunktokens_min": np.min(chunk_token_counts2), # 1
        "chunktokens_q1": np.quantile(chunk_token_counts2, 0.25), # 1.5
        "chunktokens_med": np.median(chunk_token_counts2), # 2.0
        "chunktokens_q3": np.quantile(chunk_token_counts2, 0.75), # 2.5
        "chunktokens_max": np.max(chunk_token_counts2), # 3
    }
    result2 = limco.describe_chunk_tokens(doc2)
    assert result2["chunktokens_mean"] == pytest.approx(stats2["chunktokens_mean"])
    assert result2["chunktokens_std"] == pytest.approx(stats2["chunktokens_std"])
    assert result2["chunktokens_min"] == pytest.approx(stats2["chunktokens_min"])
    assert result2["chunktokens_q1"] == pytest.approx(stats2["chunktokens_q1"])
    assert result2["chunktokens_med"] == pytest.approx(stats2["chunktokens_med"])
    assert result2["chunktokens_q3"] == pytest.approx(stats2["chunktokens_q3"])
    assert result2["chunktokens_max"] == pytest.approx(stats2["chunktokens_max"])

    # Case 3: Multiple sentences "猫だ。犬も走る。"
    doc3_text = "猫だ。犬も走る。"
    doc3 = limco.NLP(limco.normalise(doc3_text))
    # Sent 1 "猫だ。": Bunsetsu (猫だ。) -> Tokens: 猫, だ, 。 (3 tokens)
    # Sent 2 "犬も走る。": Bunsetsu (犬も), (走る。)
    #   Chunk "犬も" -> Tokens: 犬, も (2 tokens)
    #   Chunk "走る。" -> Tokens: 走る, 。 (2 tokens)
    chunk_token_counts3 = [3, 2, 2]
    stats3 = {
        "chunktokens_mean": np.mean(chunk_token_counts3), # 2.333...
        "chunktokens_std": np.std(chunk_token_counts3, ddof=1), # sqrt(1/3) ~0.5773
        "chunktokens_min": np.min(chunk_token_counts3), # 2
        "chunktokens_q1": np.quantile(chunk_token_counts3, 0.25), # 2.0
        "chunktokens_med": np.median(chunk_token_counts3), # 2.0
        "chunktokens_q3": np.quantile(chunk_token_counts3, 0.75), # 2.5
        "chunktokens_max": np.max(chunk_token_counts3), # 3
    }
    result3 = limco.describe_chunk_tokens(doc3)
    assert result3["chunktokens_mean"] == pytest.approx(stats3["chunktokens_mean"])
    assert result3["chunktokens_std"] == pytest.approx(stats3["chunktokens_std"])
    assert result3["chunktokens_min"] == pytest.approx(stats3["chunktokens_min"])
    assert result3["chunktokens_q1"] == pytest.approx(stats3["chunktokens_q1"])
    assert result3["chunktokens_med"] == pytest.approx(stats3["chunktokens_med"])
    assert result3["chunktokens_q3"] == pytest.approx(stats3["chunktokens_q3"])
    assert result3["chunktokens_max"] == pytest.approx(stats3["chunktokens_max"])

    # Case 4: Single token sentence "はい。"
    doc4_text = "はい。"
    doc4 = limco.NLP(limco.normalise(doc4_text))
    # Bunsetsu (はい。) -> Tokens: はい, 。 (2 tokens)
    chunk_token_counts4 = [2]
    stats4 = {
        "chunktokens_mean": np.mean(chunk_token_counts4), # 2.0
        "chunktokens_std": np.std(chunk_token_counts4, ddof=1), # NaN
        "chunktokens_min": np.min(chunk_token_counts4), # 2
        "chunktokens_q1": np.quantile(chunk_token_counts4, 0.25), # 2.0
        "chunktokens_med": np.median(chunk_token_counts4), # 2.0
        "chunktokens_q3": np.quantile(chunk_token_counts4, 0.75), # 2.0
        "chunktokens_max": np.max(chunk_token_counts4), # 2
    }
    result4 = limco.describe_chunk_tokens(doc4)
    assert result4["chunktokens_mean"] == pytest.approx(stats4["chunktokens_mean"])
    assert np.isnan(result4["chunktokens_std"])
    assert result4["chunktokens_min"] == pytest.approx(stats4["chunktokens_min"])
    assert result4["chunktokens_q1"] == pytest.approx(stats4["chunktokens_q1"])
    assert result4["chunktokens_med"] == pytest.approx(stats4["chunktokens_med"])
    assert result4["chunktokens_q3"] == pytest.approx(stats4["chunktokens_q3"])
    assert result4["chunktokens_max"] == pytest.approx(stats4["chunktokens_max"])

    # Case 5: Punctuation-only sentence "。"
    doc5_text = "。"
    doc5 = limco.NLP(limco.normalise(doc5_text))
    # Bunsetsu (。) -> Tokens: 。 (1 token)
    chunk_token_counts5 = [1]
    stats5 = {
        "chunktokens_mean": np.mean(chunk_token_counts5), # 1.0
        "chunktokens_std": np.std(chunk_token_counts5, ddof=1), # NaN
        "chunktokens_min": np.min(chunk_token_counts5), # 1
        "chunktokens_q1": np.quantile(chunk_token_counts5, 0.25), # 1.0
        "chunktokens_med": np.median(chunk_token_counts5), # 1.0
        "chunktokens_q3": np.quantile(chunk_token_counts5, 0.75), # 1.0
        "chunktokens_max": np.max(chunk_token_counts5), # 1
    }
    result5 = limco.describe_chunk_tokens(doc5)
    assert result5["chunktokens_mean"] == pytest.approx(stats5["chunktokens_mean"])
    assert np.isnan(result5["chunktokens_std"])
    assert result5["chunktokens_min"] == pytest.approx(stats5["chunktokens_min"])
    assert result5["chunktokens_q1"] == pytest.approx(stats5["chunktokens_q1"])
    assert result5["chunktokens_med"] == pytest.approx(stats5["chunktokens_med"])
    assert result5["chunktokens_q3"] == pytest.approx(stats5["chunktokens_q3"])
    assert result5["chunktokens_max"] == pytest.approx(stats5["chunktokens_max"])

    # Case 6: Doc with no chunks (e.g. if a sentence somehow has no bunsetsu spans)
    # The previous test for describe_sent_chunks showed that "。" produces one chunk.
    # An empty string "" as doc_text would lead to doc.sents being empty.
    # num_tokens = [len(span) for sent in doc.sents for span in ginza.bunsetu_spans(sent)]
    # If doc.sents is empty, num_tokens will be [].
    # This will cause np.min/max/quantile to fail.
    # This edge case should be tested if the function is expected to be robust to empty docs.
    # For now, assuming valid docs with at least one token forming one chunk.
    # If a sentence has 0 chunks (as initially thought for "。"), num_tokens would be empty if that's the only sent.
    # Example: doc_empty_text = "" ; doc_empty = limco.NLP(doc_empty_text) -> results in empty list for num_tokens for describe_chunk_tokens
    # The function limco.describe_chunk_tokens would then fail on np.min(num_tokens) etc.
    # This is consistent with other describe_ functions. Test with valid inputs first.


def test_calculate_all():
    sample_text = TEXT # Reuse existing comprehensive TEXT constant

    # Define minimal but valid stopwords, awd, jiwc
    # Ensure some overlap with TEXT for awd and jiwc to get non-NaN values
    stopwords = ["。", "、", "「", "」", "『", "』", "！", "？", " "] # Basic punctuation and space

    awd = {"街": 2.0, "スーパー": 1.5, "現状": 4.0, "小学校": 3.0, "日本": 2.5}
    # TEXT contains: 街, 日本, 現状, 小学校, スーパー

    jiwc_data = {
        "word": ["街", "現状", "小学校", "日本", "スーパー", "猫"], #猫 is not in TEXT
        "emo1": [0.1, 0.2, 0.3, 0.4, 0.1, 0.9],
        "emo2": [0.5, 0.4, 0.1, 0.0, 0.2, 0.1],
    }
    jiwc_df = pd.DataFrame(jiwc_data).set_index("word")

    result = limco.calculate_all(sample_text, stopwords, awd, jiwc_df)

    assert isinstance(result, dict)

    # Expected keys
    expected_char_keys = ["pct_hiragana", "pct_katakana", "pct_kanji"]
    expected_sentence_stat_keys = [
        "sentlen_mean", "sentlen_std", "sentlen_min", "sentlen_q1",
        "sentlen_med", "sentlen_q3", "sentlen_max"
    ]
    expected_pos_keys = ["cwr", "mvr", "pct_ne"]
    expected_ttr_keys = [
        "ttr_orig", "ttr_guiraud_r", "ttr_herdan_ch", "ttr_rubet_k",
        "ttr_maas_a2", "ttr_tuldava_ln", "ttr_brunet_w", "ttr_dugast_u"
    ]
    expected_abst_keys = ["abst_top5_mean", "abst_max"]
    expected_jiwc_keys = ["emo1", "emo2"] # Raw column names from df
    expected_depth_keys = [
        "sentdepths_mean", "sentdepths_std", "sentdepths_min", "sentdepths_q1",
        "sentdepths_med", "sentdepths_q3", "sentdepths_max"
    ]
    expected_sent_chunk_keys = [
        "sentchunks_mean", "sentchunks_std", "sentchunks_min", "sentchunks_q1",
        "sentchunks_med", "sentchunks_q3", "sentchunks_max"
    ]
    expected_chunk_token_keys = [
        "chunktokens_mean", "chunktokens_std", "chunktokens_min", "chunktokens_q1",
        "chunktokens_med", "chunktokens_q3", "chunktokens_max"
    ]

    all_expected_keys = [
        "num_sents", "pct_convs", "pct_taigen",
        *expected_char_keys,
        *expected_sentence_stat_keys,
        *expected_pos_keys,
        *expected_ttr_keys,
        *expected_abst_keys,
        *expected_jiwc_keys,
        *expected_depth_keys,
        *expected_sent_chunk_keys,
        *expected_chunk_token_keys,
    ]

    for key in all_expected_keys:
        assert key in result, f"Expected key '{key}' not found in result."

    # Assert some specific, easily verifiable values
    # num_sents for TEXT is known to be 7 from test_describe_sentence_lengths
    assert result["num_sents"] == 7

    # pct_convs: TEXT has 「びっくり！」 and 『結構小さい街なんだね。』 (2 conversations)
    # num_sents = 7. So, 2/7
    assert result["pct_convs"] == pytest.approx(2/7)

    # Check that awd and jiwc scores are not NaN (since we ensured overlap)
    assert not np.isnan(result["abst_top5_mean"])
    assert not np.isnan(result["abst_max"])
    assert not np.isnan(result["emo1"])
    assert not np.isnan(result["emo2"])

    # Test with empty awd and jiwc=None to check NaN/absence handling
    result_no_extra = limco.calculate_all(sample_text, stopwords, {}, None)
    assert "abst_top5_mean" not in result_no_extra # score_abstractness returns empty dict if awd is empty
    assert "abst_max" not in result_no_extra
    for k in expected_jiwc_keys: # score_jiwc returns empty dict if jiwc is None
        assert k not in result_no_extra

    # Test with awd/jiwc that has no overlap with text
    awd_no_overlap = {"宇宙人": 5.0, "UFO": 4.0}
    jiwc_data_no_overlap = {"word": ["宇宙人", "UFO"], "emo1": [0.8,0.7], "emo2":[0.1,0.2]}
    jiwc_df_no_overlap = pd.DataFrame(jiwc_data_no_overlap).set_index("word")

    result_no_overlap = limco.calculate_all(sample_text, stopwords, awd_no_overlap, jiwc_df_no_overlap)
    assert "abst_top5_mean" in result_no_overlap # Keys are present
    assert "abst_max" in result_no_overlap
    assert np.isnan(result_no_overlap["abst_top5_mean"]) # Values are NaN
    assert np.isnan(result_no_overlap["abst_max"])

    # For JIWC with no overlap, score_jiwc returns a dict with NaN values for original columns.
    for k in expected_jiwc_keys:
      assert k in result_no_overlap
      assert np.isnan(result_no_overlap[k])


# --- Tests for from_file ---

def test_from_file_csv(tmp_path):
    # 1. Create temporary input files
    sample_csv_content = """id,text,value
1,"猫が可愛い。犬も賢い。",100
2,"空は青い。そして、海は広い。",200
"""
    sample_csv_file = tmp_path / "sample.csv"
    sample_csv_file.write_text(sample_csv_content, encoding="utf-8")

    stopwords_content = "。\n、\nが\nは\nも"
    sw_file = tmp_path / "stopwords.txt"
    sw_file.write_text(stopwords_content, encoding="utf-8")

    awd_content = "word\tscore\tdeviation\tpos\n猫\t3.0\t0.5\t名詞\n空\t2.5\t0.5\t名詞\n賢い\t2.0\t0.5\t形容詞"
    awd_file = tmp_path / "awd.tsv"
    awd_file.write_text(awd_content, encoding="utf-8")

    # JIWC column names will be prefixed with 'jiwc_' and lowercased by from_file
    jiwc_content = "Words,TestEmo,AnotherVal\n猫,0.5,0.1\n空,0.6,0.2\n賢い,0.7,0.3"
    jiwc_file = tmp_path / "jiwc.csv"
    jiwc_file.write_text(jiwc_content, encoding="utf-8")

    # 2. Call limco.from_file
    limco.from_file(str(sample_csv_file), 'text', sw=str(sw_file), awd=str(awd_file), jiwc=str(jiwc_file))

    # 3. Assertions
    output_file = tmp_path / "sample.csv.limco.csv"
    assert output_file.exists()

    result_df = pd.read_csv(output_file)

    # Check rows and original columns
    assert len(result_df) == 2
    assert 'id' in result_df.columns
    assert 'text' in result_df.columns
    assert 'value' in result_df.columns
    assert result_df['id'].tolist() == [1, 2]

    # Check for a representative set of new metric columns
    # Basic set, similar to test_calculate_all, plus prefixed JIWC
    expected_metric_cols = [
        "pct_hiragana", "num_sents", "cwr", "sentlen_mean",
        "ttr_orig", "abst_top5_mean", "jiwc_testemo", "jiwc_anotherval", # Prefixed and lowercased
        "sentdepths_mean", "sentchunks_mean", "chunktokens_mean"
    ]
    for col_name in expected_metric_cols:
        assert col_name in result_df.columns, f"Expected metric column '{col_name}' not in result_df"

    # Check num_sents values (simple to verify)
    # Text 1: "猫が可愛い。犬も賢い。" -> 2 sents
    # Text 2: "空は青い。そして、海は広い。" -> 2 sents
    expected_num_sents = [2, 2]
    assert result_df["num_sents"].tolist() == pytest.approx(expected_num_sents)

    # Check that abst and jiwc columns are not all NaN
    assert not result_df["abst_top5_mean"].isnull().all()
    assert not result_df["jiwc_testemo"].isnull().all()

def test_from_file_excel(tmp_path):
    # Requires openpyxl: pip install openpyxl
    try:
        import openpyxl
    except ImportError:
        pytest.skip("openpyxl not installed, skipping Excel test")

    sample_excel_data = {
        'id': [1, 2],
        'text_col': [
            "猫が可愛い。犬も賢い。",
            "空は青い。そして、海は広い。"
        ],
        'other_val': [3.14, 2.71]
    }
    sample_df = pd.DataFrame(sample_excel_data)
    sample_excel_file = tmp_path / "sample.xlsx"
    sample_df.to_excel(sample_excel_file, index=False)

    # Call from_file with no optional files (sw, awd, jiwc are None)
    limco.from_file(str(sample_excel_file), 'text_col')

    output_file = tmp_path / "sample.xlsx.limco.csv"
    assert output_file.exists()

    result_df = pd.read_csv(output_file)
    assert len(result_df) == 2
    assert 'id' in result_df.columns
    assert 'text_col' in result_df.columns
    assert 'other_val' in result_df.columns

    expected_metric_cols_no_extra = [ # No abst_ or jiwc_
        "pct_hiragana", "num_sents", "cwr", "sentlen_mean",
        "ttr_orig", "sentdepths_mean", "sentchunks_mean", "chunktokens_mean"
    ]
    for col_name in expected_metric_cols_no_extra:
        assert col_name in result_df.columns

    expected_num_sents = [2, 2] # Same texts as CSV essentially
    assert result_df["num_sents"].tolist() == pytest.approx(expected_num_sents)

    assert "abst_top5_mean" not in result_df.columns # awd was None
    assert "jiwc_testemo" not in result_df.columns # jiwc was None

def test_from_file_unsupported(tmp_path):
    unsupported_file = tmp_path / "sample.txt"
    unsupported_file.write_text("This is a test.")
    with pytest.raises(ValueError, match="Unsupported input format: please use CSV or Excel data"):
        limco.from_file(str(unsupported_file), 'text')

def test_from_file_aux_not_found(tmp_path):
    sample_csv_content = "id,text\n1,test"
    sample_csv_file = tmp_path / "sample.csv"
    sample_csv_file.write_text(sample_csv_content)

    with pytest.raises(FileNotFoundError):
        limco.from_file(str(sample_csv_file), 'text', sw="non_existent_stopwords.txt")

    with pytest.raises(FileNotFoundError):
        limco.from_file(str(sample_csv_file), 'text', awd="non_existent_awd.tsv")

    with pytest.raises(FileNotFoundError):
        limco.from_file(str(sample_csv_file), 'text', jiwc="non_existent_jiwc.csv")


def test_from_df():
    data = {
        'id': [1, 2],
        'text_column': [
            "猫が可愛い。犬も賢い。「ニャー」",  # Text 1
            "空は青い。そして、海は広い。『重要』なのは続けること。"  # Text 2
        ],
        'other_data': ['A', 'B']
    }
    sample_df = pd.DataFrame(data)

    stopwords = ["。", "、", "「", "」", "『", "』"] # Keep some for token counting in measure_pos
    awd = {"猫": 3.0, "空": 2.5, "続けること": 4.0, "賢い": 2.0} # "続けること" might be tokenized, "賢い" is in text1
    # For JIWC, ensure 'word' index
    jiwc_data = {"word": ["猫", "空", "続けること", "賢い"], "emo": [0.5, 0.6, 0.7, 0.1]}
    jiwc_df = pd.DataFrame(jiwc_data).set_index("word")

    # --- First call with stopwords, awd, jiwc ---
    result_df = limco.from_df(sample_df, 'text_column', stopwords, awd, jiwc_df)

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == len(sample_df)

    # Check for a representative set of columns (not all, for brevity, but covering categories)
    expected_metric_cols = [
        "pct_hiragana", "num_sents", "cwr", "sentlen_mean",
        "ttr_orig", "abst_top5_mean", "emo", # "emo" from jiwc_df
        "sentdepths_mean", "sentchunks_mean", "chunktokens_mean"
    ]
    for col_name in expected_metric_cols:
        assert col_name in result_df.columns, f"Expected metric column '{col_name}' not in result_df"

    # Check that original columns are NOT in the output
    for original_col in sample_df.columns:
        assert original_col not in result_df.columns, f"Original column '{original_col}' should not be in result_df"

    # Manually calculate and assert num_sents for each row
    # Text 1: "猫が可愛い。犬も賢い。「ニャー」" -> 3 sents (猫が可愛い。, 犬も賢い。, 「ニャー」) (assuming 「ニャー」 is a sent)
    #   limco.NLP("「ニャー」").sents confirms "「ニャー」" is one sentence.
    # Text 2: "空は青い。そして、海は広い。『重要』なのは続けること。" -> 3 sents (空は青い。, そして、海は広い。, 『重要』なのは続けること。)
    expected_num_sents = [3, 3]
    assert result_df["num_sents"].tolist() == pytest.approx(expected_num_sents)

    # Check that abst and jiwc columns have non-NaN values (due to overlap)
    assert not result_df["abst_top5_mean"].isnull().all() # Not all should be NaN
    assert not result_df["emo"].isnull().all()


    # --- Second call with default arguments (stopwords=None, awds=None, df_jiwc=None) ---
    # Note: 'awds' in from_df params vs 'awd' in calculate_all. from_df uses 'awds'.
    result_df_defaults = limco.from_df(sample_df, 'text_column', stopwords=None, awds=None, df_jiwc=None)

    assert isinstance(result_df_defaults, pd.DataFrame)
    assert len(result_df_defaults) == len(sample_df)

    expected_metric_cols_defaults = [
        "pct_hiragana", "num_sents", "cwr", "sentlen_mean",
        "ttr_orig",
        "sentdepths_mean", "sentchunks_mean", "chunktokens_mean"
    ]
    for col_name in expected_metric_cols_defaults:
        assert col_name in result_df_defaults.columns, f"Expected metric column '{col_name}' not in result_df_defaults for default run"

    # Abst and JIWC keys should be absent
    assert "abst_top5_mean" not in result_df_defaults.columns
    assert "abst_max" not in result_df_defaults.columns
    assert "emo" not in result_df_defaults.columns # from our sample jiwc_df

    assert result_df_defaults["num_sents"].tolist() == pytest.approx(expected_num_sents) # num_sents should still be correct

    # --- Test for AssertionError with invalid column ---
    with pytest.raises(AssertionError, match="text_col_nonexistent is not found in the input data"):
        limco.from_df(sample_df, 'text_col_nonexistent', stopwords, awd, jiwc_df)
