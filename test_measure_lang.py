import pytest

import measure_lang as ml

TEXT = """ここは駅から程よい距離にある住宅街である。
この街は市内でも最も少子高齢化が進んでいる街として有名だった。
その現状として小学校が合併され、二校になったり、街の南の中学校のクラスは1クラスになりそうなところである。
しかも、街の北にある中学校でさえも2クラスになろうとしている。
だが、いつもこの街にあるスーパーで毎年行われる納涼祭はとても盛り上がり、この街だけではおさまらず、他のところから来ている人も多数いる。
それ以外の時は大抵静かだ。
高齢者が優雅にのびのびとくらいしている。
いつも静かなこの街は今日も静かだった。
そして、いつも通りの日常が今日も送られる。
はずだった、、、"""


def test_measure_sents():
    assert ml.measure_sents(TEXT) == pytest.approx(
        [10.0, 28.4, 18.34969, 8.0, 19.25, 21.0, 31.0, 67.0]
    )


def test_count_conversations():
    assert ml.count_conversations(TEXT) == pytest.approx(0.0)


def test_count_charcat():
    assert ml.count_charcat(TEXT) == pytest.approx([0.61267606, 0.03873239, 0.26760563])


def test_measure_pos():
    assert ml.measure_pos(TEXT) == pytest.approx(
        [
            # 2.0,
            # 0.0,
            # 2.0,
            # 2.0,
            # 2.0,
            # 2.0,
            # 2.0,
            0.448863636,
            0.221590909,
            0.521739130,
            0.0,
            0.460227273,
            6.10560473,
            0.849910600,
            2.67470438,
            0.0290281142,
            -0.193376014,
            60480.9502,
            34.4493615,
        ]
    )


def test_measure_abst():
    assert ml.measure_abst(TEXT) == pytest.approx([3.0479999999999996, 3.19])


def test_bunmatsu():
    assert ml.detect_bunmatsu(TEXT) == pytest.approx(0.0)
