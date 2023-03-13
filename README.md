# limco: LInguistic Measure COllection

A collection of the stylometric measures for authorship detection that are suggested to have relationships to the attitudes and psychological tendencies of authors.
This library can calculate 12 types of stylometrics, based on Japanese text metrics organised in [Asaishi (2017)](https://doi.org/10.20651/jslis.63.3_159).

Currently, only compatible with Japanese inputs.

Please cite [Manabe et al. (2021)](https://doi.org/10.2196/29500) (see the bottom references) if you use this library.

This collection needs more linguistic mesures to be complete, of course.
Issues and Pull Requests are welcome!

limco は著者の態度や心理的特性と関連があるとされるスタイロメトリック言語指標のコレクションです。
[浅石 (2017)](https://doi.org/10.20651/jslis.63.3_159) にまとめられている日本語の言語指標をベースにしています。

現在、日本語のみに対応しています。

このライブラリを使用した場合は、[Manabe et al. (2021)](https://doi.org/10.2196/29500) (下記参考文献) を引用してください。

本ライブラリは言語指標のコレクションとしてはまだまだ不完全です．Issue や PR を歓迎します！

## Installation / インストール

You need Python 3.9 or later.

```shell
pip install limco
```

## Usage / 使い方

Specify the path to the CSV or Excel file and the column name of the text to be analysed.

```shell
limco path/to/file.csv text
```

### Optional Resources / オプションのリソース

Furthermore, the following linguistic resources are can be used.
Specify the paths to the resources with the following options.

さらに，以下の言語リソースを使用することができます。
次のオプションでリソースへのパスを指定してください。

- `--sw`: Japanese stopwords (1 行 1 単語＝原形の形式のテキストファイル)
- `--awd`: [日本語抽象度辞書 AWD-J](https://sociocom.naist.jp/awd-j/) `-EX` データを使用してください (e.g. [`AWD-J_EX.txt`](http://sociocom.jp/~data/2019-AWD-J/data/AWD-J_EX.txt))
- `--jiwc`: [日本語感情表現辞書 JIWC](https://sociocom.naist.jp/jiwc-dictionary/) `-A` データを使用してください (e.g. [`JIWC-A_2019.csv`](https://github.com/sociocom/JIWC-Dictionary/blob/master/ver_2019/JIWC-A_2019.csv))

## Linguistic Measures / 言語指標

The following linguistic measures are implemented. / 次の言語指標が実装されています。

- **Percentages of character types / 文字種の割合**:
  The ratio of hiragana, katakana, and kanji (Chinese characters) to
  the characters in text, respectively. / ひらがな、カタカナ、漢字（中国語の文字）のそれぞれの、総文字数に対する割合。

- **Type Token Ratio (TTR)**:
  The ratio of the distinct words to the total number of words in text.
  We cover several variants of TTRs. / 異なり語数（単語の種類数）を、総単語数で割った値。いくつかの補正バリエーションを実装。

- **Percentage of content words / 内容語の割合**:
  The ratio of content words (i.e., nouns, verbs, adjectives, and
  adverbs) to the total number of words in text. / 内容語（名詞、動詞、形容詞、副詞）のそれぞれの、総単語数に対する割合。

- **Modifying words and Verb Ratio (MVR) / 相の類に対する用の類の割合**:
  The ratio of verbs to adjectives, adverbs, and pre-noun adjectival for the
  words in text. It has been used as one of the indicators of
  author estimation. / 用の類（形容詞、副詞、連体詞）に対する動詞の割合。著者推定の指標として用いられている。

- **Percentage of proper nouns / 固有名詞の割合**:
  The ratio of proper nouns (named entities) to all words in text. / 固有名詞のそれぞれの、総単語数に対する割合。

- **Word abstractness / 単語抽象度**:
  The abstraction degrees of the words in text. We specifically
  used the maximum value of the most abstract word, and the average of
  the top five abstract words. The abstraction degrees were obtained
  from the Japanese word-abstraction dictionary [AWD-J EX](http://sociocom.jp/~data/2019-AWD-J/). / 単語抽象度辞書 AWD-J EX から得られる単語抽象度。最も抽象的な単語の最大値、上位 5 語の平均値を使用。

- **Emotion scores / 感情スコア**:
  The ratios, to all the words in text, of the words that are
  associated with each of the seven kinds of emotions: sadness,
  anxiety, anger, disgust, trust, surprise, and joy. The seven
  values are transformed to meet the property of probability (each
  value spans between 0 and 1; the sum of all values is to be 1). The
  degree of association with emotion was determined according to the
  Japanese emotional-word dictionary JIWC. / 感情辞書 JIWC から得られる感情スコア。7 種類の感情（悲しみ、不安、怒り、嫌悪、信頼、驚き、喜び）に対するそれぞれの単語の割合。7 つの値は確率の性質を満たすように変換されている（各値は 0 から 1 の間にあり、合計は 1 になる）。

- **The number of sentences / 総文数**:
  The total number of sentences that make up text. / 文の総数。

- **Length of sentences / 文の長さ**:
  Descriptive statistics (mean, standard deviation, interquartile,
  minimum, and maximum) for the number of characters in each sentence
  that constitutes text. In particular, the average sentence
  length has been suggested to be linked to the writer’s creative
  attitude and personality. / 文の長さの統計量（平均、標準偏差、四分位範囲、最小値、最大値）。特に、平均文長は著者の創造的態度や性格と関連しているとされている。

- **Percentage of conversational sentences / 会話文の割合**:
  Percentage of the total number of conversational sentences contained
  in text. / 会話文（「」『』で括られたテキスト）の総文数に対する割合。

- **Depth of syntax tree / 係り受け構造の深さ**:
  Descriptive statistics calculated for the depth of the dependency
  tree for each sentence in text. / 係り受け構造の深さの統計量。

- **The number of chunks per sentence / 文ごとの文節数**:
  Descriptive statistics calculated for the average values of the
  number of chunks for each sentence in text. / 文ごとの文節数の統計量。

- **The tokens per chunk / 文節ごとの単語数**:
  Descriptive statistics calculated for the average values of the
  number of words per chunk in text. / 文節ごとの単語数の統計量。

### Summary table

| Stylometric                            | Sub-measures (value format)                                                                        |
| :------------------------------------- | :------------------------------------------------------------------------------------------------- |
| Percentages of character types         | Hiragana, katakana, and kanji (Chinese characters) (%)                                             |
| Type Token Ration (TTR)                | Plain TTR, Guiraud's R, Herdan's C_H, Rubet's k, Maas's a^2, Tuldava's LN, Brunet's W, Dugast's U, |
| Percentages of content words           |                                                                                                    |
| Modifying words and Verb Ratio (MVR)   | (%)                                                                                                |
| Percentage of proper nouns             | (%)                                                                                                |
| Word abstractness                      | The maximum, and the average of the top five abstract words (real number)                          |
| Emotion scores                         | sadness, anxiety, anger, disgust, trust, surprise, and joy (%)                                     |
| The number of sentences                | (integer)                                                                                          |
| Length of sentences                    | mean, standard deviation, interquartile, minimum, and maximum (real number)                        |
| Percentage of conversational sentences | (%)                                                                                                |
| Depth of syntax tree                   | mean, standard deviation, interquartile, minimum, and maximum (real number)                        |
| The number of chunks per sentence      | mean, standard deviation, interquartile, minimum, and maximum (real number)                        |
| The tokens per chunk                   | mean, standard deviation, interquartile, minimum, and maximum (real number)                        |

---

## References

- [Asaishi, 2017]: 浅石卓真. 2017. テキストの特徴を計量する指標の概観. 日本図書館情報学会誌, 63(3), 159–169. https://doi.org/10.20651/jslis.63.3_159
- [Manabe+, 2021]: Masae Manabe, Kongmeng Liew, Shuntaro Yada, Shoko Wakamiya, Eiji Aramaki. 2021. Estimation of Psychological Distress in Japanese Youth Through Narrative Writing: Text-Based Stylometric and Sentiment Analyses. JMIR Formative Research, 5(8):e29500. https://doi.org/10.2196/29500

## Developer

- [Shuntaro Yada](https://shuntaroy.com)
- Itaru Ota
