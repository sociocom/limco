# limco: LInguistic Measure COllection

A collection of the stylometric measures for authorship detection that are suggested to have relationships to the attitudes and psychological tendencies of authors.
This library can calculate 12 types of stylometrics, based on Japanese text metrics organised in [Asaishi (2017)](https://doi.org/10.20651/jslis.63.3_159).

Currently, only compatible with Japanese inputs.

Please cite [Manabe et al. (2021)](https://doi.org/10.2196/29500) (see the bottom references) if you use this library.

limco は著者の態度や心理的特性と関連があるとされるスタイロメトリック言語指標のコレクションです。
[浅石 (2017)](https://doi.org/10.20651/jslis.63.3_159) にまとめられている日本語の言語指標をベースにしています。

現在、日本語のみに対応しています。

このライブラリを使用した場合は、[Manabe et al. (2021)](https://doi.org/10.2196/29500) (下記参考文献) を引用してください。

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
- `--awd`: [日本語抽象度辞書 AWD-J](https://sociocom.naist.jp/awd-j/)
- `--jiwc`: [日本語感情表現辞書 JIWC](https://sociocom.naist.jp/jiwc-dictionary/)

## Linguistic Measures / 言語指標

The following linguistic measures are implemented. / 次の言語指標が実装されています。

- **Percentages of character types**:
  The ratio of hiragana, katakana, and kanji (Chinese characters) to
  the characters in text, respectively.

- **Type Token Ratio (TTR)**:
  The ratio of the different words to the total number of words in text.

- **Percentage of content words**:
  The ratio of content words (i.e., nouns, verbs, adjectives, and
  adverbs) to the total number of words in text.

- **Modifying words and Verb Ratio (MVR)**:
  The ratio of verbs to adjectives, adverbs, and conjunctions for the
  words in text. It has been used as one of the indicators of
  author estimation.

- **Percentage of proper nouns**:
  The ratio of proper nouns (named entities) to all words in text.

- **Word abstraction**:
  The abstraction degrees of the words in text. We specifically
  used the maximum value of the most abstract word, and the average of
  the top five abstract words. The abstraction degrees were obtained
  from the Japanese word-abstraction dictionary [AWD-J EX](http://sociocom.jp/~data/2019-AWD-J/).

- **Ratios of emotional words**:
  The ratios, to all the words in text, of the words that are
  associated with each of the seven kinds of emotions: sadness,
  anxiety, anger, disgust, trust, surprise, and happiness. The seven
  values are transformed to meet the property of probability (each
  value spans between 0 and 1; the sum of all values is to be 1). The
  degree of association with emotion was determined according to the
  Japanese emotional-word dictionary JIWC.

- **Number of sentences**:
  The total number of sentences that make up text.

- **Length of sentences**:
  Descriptive statistics (mean, standard deviation, interquartile,
  minimum, and maximum) for the number of characters in each sentence
  that constitutes text. In particular, the average sentence
  length has been suggested to be linked to the writer’s creative
  attitude and personality .

- **Percentage of conversational sentences**:
  Percentage of the total number of conversational sentences contained
  in text.

- **Depth of syntax tree**:
  Descriptive statistics calculated for the depth of the dependency
  tree for each sentence in text.

- **Mean of the number of chunks per sentence**:
  Descriptive statistics calculated for the average values of the
  number of chunks for each sentence in text.

- **Mean of the words per chunk**:
  Descriptive statistics calculated for the average values of the
  number of words per chunk in text.

### Summary table

| Stylometric                               | Sub-measures (value format)                                                 |
| :---------------------------------------- | :-------------------------------------------------------------------------- |
| Percentages of character types            | Hiragana, katakana, and kanji (Chinese characters) (%)                      |
| Type Token Ration (TTR)                   | (%)                                                                         |
| Percentages of content words              | (%)                                                                         |
| Modifying words and Verb Ratio (MVR)      | (%)                                                                         |
| Percentage of proper nouns                | (%)                                                                         |
| Word abstraction                          | The maximum, and the average of the top five abstract words (real number)   |
| Ratios of emotional words                 | sadness, anxiety, anger, disgust, trust, surprise, and happiness (%)        |
| Number of sentences                       | (integer)                                                                   |
| Length of sentences                       | mean, standard deviation, interquartile, minimum, and maximum (real number) |
| Percentage of conversational sentences    | (%)                                                                         |
| Depth of syntax tree                      | mean, standard deviation, interquartile, minimum, and maximum (real number) |
| Mean of the number of chunks per sentence | mean, standard deviation, interquartile, minimum, and maximum (real number) |
| Mean of the words per chunk               | mean, standard deviation, interquartile, minimum, and maximum (real number) |

---

## References

- [Asaishi, 2017]: 浅石卓真. 2017. テキストの特徴を計量する指標の概観. 日本図書館情報学会誌, 63(3), 159–169. https://doi.org/10.20651/jslis.63.3_159
- [Manabe+, 2021]: Masae Manabe, Kongmeng Liew, Shuntaro Yada, Shoko Wakamiya, Eiji Aramaki. 2021. Estimation of Psychological Distress in Japanese Youth Through Narrative Writing: Text-Based Stylometric and Sentiment Analyses. JMIR Formative Research, 5(8):e29500. https://doi.org/10.2196/29500
