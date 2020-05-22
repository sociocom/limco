# Linguistic measure collection

A collection of the stylometric measures for authorship detection that are suggested to have relationships to the attitudes and psychological tendencies of authors.
This library can calculate 12 types of stylometrics, based on Japanese text metrics organised in Asaishi (2017).

Compatible natural languages:

- Japanese
- English

## Installation

> NOTE: `pip`-installable packaging is work-in-progress; sorry for the inconvenience

### Requirements

- Python 3.6+
  - **f-string** (PEP 536):
  - **Type Hints** (PEP 484):
- (Your kind patience)

### Dependencies

If you are using `pipenv`, `pipenv install` will create a virtual environment for this library.
Otherwise, `pip install -r requirements.txt` will suffice.

This library also depends on the following external commands for Japanese processing:

- **MeCab** (for Japanese text processing):
    - **[mecab-ipadic-neologd](https://github.com/neologd/mecab-ipadic-neologd)** (for better Japanese NER):
- **CaboCha** (for `analyse_parseddoc.py`):

### Resources

Furthermore, for Japanese metrics, the following linguistic resources are required.

- [AWD-J EX](http://sociocom.jp/~data/2019-AWD-J/) as the file name `AWD-J_EX.txt`
- JIWC as the file name `2017-11-JIWC.csv`
- Japanese stopwords as the file name `stopwords_jp.txt`

Put the above in the `data/` folder.

> I will make them optional in future.
> Also, the location of resources will be made configurable.


## Usage

### `measure_lang.py`

You can generate table data from a text file by executing:

```
python measure_lang.py [csv/excel]
```

When you input `your_data.csv`, `your_data.measured.csv` will be created in the same location of the input file.

You can also use each measure by importing like:

```python
import measure_lang as ml

def your_fancy_func(string):
    # ...
    res = ml.calc_ttrs(string)
    # ...
```

All functions read a string (a long passages with new lines are OK.).
If you want to apply them to a list of strings, you need to iterate them over the list.

The detailed usage instruction for each measure will be available in its docstring (work-in-progress).

### `analyse_parseddoc.py`

(the documentation here is work-in-progress)

Prepare:
- one sentence per line
- documents are split by one blank line

Run:
```
cabocha -f1 your_text.txt > your_text.cabocha.txt
python analyse_parseddoc.py your_text.cabocha.txt
```


## Measures

```python
import measure_lang as ml
import analyse_parseddoc as ap
```

- **Percentages of character types** (`ml.count_charcat`):
The ratio of hiragana, katakana, and kanji (Chinese characters) to
the characters in text, respectively.

- **Type Token Ratio (TTR)** (`ml.calc_ttrs`):
The ratio of the different words to the total number of words in text.

- **Percentage of content words** (`ml.measure_pos`):
The ratio of content words (i.e., nouns, verbs, adjectives, and
adverbs) to the total number of words in text.

- **Modifying words and Verb Ratio (MVR)** (`ml.measure_pos`):
The ratio of verbs to adjectives, adverbs, and conjunctions for the
words in text. It has been used as one of the indicators of
author estimation.

- **Percentage of proper nouns** (`ml.measure_pos`):
The ratio of proper nouns (named entities) to all words in text.

- **Word abstraction** (`ml.measure_abst`):
The abstraction degrees of the words in text. We specifically
used the maximum value of the most abstract word, and the average of
the top five abstract words. The abstraction degrees were obtained
from the Japanese word-abstraction dictionary [AWD-J EX](http://sociocom.jp/~data/2019-AWD-J/).

- **Ratios of emotional words** (`ml.calc_jiwc`):
The ratios, to all the words in text, of the words that are
associated with each of the seven kinds of emotions: sadness,
anxiety, anger, disgust, trust, surprise, and happiness. The seven
values are transformed to meet the property of probability (each
value spans between 0 and 1; the sum of all values is to be 1). The
degree of association with emotion was determined according to the
Japanese emotional-word dictionary JIWC.

- **Number of sentences** (`ml.measure_sents`):
The total number of sentences that make up text.

- **Length of sentences** (`ml.measure_sents`):
Descriptive statistics (mean, standard deviation, interquartile,
minimum, and maximum) for the number of characters in each sentence
that constitutes text. In particular, the average sentence
length has been suggested to be linked to the writer’s creative
attitude and personality .

- **Percentage of conversational sentences** (`ml.count_conversations`):
Percentage of the total number of conversational sentences contained
in text.

- **Depth of syntax tree** (`ap.analyse_dep`):
Descriptive statistics calculated for the depth of the dependency
tree for each sentence in text.

- **Mean of the number of chunks per sentence** (`ap.analyse_dep`):
Descriptive statistics calculated for the average values of the
number of chunks for each sentence in text.

- **Mean of the words per chunk** (`ap.analyse_dep`):
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
| Depth of syntax tree                     | mean, standard deviation, interquartile, minimum, and maximum (real number) |
| Mean of the number of chunks per sentence | mean, standard deviation, interquartile, minimum, and maximum (real number) |
| Mean of the words per chunk               | mean, standard deviation, interquartile, minimum, and maximum (real number) |

---
## References

- [Asaishi, 2007]: 浅石卓真. (2017). テキストの特徴を計量する指標の概観. 日本図書館情報学会誌, 63(3), 159–169. https://doi.org/10.20651/jslis.63.3_159
