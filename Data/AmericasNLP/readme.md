# AmericasNLP Indigenous Training Languages

This directory contains a preparation of the training data from
[_The First Workshop on NLP for Indigenous Languages of the Americas_](http://turing.iimas.unam.mx/americasnlp/),
("AmericasNLP"). The data was originally curated for an shared task on Machine
Translation for low-resource languages, and includes data from ten Indigenous
languages of Central and South America. See the
[accompanying paper](https://aclanthology.org/2021.americasnlp-1.23/) for more
details on the original data collection

The preparation of the data here is specialized for training a multilingual
language model on Indigenous languages. As such, combined training and
development files have been compiled containing the data from all ten Indigenous
languages. The original parallel Spanish sentences have been omitted. For
Asháninka and Shipibo Konibo, additional monolingual-only data was available and
linked to the AmericasNLP repository. This monolingual data was compiled by
[Bustamente, Oncevay, and Zariquiey (2020)](https://www.aclweb.org/anthology/2020.lrec-1.356)

## Included Languages
- [Asháninka (cni)](https://en.wikipedia.org/wiki/Asháninka_language)
- [Aymara (aym)](https://en.wikipedia.org/wiki/Aymara_language)
- [Bribri (bzd)](https://en.wikipedia.org/wiki/Bribri_language)
- [Guaraní (gug)](https://en.wikipedia.org/wiki/Guarani_language)
- [Hñähñu (oto)](https://en.wikipedia.org/wiki/Otomi_language)
- [Nahuatl (nah)](https://en.wikipedia.org/wiki/Nahuatl)
- [Quechua (quy)](https://en.wikipedia.org/wiki/Southern_Quechua)
- [Rarámuri (tar)](https://en.wikipedia.org/wiki/Tarahumara_language)
- [Shipibo Konibo (shp)](https://en.wikipedia.org/wiki/Shipibo_language)
- [Wixarika (hch)](https://en.wikipedia.org/wiki/Huichol_language)

## Data Composition

`train.anlp` is composed of the concatenated training sets of the 10 languages
(including the monolingual data for Asháninka and Shipibo Konibo). `dev.anlp` is
similarly composed of the concatenated development sets. Because the original
collection is heavily skewed towards Quechua, we also create
`train_balanced.anlp`, which includes a downsampled set of the Quechua examples

### Training Sets

| Language | File | Lines | Total Tokens | Unique Tokens | Total Characters | Unique Characters | Mean Token Length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | train.anlp | 259,207 | 2,682,609 | 400,830 | 18,982,453 | 253 | 7.08 |
| All | train_balanced.anlp | 171,830 | 1,839,631 | 320,331 | 11,981,011 | 241 | 6.51 |
| All | train_downsampled.anlp | 120,145 | 1,284,440 | 255,392 | 8,365,710 | 221 | 6.51 |
| Asháninka | train.cni | 3,883 | 26,096 | 12,490 | 232,494 | 65 | 8.91 |
| Asháninka | train_1.mono.cni | 12,010 | 99,329 | 27,963 | 919,897 | 48 | 9.26 |
| Asháninka | train_2.mono.cni | 593 | 4,515 | 2,325 | 42,093 | 41 | 9.32 |
| Aymara | train.aym | 6,424 | 96,075 | 33,590 | 624,608 | 156 | 6.50 |
| Bribri | train.bzd | 7,508 | 41,141 | 7,858 | 167,531 | 65 | 4.07 |
| Guaraní | train.gug | 26,002 | 405,449 | 44,763 | 2,718,442 | 120 | 6.70 |
| Hñähñu | train.oto | 4,889 | 72,280 | 8,664 | 275,696 | 90 | 3.81 |
| Nahuatl | train.nah | 16,684 | 351,702 | 53,743 | 1,984,685 | 102 | 5.64 |
| Quechua | train.quy | 120,145 | 1,158,273 | 145,899 | 9,621,816 | 114 | 8.31 |
| Quechua | train_downsampled.quy | 32,768 | 315,295 | 64,148 | 2,620,374 | 95 | 8.31 |
| Rarámuri | train.tar | 14,720 | 103,745 | 15,691 | 398,898 | 74 | 3.84 |
| Shipibo Konibo | train.shp | 14,592 | 62,850 | 17,642 | 397,510 | 56 | 6.32 |
| Shipibo Konibo | train_1.mono.shp | 22,029 | 205,866 | 29,534 | 1,226,760 | 61 | 5.96 |
| Shipibo Konibo | train_2.mono.shp | 780 | 6,424 | 2,618 | 39,894 | 39 | 6.21 |
| Wixarika | train.hch | 8,948 | 48,864 | 17,357 | 332,129 | 67 | 6.80 |

### Development Sets

| Language | File | Lines | Total Tokens | Unique Tokens | Total Characters | Unique Characters | Mean Token Length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | dev.anlp | 9,122 | 79,901 | 27,597 | 485,179 | 105 | 6.07 |
| Asháninka | dev.cni | 883 | 6,070 | 3,100 | 53,401 | 63 | 8.80 |
| Aymara | dev.aym | 996 | 7,080 | 3,908 | 53,852 | 64 | 7.61 |
| Bribri | dev.bzd | 996 | 12,974 | 2,502 | 50,573 | 73 | 3.90 |
| Guaraní | dev.gug | 995 | 7,191 | 3,181 | 48,516 | 70 | 6.75 |
| Hñähñu | dev.oto | 599 | 5,069 | 1,595 | 22,712 | 69 | 4.48 |
| Nahuatl | dev.nah | 672 | 4,300 | 1,839 | 31,338 | 56 | 7.29 |
| Quechua | dev.quy | 996 | 7,406 | 3,826 | 58,005 | 62 | 7.83 |
| Rarámuri | dev.tar | 995 | 10,377 | 2,964 | 55,644 | 48 | 5.36 |
| Shipibo Konibo | dev.shp | 996 | 9,138 | 3,296 | 54,996 | 65 | 6.02 |
| Wixarika | dev.hch | 994 | 10,296 | 3,895 | 56,142 | 62 | 5.45 |
