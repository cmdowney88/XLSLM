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
languages. The original parallel Spanish sentences have been omitted

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

`train.anlp` is composed of the concatenated training sets of the 10 languages.
`dev.anlp` is similarly composed of the concatenated development sets

### Training Sets

| Language | File | Lines | Total Tokens | Unique Tokens | Total Characters | Unique Characters | Mean Token Length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | train.anlp | 223,117 | 2,368,505 | 347,438| 16,757,586 | 252 | 7.08 |
| Asháninka | train.cni | 3,883 | 26,096 | 12,490 | 232,494 | 65 | 8.91 |
| Aymara | train.aym | 6,424 | 96,075 | 33,590 | 624,608 | 156 | 6.50 |
| Bribri | train.bzd | 7,508 | 41,141 | 7,858 | 167,531 | 65 | 4.07 |
| Guaraní | train.gug | 26,002 | 405,449 | 44,763 | 2,718,442 | 120 | 6.70 |
| Hñähñu | train.oto | 4,889 | 72,280 | 8,664 | 275,696 | 90 | 3.81 |
| Nahuatl | train.nah | 16,061 | 351,703 | 53,743 | 1,984,687 | 102 | 5.64 |
| Quechua | train.quy | 120,090 | 1,160,302 | 145,914 | 9,625,591 | 114 | 8.30 |
| Rarámuri | train.tar | 14,720 | 103,745 | 15,691 | 398,898 | 74 | 3.84 |
| Shipibo Konibo | train.shp | 14,592 | 62,850 | 17,642 | 397,510 | 56 | 6.32 |
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
