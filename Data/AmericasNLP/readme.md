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
- [Ashaninka](https://en.wikipedia.org/wiki/Asháninka_language)
- [Aymara](https://en.wikipedia.org/wiki/Aymara_language)
- [Bribri](https://en.wikipedia.org/wiki/Bribri_language)
- [Guarani](https://en.wikipedia.org/wiki/Guarani_language)
- [Hñähñu](https://en.wikipedia.org/wiki/Otomi_language)
- [Nahuatl](https://en.wikipedia.org/wiki/Nahuatl)
- [Quechua](https://en.wikipedia.org/wiki/Southern_Quechua)
- [Raramuri](https://en.wikipedia.org/wiki/Tarahumara_language)
- [Shipibo Konibo](https://en.wikipedia.org/wiki/Shipibo_language)
- [Wixarika](https://en.wikipedia.org/wiki/Huichol_language)

## Data Composition

`train.anlp` is composed of the concatenated training sets of the 10 languages. `dev.anlp` is similarly composed of the concatenated development sets

| Language | File | Lines | Total Tokens | Unique Tokens | Total Characters | Unique Characters | Mean Token Length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | train.anlp | 225,958 | 2,376,865 | 347,897| 16,787,156 | 253 | 7.06 |
| All | dev.anlp | 9,122 | 79,901 | 27,597 | 485,179 | 105 | 6.07 |
| Ashaninka | train.cni |
