# lang_identification
A simple project to identify language of a text

## Introduction

It is a simple project of language identification using n-gram method written on Python.

The n-gram method is implemented in ``lang_iden.py`` file.

Other method based on counting stopwords is available in ``baseline.py`` file for comparison. To run this file, ``nltk`` needs to be installed first.

## Usage

The implementation is one-shot run, however it is very easy to extend for more general purpose.

```{bash}
python lang_iden.py --h
```

For instance

```{bash}
python lang_iden.py --n=2 --snippet_len=10
```

By default the program will run with the texts provided in ``train_data`` and ``test_data`` directory.

If you want to predict a particular text:

```{python}
>>>python
>>>from lang_iden import *
>>>lang_profiles = train ()
>>>predict (lang_profiles, "This is a new text that I want to predict")
{'fr': 3004550.0, 'de': 3003701.0, 'en': 3001772.0, 'it': 3005339.0}
```

The text should be written in English, because the distance to English profile (3001772) is minimum.

You can get this value

```{python}
distances = predict (lang_profiles, "This is a new text that I want to predict")
min(distances, key = distances.get)
```

## Adding language

Right now, English, German, Italian and French are supported. If you want to add more language, just follow the structure of ``train_data`` folder.

For instance, if you want to add Portugese.

- create a folder ``pt`` inside ``train_data``.
- place one or more Portuges texts (in ``.txt`` format) in this new folder.

That's it.