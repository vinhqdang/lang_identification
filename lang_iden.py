'''
    A simple language identification program
    Using n-gram to detect language
    without using any existing library.

    n-gram is used because:
    - it does not require linguistic knowledge.
    - easy to train
    - easy to scale (just add data)
'''

import argparse
import glob, os
import math
import operator
import timeit

# for compared baseline models
from baseline import *

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Language identification of a text.")

    parser.add_argument('--n', type = int, nargs='?', default=2,
                        help='n parameter using in n-gram setting.')

    parser.add_argument('--snippet_len', type = int, nargs='?', default=200,
                        help='The length of text snippet to predict')

    parser.add_argument('--train_dir', nargs='?', default='./train_data/',
                        help='Train directory that contains text files by language')

    parser.add_argument('--test_dir', nargs='?', default='./test_data/',
                        help='Test directory that contains text files by language')

    return parser.parse_args()

args = parse_args()

langs = ['en','fr','it','de']

def ngrams (text, n):
    """
    @brief      Return n-gram list from text
    
    @param      text  The text
    @param      n     The parameter n
    
    @return     { list of n-grams }
    """
    return zip(*[text[i:] for i in range(n)])

def normalize (text):
    """
    @brief      simple normalize a text: remove duplicate whitespace, lowering text
    
    @param      text  The text
    
    @return     the normalized text
    """
    res = text.lower()
    res = ' '.join(res.split())
    return res

def ngram_stats (ngrams):
    """
    @brief      return a sorted ngram by frequency
    
    @param      ngrams  The list of n-gram
    
    @return     { a sorted list }
    """
    ngrams_statistics = {}
    for ngram in ngrams:
        if not ngrams_statistics.has_key(ngram):
            ngrams_statistics.update({ngram:1})
        else:
            ngram_occurrences = ngrams_statistics[ngram]
            ngrams_statistics.update({ngram:ngram_occurrences+1})
    ngrams_statistics_sorted = sorted(ngrams_statistics.iteritems(),
                            key=operator.itemgetter(1),
                            reverse=True)
    return ngrams_statistics_sorted

def dist_ngram (ngram_stat_1, ngram_stat_2):
    """
    @brief      calculate distance between two ngram sorted lists
    
    @param      ngram_stat_1  The trained ngram stats
    @param      ngram_stat_2  The tested ngram stats
    
    @return     { distance }
    """
    MAX_DISTANCE = 1000000
    sum_distance = 0.0

    sorted_ngram_1 = [row[0] for row in ngram_stat_1]
    sorted_ngram_2 = [row[0] for row in ngram_stat_2]

    for i in range(len(sorted_ngram_2)):
        ngram = sorted_ngram_2[i]
        if ngram in sorted_ngram_1:
            sum_distance += abs (i - sorted_ngram_1.index(ngram))            
        else:
            sum_distance += MAX_DISTANCE
    return sum_distance

def train ():
    """
    @brief      get n-gram of trained data
    
    @return     n-gram statistics of each language
    """
    contents = {}
    for lang in langs:
        lang_content = ''
        text_dir = args.train_dir + lang + '/'
        for file_name in os.listdir(text_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join (text_dir, file_name)
                f = open (file_path)
                cur_content = f.read ()
                lang_content = lang_content + cur_content
                f.close ()
        contents[lang] = normalize (lang_content)
    lang_stats = {}
    for lang in langs:
        lang_stats [lang] = ngram_stats (ngrams (contents[lang], n = args.n))
    return lang_stats

def predict (lang_profiles, text):
    """
    @brief      test the n-gram program

    @param      lang_profiles the language statistics comes from train() function

    @param      text the text whose language needed to be identified
    
    @return     the predicting language
    """

    test_ngrams = ngrams (text, n=args.n)
    test_stats = ngram_stats (test_ngrams)
    distances = {}
    for lang in lang_profiles.keys ():
        ngram_stat_1 = lang_profiles [lang]
        ngram_stat_2 = test_stats
        distances [lang] = dist_ngram (ngram_stat_1, ngram_stat_2)
    return distances

def main ():
    print ('Training')
    lang_stats = train ()

    print ('Testing full content')
    contents = {}
    for lang in langs:
        text_dir = args.train_dir + lang + '/'
        for file_name in os.listdir(text_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join (text_dir, file_name)
                f = open (file_path)
                cur_content = f.read ()

                cur_content = normalize (cur_content)

                distances = predict (lang_stats, text = cur_content)

                print ('Correct language: ' + lang)
                start_time = timeit.default_timer()
                print ('Predict of n-gram: ' + min(distances, key=distances.get))
                elapsed = timeit.default_timer() - start_time
                print ('Time = ' + str (elapsed))

                start_time = timeit.default_timer()
                print ('Predict of baseline: ' + detect_language (cur_content))
                elapsed = timeit.default_timer() - start_time
                print ('Time = ' + str (elapsed))
                print ('******')

                f.close ()

    print ('Testing with only text snippets')
    results_ngram = {}
    results_baseline = {}
    for lang in langs:
        cur_results_ngram = {}
        cur_results_baseline = {}
        ngram_time = 0.0
        baseline_time = 0.0

        NUM_PREDICT = 100

        text_dir = args.train_dir + lang + '/'
        for file_name in os.listdir(text_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join (text_dir, file_name)
                f = open (file_path)
                cur_content = f.read ()

                cur_content = normalize (cur_content)

                num_snippets = len (cur_content.split()) / args.snippet_len

                for i in range (num_snippets):
                    # print ('Snippet number: ' + str (i))
                    if i >= NUM_PREDICT:
                        break

                    text = ' '.join(cur_content.split()[i*args.snippet_len:(i+1)*args.snippet_len-1])

                    # predict using ngram
                    start_time = timeit.default_timer()
                    distances = predict (lang_stats, text = text)                    
                    pre = min(distances, key=distances.get)
                    elapsed = timeit.default_timer() - start_time

                    ngram_time += elapsed

                    if pre not in cur_results_ngram.keys():
                        cur_results_ngram [pre] = 1
                    else:
                        cur_results_ngram.update({pre:cur_results_ngram[pre] + 1})

                    # predict using baseline
                    start_time = timeit.default_timer()
                    pre = detect_language (text)
                    elapsed = timeit.default_timer() - start_time

                    baseline_time += elapsed

                    if pre not in cur_results_baseline.keys():
                        cur_results_baseline [pre] = 1
                    else:
                        cur_results_baseline.update({pre:cur_results_baseline[pre] + 1})

                f.close ()

        results_ngram [lang] = cur_results_ngram
        results_baseline [lang] = cur_results_baseline

    print ('Prediction using n-grams')
    print (results_ngram)
    print ('average running time = ' + str(ngram_time / NUM_PREDICT / len(langs)))
    print ('-------')
    print ('Prediction using stopwords')
    print (results_baseline)
    print ('average running time = ' + str(baseline_time / NUM_PREDICT / len(langs)))

if __name__ == '__main__':
    main()
