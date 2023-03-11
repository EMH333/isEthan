import csv
import os
import re
import gzip


def get_combinations():
    combinations = list()
    alphabet = "abcdefghijklmnopqrstuvwxyz_,.()"
    for a in range(len(alphabet)):
        for b in range(len(alphabet)):
            for c in range(len(alphabet)):
                combinations.append(alphabet[a] + alphabet[b] + alphabet[c])
    return combinations


def split(s, num):
    return [s[start:start + num] for start in range(0, len(s), num)]


def analyze(text, wordmap=None):
    if wordmap is None:
        wordmap = dict()
    # split into 3grams then add to wordmap
    for x in split(text, 3):
        try:
            wordmap[x] += 1
        except KeyError:
            wordmap[x] = 1
    return wordmap


combos = get_combinations()


def record(out, is_ethan, data):
    row = list()
    if is_ethan:
        row.append("TRUE")
    else:
        row.append("FALSE")
    for i in combos:
        try:
            row.append(data[i])
        except KeyError:
            row.append("0")
    out.writerow(row)
    return


def iterate_analyse(output_csv, full_text, is_ethan):
    length = 700
    # every 700 characters do a split
    for text in split(full_text, length):
        hashmap = analyze(text)
        record(output_csv, is_ethan, hashmap)

        text = " " + text
        hashmap = analyze(text, hashmap)
        record(output_csv, is_ethan, hashmap)
        hashmap_new = analyze(text)
        record(output_csv, is_ethan, hashmap_new)

        text = " " + text
        hashmap = analyze(text, hashmap)
        record(output_csv, is_ethan, hashmap)
        hashmap_new = analyze(text)
        record(output_csv, is_ethan, hashmap_new)
    
    # do an offset of half the length to mix up the data a bit more
    if len(full_text) > length * 1.5:
        iterate_analyse(output_csv, full_text[round(length*0.5):], is_ethan)


def into_form(test_text):
    wordmap = analyze(test_text)
    inp = list()
    for i in combos:
        try:
            inp.append(wordmap[i])
        except KeyError:
            inp.append(0)
    return inp


def normalize(fh):
    # only allow alphabet,commas,periods,parethesis and underscores, remove multiple spaces and lowercase
    lines = re.sub('[^a-z_,.()]+', '_',
                       " ".join(" ".join(str(x) for x in fh.readlines()).split())
                       .replace(" ", "_").lower())
    return lines

if __name__ == '__main__':
    gramsFile = gzip.open('data/grams.data.gz', 'wt')
    output = csv.writer(gramsFile)
    # load from data/raw, analyze, then put into grams.data
    for file in os.listdir("data/raw/isEthan"):
        f = os.path.join("data/raw/isEthan", file)
        fh = open(f, "r")
        lines = normalize(fh)
        iterate_analyse(output, lines, True)
    for file in os.listdir("data/raw/notEthan"):
        f = os.path.join("data/raw/notEthan", file)
        fh = open(f, "r")
        lines = normalize(fh)
        iterate_analyse(output, lines, False)
