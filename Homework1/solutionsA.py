import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    unigram_count = {}
    bigram_count = {}
    trigram_count = {}
    word_count = 0.0

    for line in training_corpus:
        tokens = line.split()
        tokens.insert(0, START_SYMBOL)
        word_count += len(tokens)
        tokens.append(STOP_SYMBOL)

        trigram_tokens = list(tokens)
        trigram_tokens.insert(0, START_SYMBOL)
        unigram_tuples = [(word,) for word in tokens]
        bigram_tuples = list(nltk.bigrams(trigram_tokens))
        trigram_tuples = list(nltk.trigrams(trigram_tokens))

        for word in unigram_tuples:
            if word in unigram_count:
                unigram_count[word] += 1
            else:
                unigram_count[word] = 1

        for word in bigram_tuples:
            if word in bigram_count:
                bigram_count[word] += 1
            else:
                bigram_count[word] = 1

        for word in trigram_tuples:
            if word in trigram_count:
                trigram_count[word] += 1
            else:
                trigram_count[word] = 1

    for word in unigram_count:
        unigram_p[word] = math.log(unigram_count[word]/word_count, 2)

    for word in bigram_count:
        bigram_p[word] = math.log(float(bigram_count[word])/unigram_count[(word[0],)], 2)

    for word in trigram_count:
        trigram_p[word] = math.log(float(trigram_count[word])/bigram_count[(word[0], word[1])], 2)

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []

    for line in corpus:
        score = 0.0
        unigram_tokens = line.split()
        unigram_tokens.append(STOP_SYMBOL)

        bigram_tokens = list(unigram_tokens)
        bigram_tokens.insert(0, START_SYMBOL)
        bigram_tuples = list(nltk.bigrams(bigram_tokens))

        trigram_tokens = list(bigram_tokens)
        trigram_tokens.insert(0, START_SYMBOL)
        trigram_tuples = list(nltk.trigrams(trigram_tokens))

        if n == 1:
            for word in unigram_tokens:
                if (word,) in ngram_p:
                    score += ngram_p[(word,)]
                else:
                    score = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break

        elif n == 2:
            for word in bigram_tuples:
                if word in ngram_p:
                    score += ngram_p[word]
                else:
                    score = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break
 
        elif n == 3:
            for word in trigram_tuples:
                if word in ngram_p:
                    score += ngram_p[word]
                else:
                    score = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break

        scores.append(score)

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []

    for line in corpus:
        score = 0.0
        unigram_tokens = line.split()
        unigram_tokens.append(STOP_SYMBOL)

        bigram_tokens = list(unigram_tokens)
        bigram_tokens.insert(0, START_SYMBOL)
        bigram_tuples = list(nltk.bigrams(bigram_tokens))

        trigram_tokens = list(bigram_tokens)
        trigram_tokens.insert(0, START_SYMBOL)
        trigram_tuples = list(nltk.trigrams(trigram_tokens))

        for word in range(0, len(bigram_tuples)):
            if (unigram_tokens[word],) in unigrams:
               unigram_log_prob = unigrams[(unigram_tokens[word],)]
               if bigram_tuples[word] in bigrams:
                   bigram_log_prob = bigrams[bigram_tuples[word]]
                   if trigram_tuples[word] in trigrams:
                       trigram_log_prob = trigrams[trigram_tuples[word]]
                       score += math.log((1.0/3.0) * ((math.pow(2,unigram_log_prob)) + (math.pow(2,bigram_log_prob)) + (math.pow(2,trigram_log_prob))), 2)
                   else:
                       score += math.log((1.0/3.0) * ((pow(2,unigram_log_prob)) + (pow(2,bigram_log_prob))), 2)
               else:
                   score += math.log((1.0/3.0) * (pow(2,unigram_log_prob)), 2)
            else:
                score = MINUS_INFINITY_SENTENCE_LOG_PROB
                break


        scores.append(score)
        
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
