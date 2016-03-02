import sys
import nltk
import math
import time

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the words of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    
    for line in brown_train:
        tokens = line.split()
        
        word_line = []
        tag_line = []

        # Adding the start symbol before sentence
        word_line.append(START_SYMBOL)
        word_line.append(START_SYMBOL)
        tag_line.append(START_SYMBOL)
        tag_line.append(START_SYMBOL)

        # Adding the words and tags in the line
        for item in tokens:
            word_token = item.rsplit('/', 1)
            word_line.append(word_token[0])
            tag_line.append(word_token[1])

        # Adding the stop symbol after sentence
        word_line.append(STOP_SYMBOL)
        tag_line.append(STOP_SYMBOL)
        
        brown_words.append(word_line)
        brown_tags.append(tag_line)
    
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    unigram_count = {}
    bigram_count = {}
    trigram_count = {}

    for tag_list in brown_tags:
        unigram_tuples = [(word,) for word in tag_list]
        bigram_tuples = list(nltk.bigrams(tag_list))
        trigram_tuples = list(nltk.trigrams(tag_list))

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

    for word in trigram_count:
        q_values[word] = math.log(float(trigram_count[word])/bigram_count[(word[0], word[1])], 2)
   
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    word_count = {}

    for sentence in brown_words:
        for word in sentence:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    for word in word_count:
        if word_count[word] > RARE_WORD_MAX_FREQ:
            known_words.add(word)

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []

    for sentence in brown_words:
        sentence_with_rare = []
        for word in sentence:
            if word in known_words:
                sentence_with_rare.append(word)
            else:
                sentence_with_rare.append(RARE_SYMBOL)
        brown_words_rare.append(sentence_with_rare)

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    e_count = {}
    tag_count = {}

    for tag_list in brown_tags:
        for tag in tag_list:
            if tag in taglist:
                tag_count[tag] += 1
            else:
                tag_count[tag] = 1
                taglist.add(tag)

    for word_list, tag_list in zip(brown_words_rare, brown_tags):
        for word, tag in zip(word_list, tag_list):
            if (word, tag) in e_count:
                e_count[(word, tag)] += 1
            else:
                e_count[(word, tag)] = 1

    for (word, tag) in e_count:
        e_values[(word, tag)] = math.log(float(e_count[(word, tag)])/tag_count[tag], 2)

    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    taglist.remove(START_SYMBOL)
    taglist.remove(STOP_SYMBOL)
    taglist = list(taglist)
    brown_dev_words_rare = replace_rare(brown_dev_words, known_words)
    
    for ind in xrange(0, len(brown_dev_words_rare)):
	sentence = brown_dev_words_rare[ind]
        tag_states = []
        state_prob = []
        for tag in taglist:
            if (sentence[0], tag) in e_values:
                emission_prob = e_values[(sentence[0], tag)]
            else:
                tag_states.append((1, -1))
                continue
            if (START_SYMBOL, START_SYMBOL, tag) in q_values:
                trans_prob = q_values[(START_SYMBOL, START_SYMBOL, tag)]
            else:
                trans_prob = LOG_PROB_OF_ZERO
            tag_states.append((emission_prob + trans_prob, START_SYMBOL))
        state_prob.append(tag_states)

        for i, word in enumerate(sentence):
            if i == 0:
                continue
            word_states = []
            for tag in taglist:
                max_trans_prob = -10000
                max_history_tag = -1
                if (word, tag) in e_values:
                    emission_prob = e_values[(word, tag)]
                else:
                    word_states.append((1, -1))
                    continue
                for k, prev_tag in enumerate(taglist):
                    prev_prob = state_prob[i - 1][k][0]
                    prev_prev_tag_index = state_prob[i - 1][k][1]

                    if prev_prob == 1:
                        continue
                    if prev_prev_tag_index == START_SYMBOL:
                        prev_prev_tag = START_SYMBOL
                    else:
                        prev_prev_tag = taglist[prev_prev_tag_index]
                    if (prev_prev_tag, prev_tag, tag) in q_values:
                        trans_prob = prev_prob + q_values[(prev_prev_tag, prev_tag, tag)]
                    else:
                        trans_prob = prev_prob + LOG_PROB_OF_ZERO
                    if trans_prob > max_trans_prob:
                        max_trans_prob = trans_prob
                        max_history_tag = k
                word_states.append((emission_prob + max_trans_prob, max_history_tag))
            state_prob.append(word_states)

        tag = STOP_SYMBOL
        max_trans_prob = -10000
        max_history_tag = -1
        word_states = []
        for i, prev_tag in enumerate(taglist):
            prev_prob = state_prob[len(sentence) - 1][i][0]
            prev_prev_tag_index = state_prob[len(sentence) - 1][i][1]

            if prev_prob == 1:
                continue
            prev_prev_tag = taglist[prev_prev_tag_index]
            if (prev_prev_tag, prev_tag, tag) in q_values:
                trans_prob = prev_prob + q_values[(prev_prev_tag, prev_tag, tag)]
            else:
                trans_prob = prev_prob + LOG_PROB_OF_ZERO
            if trans_prob > max_trans_prob:
                max_trans_prob = trans_prob
                max_history_tag = i
        word_states.append((max_trans_prob, max_history_tag))
        state_prob.append(word_states)

        tags = []
        for index in xrange(len(sentence)):
            tags.insert(0, max_history_tag)
            max_history_tag = state_prob[len(sentence) - index - 1][max_history_tag][1]

        tagged_sentence = ''
        for index in xrange(0, len(sentence)):
            if index == len(sentence) - 1:
                tagged_sentence = tagged_sentence + brown_dev_words[ind][index] + '/' + taglist[tags[index]]
            else:
                tagged_sentence = tagged_sentence + brown_dev_words[ind][index] + '/' + taglist[tags[index]] + ' '
        tagged_sentence += '\n'
        tagged.append(tagged_sentence)

    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    tagged_sentence = []

    for sentence in brown_dev_words:
        sentence.insert(0, START_SYMBOL)
        sentence.insert(0, START_SYMBOL)
        sentence.append(STOP_SYMBOL)
        tagged_sentence = trigram_tagger.tag(sentence)
        sentence_string = ''
        for word in xrange(2, len(sentence)-1):
            if word == len(sentence) - 2:
                sentence_string += tagged_sentence[word][0] + '/' + tagged_sentence[word][1]
            else:
                sentence_string += tagged_sentence[word][0] + '/' + tagged_sentence[word][1] + ' ' 

        sentence_string += '\n'
        tagged.append(sentence_string)
 
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
