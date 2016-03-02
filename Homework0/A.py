import nltk
import sys

def input(variable):
    print variable

greeting = sys.stdin.read()
input(greeting)

token_list = nltk.word_tokenize(greeting)
squirrel = 0
girl = 0
print "The tokens in the greeting are"
for token in token_list:
    print token
    if token.lower() == 'squirrel':
        squirrel = squirrel + 1
    if token.lower() == 'girl':
        girl = girl + 1
print "There were %d instances of the word 'squirrel' and %d instances of the word 'girl.'" % (squirrel, girl)
