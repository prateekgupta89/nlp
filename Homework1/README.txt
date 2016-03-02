PART A:

1) UNIGRAM near -12.4560686964
   BIGRAM near the -1.56187888761
   TRIGRAM near the ecliptic -5.39231742278

2) Perplexity for unigram model: 1052.4865859
   Perplexity for bigram model: 53.8984761198
   Perplexity for trigram model: 5.7106793082

3) Perplexity with linear interploation model: 12.5516094886

4) The linear interpolation model is better than the unigram and bigram model but
   not better than the trigram model. The three models are created using a corpus
   and the trigram model is the most accurate we can generate since it has more
   context. When we use the linear interpolation model, we use all the three models,
   assigning weights to these models which increases the overall perplexity of the
   linear interpolation model as compared to the trigram model. Hence, the results
   we got are as expected.

5) python perplexity.py output/Sample1_scored.txt data/Sample1.txt 11.1670289158
   python perplexity.py output/Sample2_scored.txt data/Sample2.txt 1611240282.44

   Based on the above perplexities, it can be argued that Sample 1 is similar to
   brown corpus. A low perplexity indicates the probability distribution is good 
   at predicting the sample. Here, since the model is trained on a dataset, it
   can predict a similar sample with a low perplexity. The fact that we get a low
   perplexity indicates that the model is able to do that. If we compare the two
   samples, we can see that sample 2 has no words in common with the original
   trainign set and has sentences in a different language which explains the
   reason for missing unigrams.
   

PART B:

2) TRIGRAM CONJ ADV NOUN -4.46650366731
   TRIGRAM DET NUM NOUN -0.713200128516
   TRIGRAM NOUN PRT CONJ -6.38503274104

4) * * 0.0
   midnight NOUN -13.1814628813
   Place VERB -15.4538814891
   primary ADJ -10.0668014957
   STOP STOP 0.0
   _RARE_ VERB -3.17732085089
   _RARE_ X -0.546359661497

5) Percent correct tags: 93.2413469619

6) Percent correct tags: 91.3944534563

Running Times
---------------

Running time for Part A: 15.7 sec
Running time for Part B: 18.54 sec 
