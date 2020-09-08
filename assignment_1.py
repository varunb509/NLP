"""
**Tokenize Words and Sentences with NLTK**
Tokenization is the process by which big quantity of text is divided into smaller parts called tokens.
These tokens are useful for finding such patterns as well as is considered as a base step for stemming and lemmatization.
**Tokenization of words**
"""

import nltk
nltk.download("popular")

from nltk.tokenize import word_tokenize
text = "Splitting words in a sentence using the word_tokenize function which is a wrapper function that calls tokenize on an instance of the TreebankWordTokenizer class."
print(word_tokenize(text))

"""**Tokenization of Sentences**"""

from nltk.tokenize import sent_tokenize
text = "Splitting sentences in the paragraph. The sent_tokenize function uses an instance of PunktSentenceTokenizer from the nltk.tokenize.punkt module, which is already been trained and thus very well knows to mark the end and beginning of sentence at what characters and punctuation."
print(sent_tokenize(text))

"""**StopWords**
> Stopwords are the words in any language which does not add much meaning to a sentence.
They can safely be ignored without sacrificing the meaning of the sentence. For some search engines, 
these are some of the most common, short function words, such as the, is, at, which, and on. In this case,
stop words can cause problems when searching for phrases that include them, particularly in names such as
“The Who” or “Take That”.
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

print(stopwords.words('english'))

# random sentecnce with lot of stop words
sample_text = "Oh man, this is pretty cool. We will do more such things."
text_tokens = word_tokenize(sample_text)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]

print(text_tokens)
print(tokens_without_sw)

"""**Stemming** 
> Stemming is basically removing the suffix from a word and reduce it to its root word.
For example: “Flying” is a word and its suffix is “ing”, if we remove “ing” from “Flying” then we will get base word or root word which is “Fly”.
"""

from nltk.stem.porter import *

porterStemmer = PorterStemmer()

sentence="A green hunting cap squeezed the top of the fleshy balloon of a head. The green earflaps, full of large ears and uncut hair and the fine bristles that grew in the ears themselves, stuck out on either side like turn signals indicating two directions at once. Full, pursed lips protruded beneath the bushy black moustache and, at their corners, sank into little folds filled with disapproval and potato chip crumbs. In the shadow under the green visor of the cap Ignatius J. Reilly’s supercilious blue and yellow eyes looked down upon the other people waiting under the clock at the D.H. Holmes department store, studying the crowd of people for signs of bad taste in dress. Several of the outfits, Ignatius noticed, were new enough and expensive enough to be properly considered offenses against taste and decency. Possession of anything new or expensive only reflected a person’s lack of theology and geometry; it could even cast doubts upon one’s soul."
wordList = nltk.word_tokenize(sentence)

stemWords = [porterStemmer.stem(word) for word in wordList]

print(' '.join(stemWords))

"""**Lemmatization**
 
Both in stemming and in lemmatization, we try to reduce a given word to its root word. The root word is called a stem in the stemming process, and it is called a lemma in the lemmatization process.
In lemmatization,the algorithms refer a dictionary to understand the meaning of the word before reducing it to its root word, or lemma.
"""

#lemmatize excluding verbs
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

sentence = "Mother died today. Or maybe yesterday, I don’t know. I had a telegram from the home: ‘Mother passed away. Funeral tomorrow. Yours sincerely.’ That doesn’t mean anything. It may have been yesterday."
punctuations="?:!.,;"
sentence_words = nltk.word_tokenize(sentence)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

sentence_words
print("{0:20}{1:20}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word)))

#lemmatize including verbs with pos_tag function
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
txt = """In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters."""
[wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in pos_tag(word_tokenize(txt))]
