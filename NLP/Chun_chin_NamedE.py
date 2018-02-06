import nltk
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import state_union

# chunking ###
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def chunking():
    try:
        for i in tokenized:
            words = word_tokenize(i)
            tagged = nltk.pos_tag(words)

            ChunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

            ChunkParser = nltk.RegexpParser(ChunkGram)
            chunked = ChunkParser.parse(tagged)

            chunked.draw()
            print(tagged)

    except Exception as e:
        print(e)

# chunking()

# chinking ###
def chinking():
    try:
        for i in tokenized:
            words = word_tokenize(i)
            tagged = nltk.pos_tag(words)

            ChunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""

            ChunkParser = nltk.RegexpParser(ChunkGram)
            chunked = ChunkParser.parse(tagged)

            chunked.draw()
            print(tagged)

    except Exception as e:
        print(e)

# chinking()

# NamedEntities ###
def process():
    try:
        for i in tokenized:
            words = word_tokenize(i)
            tagged = nltk.pos_tag(words)

            namedEnt = nltk.ne_chunk(tagged)

            #namedEnt = nltk.ne_chunk(tagged, binary=True) # will show NE(if chunk is named)

            namedEnt.draw()

    except Exception as e:
        print(e)

# process()
