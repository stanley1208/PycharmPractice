tag2id,id2tag={},{}
word2id,id2word={},{}

for line in open('traintext.txt'):
    items=line.split('/')
    word,tag=items[0],items[1].strip()

    if word not in word2id:
        word2id[word]=len(word2id)
        id2word[len(id2word)]=word
    if tag not in tag2id:
        tag2id[tag]=len(tag2id)
        id2tag[len(id2tag)]=tag

M=len(word2id)  # M: 辭典的大小
N=len(tag2id)   # N: 詞性的種類個數

print(M,N)
print(word2id)
print(id2word)
print(tag2id)
print(id2tag)