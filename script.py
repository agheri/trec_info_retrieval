import spacy
import math
import operator
import xml.etree.ElementTree as XML

nlp = spacy.load('en_core_web_sm')

'''
Step 1. Preprocessing
input: document (a tweet) and list of stopwords as input, 
output: tokenized document in the form of a list, with tweet ID at index 0
'''
def tokenize(tweet, stopwords):
    
    doc = nlp(tweet)
    tokenizedTweet = [] # To store each individual tokenized tweet
    for token in doc:
        tokenizedTweet.append(token.lemma_.lower())
    tweetID = tokenizedTweet[0]
    # Clean up tokenized doc by removing stopwords, non-alphabet, and links
    tokenizedTweet = [word for word in tokenizedTweet if not word in stopwords]
    tokenizedTweet = [word for word in tokenizedTweet if word.isalpha()]
    tokenizedTweet.insert(0, tweetID) # Insert tweetID to beginning of the tokenized tweet

    return tokenizedTweet


'''
Step 2. Indexing
input: tokens obtained from step1 in the form of a list of lists
output: inverted index (dictionary)
'''
def buildIndex(tokens):
    invIndex = dict()

    docCount=0
    for tweet in tokens:
        for word in tweet:
            if word == tweet[0]:
                continue

            frequency = tweet.count(word)

            if word not in invIndex.keys():
                invIndex[word] = [[tweet[0], frequency]]
            else:
                li = [tweet[0], frequency]
                if (li not in invIndex[word]):
                    invIndex[word].append(li)
            
        docCount+=1
    return invIndex

''' Step 3. Retrieval and Ranking'''
def retrieve(invertedIndex,query,stopwords):
    totalNumTweets = 45899
    tf_idf = invertedIndex.copy()
    for i in tf_idf:
        for j in tf_idf.get(i):
            j.append(math.log(totalNumTweets/len(tf_idf.get(i)),2))
            j.append(j[1]*j[2])
    #preprocess the query
    cleaned_query = nlp(query)
    tokenized_query= []
    for token in cleaned_query:
        tokenized_query.append(token.lemma_.lower())
    tokenized_query = [word for word in tokenized_query if not word in stopwords]
    tokenized_query = [word for word in tokenized_query if word.isalpha()]
    #query tf-idf vector
    queryIndex = dict()
    for i in  tokenized_query:
        tf = tokenized_query.count(i)
        if(tf_idf.get(i)!= None): #just making sure nothing is null
            queryIndex[i] = len(tokenized_query)/tf*(tf_idf.get(i)[0][2]) #so here getting the 3rd attribute which would be idf
        else:
            print("Word not in dict")

    #Finding all the docid and tf-idf to calculate the doc lengths
    length_doc = dict()
    for i in tf_idf:
        for j in tf_idf.get(i):
            if(length_doc.get(j[0])!=None):
                length_doc[j[0]].append(j[3])
            else:
                 length_doc[j[0]]= [j[3]]
    
    # #Calculating doc lengths
    calc_length_doc = dict()
    
    for i in length_doc:
        tempVal=0
        for j in length_doc.get(i):
            tempVal += j**2
        calc_length_doc[i] = math.sqrt(tempVal)
    #calculating query lengths
    queryLength =0
    for key,value in queryIndex.items():
        queryLength += value**2
    queryLength = math.sqrt(queryLength)
    #Calculating CosSim
    storing_results = dict() #storing the partial results and then
    for i in tokenized_query:
        if(tf_idf.get(i)!= None):
            for j in tf_idf.get(i):
                if(storing_results.get(j[0])==None):
                    storing_results[j[0]] = j[3]*queryIndex[i]
                else:
                    storing_results[j[0]] += j[3]*queryIndex[i]
        else:
            print("word not in query")
    
    for key,value in storing_results.items(): #/ the whole value set by doc length*query length
        storing_results[key] = value/(calc_length_doc[key]*queryLength)
    return storing_results

def getResults(invertedIndex,query,stopwords,tag): #providing the inverted index, the query, the stopwords and the tag(dont need stopwords if we preprocess the query using another method)
    f = open("results.txt","a")
    f.write("topic_id Q0 docno rank score tag \n")
    
    counter_rank = 0
    if (counter_rank < 1000):
        for i in range(len(query)): #assuming query is a array of queries to pass
            stored_results = retrieve(invertedIndex,query[i],stopwords)
            sorted_d = dict( sorted(stored_results.items(), key=operator.itemgetter(1),reverse=True))
            
            for key,value in sorted_d.items():
                counter_rank +=1
                f.write(str(i)+" "+"Q0"+" "+key+" "+str(counter_rank)+" "+str(value)+" "+tag + "\n")
            
def getQueries():
    queries = XML.parse("queries.txt").getroot()

    out = []
    for q in queries:
        out.append(q[1].text.strip())
    return out

    


def main():
    # Will store all tokenized tweets
    tokenizedTweets = []

    # Open file and store stopwords in list
    with open('StopWords.txt') as f:
        content = f.readlines()
    stopwords = [x.strip() for x in content]
    
    # Open tweets file and tokenize
    with open('tweets.txt', encoding='utf8') as tweets:
        tweets.read(1)
        # There are 45899 total tweets
        for _ in range(45899):
            data = tweets.readline()
            tokenizedTweets.append(tokenize(data, stopwords))

    #pass tokenizedTweets into buildIndex function in order to build inverted index
    invIndex = buildIndex(tokenizedTweets)
    query = getQueries()
    
    #pass the values and begin
    getResults(invIndex,query,stopwords,"FirstRun")
    print(len(invIndex))


if __name__ == "__main__":
    main()  