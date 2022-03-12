import spacy
import operator
import xml.etree.ElementTree as XML
import tensorflow_hub as hub
from scipy import spatial

nlp = spacy.load('en_core_web_sm')
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
stopwords = set()

def tokenize(text):
    # Process the text with spacy
    doc = nlp(text)

    # Get non-stopword and alphabet tokens
    tokenizedText = []
    for token in doc:
        t = token.lemma_.lower()
        if t not in stopwords and t.isalpha():
            tokenizedText.append(t)

    # Rare cases where tweet gets filtered to nothing
    if not tokenizedText:
        tokenizedText = ["null"]

    return tokenizedText

def parseQueries(fileName):
    queries = []
    queries_raw = XML.parse(fileName).getroot()
    for q in queries_raw:
        queries.append(tokenize(q[1].text.strip()))

    #change each query from list to 1 string
    temp = []
    for query in queries:
        temp.append(" ".join(query))
    queries = temp

    return queries

def vectorize(x):
    return embed(x)



'''
Main
'''
def main():
    
    with open("StopWords.txt") as stopwords_file:
        for w in stopwords_file:
            stopwords.add(w.strip())
    
    with open("tokenized.txt") as file:
        tokenized = [next(file) for x in range(45899)]

    tweetIDs = []
    #removing tweetID at beginning and \n at end of each tokenized document
    for i in range(len(tokenized)):
        tweetIDs.append(tokenized[i][:17])
        tokenized[i] = tokenized[i][:-2]

    docDict = dict.fromkeys(tweetIDs) #key = docID, val = tokenized tweet text
    for i in range(len(tokenized)):
        docDict[tokenized[i][:17]] = tokenized[i][18:]


    #parse the queries
    queries = parseQueries("queries.txt")

    queriesResults = dict.fromkeys(queries) #key = query text, val = list of 1000 docIDs with highest simScores from highest to lowest
    results_file = open("results.txt", "r")


    for i in range(len(queries)):
        resultsArray = []
        for j in range(1000):
            line = results_file.readline().split()
            resultsArray.append(line[2])
        queriesResults[queries[i]] = resultsArray

    #vectorize the list containing all queries in text form
    queryVector = vectorize(queries)
    

    #get similarity and store in results array
    #results array is completely updated once for every outer iteration
    #one outer iteration = 1 query
    for i in range(len(queries)):
        relevantDocs = queriesResults[queries[i]]
        results = dict.fromkeys(relevantDocs) #key = docID for each doc relevant to the query, val = similarity score (computed later)
        relevantDocsText = []
        
        #for each query, we get the text of each doc and add it to relevantDocsText (size is 1000)
        for j in range(len(relevantDocs)):
            relevantDocsText.append(docDict[relevantDocs[j]])

        #vectorize the list that contains the text of all relevant docs
        docsVector = vectorize(relevantDocsText)
        
        #compute similarity and store in results dict
        for j in range(len(relevantDocs)):
            sim = 1-spatial.distance.cosine(queryVector[i], docsVector[j])
            results[relevantDocs[j]] = sim
        
        #append to resultsExp1.txt
        with open("resultsExp1.txt", "a") as results_file:
            sorted_d = dict(sorted(results.items(), key=operator.itemgetter(1), reverse=True))
            counter_rank = 0
            for key, val in sorted_d.items():
                if counter_rank >= 1000:
                    break
                counter_rank += 1
                results_file.write(f"{i+1} Q0 {key} {counter_rank} {val} FirstRun\n")
        #restart loop for next query

    
if __name__ == "__main__":
    main()