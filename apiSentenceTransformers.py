from flask import Flask,request,jsonify
from sentence_transformers import SentenceTransformer
import scipy
import json   
from numpyencoder import NumpyEncoder


embedder = SentenceTransformer('bert-base-nli-mean-tokens')
app = Flask(__name__)

baseUrl ="/sentenceTransformers"

corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]
corpus_embeddings = embedder.encode(corpus)

queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']
query_embeddings = embedder.encode(queries)

# api's are below:

#base url testing
@app.route(baseUrl+'/')
def hello_world():
    return "welcome to sentence transformers!!!"

# this api returns all corpus sentences
@app.route(baseUrl+'/getCorpus',methods=['GET'])
def getAllCorpus():
    return jsonify(corpus)

# this api returns all given queries
@app.route(baseUrl+'/getQueries',methods=['GET'])
def getAllQueries():
    return jsonify(queries)


#this api returns embeddings of corpus sentences
@app.route(baseUrl+'/getCorpusEmbedding',methods=['GET'])
def corpusEmbedding():
    corpus_embeddings = embedder.encode(corpus)
    corpEmbedJson = json.dumps(corpus_embeddings,cls=NumpyEncoder)
    return jsonify(corpEmbedJson)

#this api gets all query embeddings
@app.route(baseUrl+'/getQueryEmbedding',methods=['GET'])
def queryEmbedding():
    query_embeddings = embedder.encode(queries)
    queryEmbedJson = json.dumps(query_embeddings,cls=NumpyEncoder)
    return jsonify(queryEmbedJson)


# this api takes number (number of sentences to be matched) and returns matched sentences for ALL QUERIES
@app.route(baseUrl+'/getForAllQueries',methods=['GET'])
def getRelatedStatements():
    try:
        closest_n = (request.args.get('number'))
        outputDict = []
        for query, query_embedding in zip(queries, query_embeddings):
            queryKey = query
            sentencesValue = []
            distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            for idx, distance in results[0:int(closest_n)]:
                # print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))
                sentencesValue.append(corpus[idx].strip())
            tempDict = { queryKey : sentencesValue }
            outputDict.append(tempDict)
        json_object = json.dumps(outputDict)   
        return json_object
    except:
        return "Error occured : Please check your input"

#this api takes index of the query required and number of sentences to be matched as input and returns matched sentences for the given query
@app.route(baseUrl+'/getForSingleQuery',methods=['GET'])
def getRelatedStatementsForQuery():
    try:
        closest_n = request.args.get('number')
        index = request.args.get('index')
        queryKey = queries[int(index)]
        sentencesValue = []
        distances = scipy.spatial.distance.cdist([query_embeddings[int(index)]], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        for idx, distance in results[0:int(closest_n)]:
                # print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))
            sentencesValue.append(corpus[idx].strip())
        tempDict = { queryKey : sentencesValue }
        json_object = json.dumps(tempDict)   
        return json_object
    
    except:
        return "Error occured : Please check your input"


if __name__ == '__main__':
   app.run()


