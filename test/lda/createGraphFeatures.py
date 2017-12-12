import networkx as nx
import string
import numpy as np
import math

# corresponds to the number of edges that a vertex is connected to, regardless of either the direction or the weight
def degree_centrality(G):
    
    centrality={}
    s=1.0
    centrality=dict((n,d*s) for n,d in G.degree_iter())
    return centrality

# only consider one direction in the co-occurrence
def in_degree_centrality(G):    
    if not G.is_directed():
        raise nx.NetworkXError("in_degree_centrality() not defined for undirected graphs.")
    centrality={}
    s=1.0
    centrality=dict((n,d*s) for n,d in G.in_degree_iter())
    return centrality

def out_degree_centrality(G):
    if not G.is_directed():
             raise nx.NetworkXError("out_degree_centrality() not defined for undirected graphs.")
    centrality={}
    s=1.0
    centrality=dict((n,d*s) for n,d in G.out_degree_iter())
    return centrality

# considers all contexts of co-occurrence   
def weighted_centrality(G):
    centrality={}    
    s=1.0
    centrality=dict((n,d*s) for n,d in G.degree_iter(weight='weight'))
    return centrality

def createGraphFeatures(num_documents,clean_train_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par):
    features = np.zeros((num_documents,len(unique_words)))
    term_num_docs = {}

    print("Creating the graph of words for collection...")

    if centrality_col_par=="pagerank_centrality" or centrality_col_par=="out_degree_centrality" or centrality_col_par=="in_degree_centrality" or centrality_col_par=="betweenness_centrality_directed" or centrality_col_par=="closeness_centrality_directed":
        dGcol = nx.DiGraph()
    else:
        dGcol = nx.Graph()

    totalLen = 0
    for i in range(0,num_documents):
        #dG = nx.Graph()
        found_unique_words = []
        wordList1 = clean_train_documents[i].split(None)
        wordList2 = [x.lower().rstrip(',.!?;') for x in wordList1]

        docLen = len(wordList2)
        totalLen += docLen

        # print clean_train_documents[i]

        for k, word in enumerate(wordList2):

            if word not in found_unique_words:
                found_unique_words.append(word)
                if word not in term_num_docs:
                    term_num_docs[word] = 1
                else:
                    term_num_docs[word] += 1

            for j in range(1,sliding_window):
                try:
                    next_word = wordList2[k + j]
                
                    if not dGcol.has_node(word):
                        dGcol.add_node(word)
                        dGcol.node[word]['count'] = 1
                        
                    else:
                        dGcol.node[word]['count'] += 1
                        
                    if not dGcol.has_node(next_word):
                        dGcol.add_node(next_word)
                        dGcol.node[next_word]['count'] = 0

                    if not dGcol.has_edge(word, next_word):
                        dGcol.add_edge(word, next_word, weight = 1)

                    else:
                        dGcol[word][next_word]['weight'] += 1
                except IndexError:
                    if not dGcol.has_node(word):
                        dGcol.add_node(word)
                        dGcol.node[word]['count'] = 1
                    else:
                        dGcol.node[word]['count'] += 1
                except:
                    raise

    avgLen = float(totalLen)/num_documents
    print("Number of nodes in collection graph:"+str(dGcol.number_of_nodes()))
    print("Number of edges in collection graph:"+str(dGcol.number_of_edges()))

    print("Average document length:"+str(avgLen))
    print("Number of self-loops for collection graph:"+str(dGcol.number_of_selfloops()))

    if idf_par=="icw":
        icw_col = {}
        dGcol.remove_edges_from(dGcol.selfloop_edges())

        nx.write_edgelist(dGcol, "test.edgelist")

        centrality_col = nx.degree_centrality(dGcol)

        centr_sum = sum(centrality_col.values())
        for k, g in enumerate(dGcol.nodes()):
            if centrality_col[g]>0:
                icw_col[g] = math.log10((float(centr_sum)) / (centrality_col[g]))
            else:
                icw_col[g] = 0

    idf_col = {}
    for x in term_num_docs:
        idf_col[x] = math.log10((float(num_documents)+1.0) / (term_num_docs[x]))

    print("Creating the graph of words for each document...")
    totalNodes = 0
    totalEdges = 0
    for i in range( 0,num_documents ):

        if centrality_par=="pagerank_centrality" or centrality_par=="out_degree_centrality" or centrality_par=="in_degree_centrality" or centrality_par=="betweenness_centrality_directed" or centrality_par=="closeness_centrality_directed":
            dG = nx.DiGraph()
        else:
            dG = nx.Graph()

        wordList1 = clean_train_documents[i].split(None)
        wordList2 = [x.lower().rstrip(',.!?;') for x in wordList1]
        docLen = len(wordList2)
        
        if docLen==2 :
            print(wordList2)
        if docLen>1 and wordList2[0]!=wordList2[1] :
            # print clean_train_documents[i]
            for k, word in enumerate(wordList2):
                for j in range(1,sliding_window):
                    try:
                        next_word = wordList2[k + j]
                        if not dG.has_node(word):
                            dG.add_node(word)
                            dG.node[word]['count'] = 1
                        else:
                            dG.node[word]['count'] += 1
                        if not dG.has_node(next_word):
                            dG.add_node(next_word)
                            dG.node[next_word]['count'] = 0
    
                        if not dG.has_edge(word, next_word):
                            dG.add_edge(word, next_word, weight = 1)

                        else:
                            dG[word][next_word]['weight'] += 1
                    except IndexError:
                        if not dG.has_node(word):
                            dG.add_node(word)
                            dG.node[word]['count'] = 1
                        else:
                            dG.node[word]['count'] += 1
                    except:
                        raise
    
            dG.remove_edges_from(dG.selfloop_edges())
            
            centrality=degree_centrality(dG)
    
            totalNodes += dG.number_of_nodes()
            totalEdges += dG.number_of_edges()
        
            for k, g in enumerate(dG.nodes()):
                # Degree centrality (local feature)
                if g in unique_words:
                    if idf_par=="no":
                        features[i,unique_words.index(g)] = centrality[g]#centrality[g]/(1-b+(b*(float(docLen)/avgLen)))dG.node[g]['count']
                    elif idf_par=="idf":
                        features[i,unique_words.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * idf_col[g]
                    elif idf_par=="icw":
                        features[i,unique_words.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g]

    print("Average number of nodes:"+str(float(totalNodes)/num_documents))
    print("Average number of edges:"+str(float(totalEdges)/num_documents))
    return features
