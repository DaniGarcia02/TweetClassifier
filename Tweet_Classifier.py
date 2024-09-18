import pandas as pd
import numpy as np

def SplitData(df):
    neg = df.loc[df['sentimentLabel'] == 0]
    pos = df.loc[df['sentimentLabel'] == 1]
    
    msk_neg = np.random.rand(len(neg)) < 0.7
    msk_pos = np.random.rand(len(pos)) < 0.7
    
    train = pd.concat([neg[msk_neg], pos[msk_pos]])
    test = pd.concat([neg[~msk_neg], pos[~msk_pos]])
    
    return train, test


def CreateDictionary(tweets):
    total_words = 0
    dictionary = {}
    i = 0
    for tweet in tweets:
        words = tweet.split()
        total_words += len(words)
        for word in words:
            if word in dictionary.keys():
                dictionary[word] += 1
            else:
                dictionary[word] = 1
            i+=1

    return dictionary, total_words


# ls stands for laplace smoothing, if not specified it's not applied
def CreateTable(words, neg_dict, pos_dict, n_neg_words, n_pos_words, ls = 0):
    pos_table = {}
    neg_table = {}
    
    for word in words:
        if word in pos_dict.keys():
            pos_prob = (pos_dict[word] + ls) / (n_pos_words + ls * words.size)
        else:
            pos_prob = (0 + ls) / (n_pos_words + ls * words.size)
            
        if word in neg_dict.keys():
            neg_prob = (neg_dict[word] + ls) / (n_neg_words + ls * words.size)
        else:
            neg_prob = (0 + ls) / (n_neg_words + ls * words.size)
            
        pos_table[word] = pos_prob
        neg_table[word] = neg_prob
    
    return pos_table, neg_table


def Predict(tweets, pos_t, neg_t, neg_prob, pos_prob, n_pos_tweets, n_neg_tweets):
    result = np.zeros(tweets.size)
    i=0
    for tweet in tweets:
        pos_classif = pos_prob
        neg_classif = neg_prob
        words = tweet.split()
        
        for word in words:
            if word in pos_t.keys():
                pos_classif *= pos_t[word]
                
            if word in neg_t.keys():
                neg_classif *= neg_t[word]
        
        pos_classif = (pos_classif * n_pos_tweets) / (n_pos_tweets + n_neg_tweets)
        neg_classif = (neg_classif * n_neg_tweets) / (n_pos_tweets + n_neg_tweets)
        
        total_pos_prob = 0
        total_neg_prob = 0
        
        if pos_classif == 0 and neg_classif == 0:
            i += 1
            continue
        
        total_pos_prob = pos_classif / (pos_classif + neg_classif)
        total_neg_prob = neg_classif / (pos_classif + neg_classif)
        
        # print(total_pos_prob, total_neg_prob)
        
        if total_pos_prob > total_neg_prob:
            result[i] = 1
        
        i += 1
        
    return result


def KFolds(df, k):
    acc_array = np.zeros(k)
    
    neg = df.loc[df['sentimentLabel'] == 0]
    pos = df.loc[df['sentimentLabel'] == 1]
        
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    folds = []

    for df_i, df_j in zip(pos_folds, neg_folds):
        folds.append(pd.concat([df_i, df_j]))
        
    for i, df_i in enumerate(folds):
        validation = df_i
        emptyTrain = True
        for j, df_j in enumerate(folds):
            if i == j:
                continue            
            if emptyTrain:
                train = df_j
                emptyTrain = False
            else:
                train = pd.concat([train, df_j])
        
        n_neg_tweets = np.count_nonzero(train['sentimentLabel'] == 0)
        n_pos_tweets = np.count_nonzero(train['sentimentLabel'] == 1)
        
        neg_dict, n_neg_words = CreateDictionary(train['tweetText'].loc[train['sentimentLabel'] == 0].values)
        pos_dict, n_pos_words = CreateDictionary(train['tweetText'].loc[train['sentimentLabel'] == 1].values)    
    
        words = np.array(list(neg_dict.keys() | pos_dict.keys()))          
        neg_prob = n_neg_words / (n_neg_words + n_pos_words)
        pos_prob = n_pos_words / (n_pos_words + n_neg_words)
        
        pos_t, neg_t = CreateTable(words, neg_dict, pos_dict, n_neg_words, n_pos_words, 1)
        
        prediction = Predict(validation['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
        correct = sum(prediction == validation['sentimentLabel'].values)
        accuracy = correct / prediction.size
        print("K"+ str(i), accuracy)
        acc_array[i] = accuracy
        
    return np.mean(acc_array)


def main():
    df = pd.read_csv("data/FinalStemmedSentimentAnalysisDataset.csv", sep=";")
    
    del df["tweetId"]
    del df["tweetDate"]
    
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['tweetText'], inplace=True)
    
    train, test = SplitData(df)
    print("Creating Dictionary...\n")
    n_neg_tweets = np.count_nonzero(train['sentimentLabel'] == 0)
    n_pos_tweets = np.count_nonzero(train['sentimentLabel'] == 1)
    
    neg_dict, n_neg_words = CreateDictionary(train['tweetText'].loc[train['sentimentLabel'] == 0].values)
    pos_dict, n_pos_words = CreateDictionary(train['tweetText'].loc[train['sentimentLabel'] == 1].values)    

    words = np.array(list(neg_dict.keys() | pos_dict.keys()))          
    neg_prob = n_neg_words / (n_neg_words + n_pos_words)
    pos_prob = n_pos_words / (n_pos_words + n_neg_words)
    
    pos_t, neg_t = CreateTable(words, neg_dict, pos_dict, n_neg_words, n_pos_words, 1)  
    
    print("--- K-Fold Cross Validation ---")
    accuracy = KFolds(train, 4)
    print("Mean Accuracy:", accuracy * 100, '%\n')  
    
    print("--- Making Prediction for Test ---")
    prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
    
    correct = sum(prediction == test['sentimentLabel'].values)
    print("Accuracy:", (correct / prediction.size) *100, '%' )    
    
    
if __name__ == "__main__":
    main()