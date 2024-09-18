# Aprenentatge Bayesià: Naïve Bayes

Aquesta pràctica consisteix en crear una xarxa bayesiana per classifiar si tuits són positius o negatius, després es modifica el tamany de les dades per analitzar com afecten aquest canvis a la xarxa, i finalment aplicar Laplace-Smoothing i comprobar com afecta als resultats.

## Apartat C

En aquest apartat creo la xarxa bayesiana, comencem important les llibreries.


```python
import pandas as pd
import numpy as np
```

### Divisió en train i test

Divideixo el dataset en train i test mantenint un percentatge similiar entre valors positius i negatius


```python
def SplitData(df):
    neg = df.loc[df['sentimentLabel'] == 0]
    pos = df.loc[df['sentimentLabel'] == 1]
    
    msk_neg = np.random.rand(len(neg)) < 0.7
    msk_pos = np.random.rand(len(pos)) < 0.7
    
    train = pd.concat([neg[msk_neg], pos[msk_pos]])
    test = pd.concat([neg[~msk_neg], pos[~msk_pos]])
    
    return train, test
```

### Generació de Diccionaris

Per crear el diccionari dividim cada tuit del Train en paraules separades, afegim cada paraula com a clau d'un diccionari i incrementem el valor que hi ha en aquesta clau cada vegada que es repetiexi aquesta paraula.

Fem això tant per tuits positius com en tuits negatius i acabem amb dos diccionaris amb les repeticions de les paraules en tuits positius en un i les paraules en tuits negatius en l'altre.

També retornem el número de paraules total, ja que necessitarem saber el total de paraules en tuits positius i en negatius per fer calcular les probabilitats de les paraules.


```python
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
```

### Creació de Taula de Probabilitats

Aquesta funció rep la combinació de les paraules no repetides dels dos diccionaris, d'aquestes paraules calculem la probabilitat de que siguin positives i la probabilitat que siguin negatives. Aquesta taula és un diccionari on les claus son les paraules i el valor la probabilitat de la paraula. 

Retorno dos diccionaris un amb les paraules positives i un amb les negative, però també es podria retornar un sol diccionari amb un array amb ambdos valors, no ho faig perque la diferencia de temps no es molt significativa al meu programa i s'em fa més intuitiu amb dos diccionaris.

La probabilitat de cada paraula es calula dividint el número de vegades que la paraula es positiva o negativa entre el total de paraules positives o negatives.

En cas d'utilitzar Laplace-Smoothing s'especifica a la funció, per defecte és 0, per tant, les probabilitats de paraules que no apareguin a un diccionari serà 0. En cas d'especificar un valor per Laplace-Smoothing es sumarà aquest número a les repeticions de la paraula i es dividirà entre les paraules positives o negatives mes el total de paraules per el número especificat com a Laplace-Smoothing.


```python
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
```

### Predicció

Per fer la predicció calculo si el total de les paraules que apareixen a un tuit es positiu o negatiu i retorno un array amb 1 i 0 segons si son positius o negatius, en cas de tenir la mateixa probabilitat retornarem 2 ja que no podem predir si es positiu o negatiu. 

Si en comptes de 2 retornessi 1 o 0 augmentaria l'accuracy ja que de casualitat encertarem si son positius o negatius al només haver aquestes dues opcions, però amb diccionaris grans no passarà gaire i amb Laplace-Smoothing encara menys ja que mai passarà que ambdúes probabilitats siguin 0 al no haver una paraula al diccionari.

Per calcular la probabilitat compararo aquests valors amb els valors del tuits de test o validació i divideixo els resultats correctes entre el total.


```python
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
        
        if pos_classif == neg_classif:
            result[i] = 2
            i += 1
            continue
        
        total_pos_prob = pos_classif / (pos_classif + neg_classif)
        total_neg_prob = neg_classif / (pos_classif + neg_classif)
        
        # print(total_pos_prob, total_neg_prob)
        
        if total_pos_prob > total_neg_prob:
            result[i] = 1
        
        i += 1
        
    return result
```

### Cross-Validation

He decidit utilitzar K-Folds ja que permet utilitzar totes les dades com a validació i Leave-One-Out amb un milió de tuits trigaria molt temps.

Per fer-ho divideixo el train en K divisions i utilitzo una d'aquestes com a validació i la resta com a train, repeteixo aquest procés K vegades per utilitzar totes les divisions com a validació. Calculo l'accuracy per a cada K i retorno un array amb la probabilitat de cadascuna


```python
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
        acc_array[i] = accuracy
        
    return acc_array
```

### Resultats i Anàlisi

##### Lectura i divisió en Train i Test de les dades:


```python
df = pd.read_csv("data/FinalStemmedSentimentAnalysisDataset.csv", sep=";")

del df["tweetId"]
del df["tweetDate"]
    
df.dropna(inplace=True)
df.drop_duplicates(subset=['tweetText'], inplace=True)

train, test = SplitData(df)
```

##### Creació de la taula de probabilitats:


```python
n_neg_tweets = np.count_nonzero(train['sentimentLabel'] == 0)
n_pos_tweets = np.count_nonzero(train['sentimentLabel'] == 1)
    
neg_dict, n_neg_words = CreateDictionary(train['tweetText'].loc[train['sentimentLabel'] == 0].values)
pos_dict, n_pos_words = CreateDictionary(train['tweetText'].loc[train['sentimentLabel'] == 1].values)    

words = np.array(list(neg_dict.keys() | pos_dict.keys()))          
neg_prob = n_neg_words / (n_neg_words + n_pos_words)
pos_prob = n_pos_words / (n_pos_words + n_neg_words)
    
# L'ultim parametre es el número que s'utilitzarà per Laplace-Smoothing, si no s'especifica serà 0 per defecte
pos_t, neg_t = CreateTable(words, neg_dict, pos_dict, n_neg_words, n_pos_words, 1) 
```

##### Cross-Validation:


```python
# K=3
accuracy = KFolds(train, 3)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.69485925 0.74472678 0.74014126]
    Mitjana: 0.7265757647604278
    


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.68389342 0.73048497 0.74379055 0.74653684]
    Mitjana: 0.7261764457515044
    


```python
# K=10
accuracy = KFolds(train, 10)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.69880126 0.65938796 0.64134855 0.7146686  0.73825805 0.75569772
     0.72497384 0.74063297 0.75505296 0.75427276]
    Mitjana: 0.7183094685886784
    


```python
# K=50
accuracy = KFolds(train, 50)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.74093621 0.69508949 0.67742084 0.67163837 0.67310693 0.66847178
     0.66365305 0.65598899 0.65539238 0.65291418 0.65823772 0.66177145
     0.66030289 0.6801285  0.7211106  0.74240477 0.76388251 0.7676916
     0.77980725 0.73671409 0.71955025 0.71569527 0.75511703 0.75979807
     0.77572281 0.79775126 0.78305567 0.78126578 0.7358759  0.71283675
     0.70989949 0.71100096 0.74317316 0.74785442 0.74381569 0.74578916
     0.73633485 0.74381569 0.73289274 0.74753316 0.73881316 0.73555464
     0.76620158 0.75477327 0.76803745 0.79236277 0.71300716 0.76702772
     0.7441711  0.76500826]
    Mitjana: 0.7274079767690239
    

##### Predicció amb el test:


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.7775948905891682
    

### Resultats

Com podem observar als resultats, l'accuracy no varia molt segons la K especificada, mentre el train tingui un diccionari on apareguin moltes paraules i el percentatge de valors positius i negatius sigui similar l'accuracy no variarà molt.

Per lògica quan més gran sigui la K més grans haurien de ser els diccionaris i major hauria de ser l'accuracy però si no es per valors molt grans de K no es notarà la diferència. Si s'utilitzés Leave-One-Out l'accuracy seria major que els resultats que he obtingut amb aquests valors de K.

L'accuracy que he obtingut amb el test es mes o menys un 2.5% més gran que els valors del K-Fold, això es degut a que per crear el diccionari s'utilitza tot el train en comptes de utilitzar una divisió d'aquest, per tant al haber més paraules es més precís.

## Apartat B

### Ampliació del Train

##### Utilitzant 80% de les dades per al Train


```python
# K=3
accuracy = KFolds(train, 3)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.6947304  0.74451243 0.74073574]
    Mitjana: 0.7266595226032099
    


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.68397496 0.73081285 0.7445805  0.74749146]
    Mitjana: 0.7267149439374461
    


```python
# K=10
accuracy = KFolds(train, 10)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.69975742 0.65929829 0.6414182  0.71650495 0.73729065 0.75730557
     0.726284   0.74145742 0.75754655 0.75360258]
    Mitjana: 0.7190465634903114
    


```python
# K=50
accuracy = KFolds(train, 50)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.73333333 0.69911647 0.68028112 0.67240964 0.67289157 0.67606426
     0.6615261  0.65694779 0.65799197 0.65409639 0.65859438 0.67052209
     0.66072289 0.68196787 0.7229206  0.74416643 0.76858508 0.77095466
     0.77890678 0.74533114 0.72412547 0.71737821 0.75179726 0.75834371
     0.77015141 0.80200008 0.78493112 0.77958954 0.7350496  0.71982811
     0.71625366 0.70633359 0.74573276 0.74798185 0.7438853  0.74652583
     0.73475781 0.74363403 0.7348783  0.74564222 0.73580207 0.73672584
     0.76263154 0.76395694 0.76974054 0.79315608 0.71423408 0.76363563
     0.74194714 0.76459957]
    Mitjana: 0.7284515967723655
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.7487621125751286
    

##### Utilitzant 95% de les dades per al Train


```python
# K=3
accuracy = KFolds(train, 3)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.69507628 0.74666932 0.74279587]
    Mitjana: 0.7281804895258707
    


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.68439053 0.73173407 0.74654001 0.74949972]
    Mitjana: 0.7280410808523846
    


```python
# K=10
accuracy = KFolds(train, 10)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.69918562 0.65958786 0.64070729 0.71639204 0.73994192 0.75860598
     0.72739766 0.74394281 0.75854506 0.75507897]
    Mitjana: 0.7199385209160536
    


```python
# K=50
accuracy = KFolds(train, 50)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.73615624 0.69824668 0.68220282 0.67269158 0.67516247 0.6778703
     0.66257108 0.65999865 0.6598971  0.65363526 0.66392499 0.66893447
     0.66304495 0.68260899 0.7259342  0.74367046 0.76905632 0.77382887
     0.7825616  0.74631059 0.72390333 0.72011237 0.75534796 0.76204982
     0.77311806 0.80334416 0.78770647 0.78259545 0.73439615 0.71889385
     0.71540753 0.71103138 0.7446434  0.75229327 0.74427106 0.74900992
     0.73557188 0.74779135 0.73685814 0.74890837 0.73956606 0.73871983
     0.76393731 0.76454659 0.76870219 0.79574843 0.71342495 0.76518178
     0.74625956 0.76528333]
    Mitjana: 0.7301386314304247
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.752885075147646
    

### Resultats

Podem observar que quan major es el train major major es l'accuracy, això es deu a que al augmentar el train també augmenta el tamany del diccionari. L'augment es un 0.2% de 70% a 80% de les dades al train i de 0.4% de 80% a 95% de les dades.

## Diferents mides de diccionari amb la mateixa mida de Train

##### 1000 paraules al diccionari


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.50082843 0.50060102 0.50068359 0.50062809]
    Mitjana: 0.5006852823559435
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.49884336606179
    

##### 10000 paraules al diccionari


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.51887247 0.51405621 0.51801566 0.52718668]
    Mitjana: 0.5195327583892161
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.5174651724669717
    

##### 100000 al diccionari


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.62339797 0.62794349 0.60824695 0.61959759]
    Mitjana: 0.6197964985638694
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.5967845576517761
    

### Resultats

Podem veure clarament que quan major sigui el diccionari més accuracy obtenim, també es pot observar com la precisió mitjana obtinguda amb K-Fold es major que la obtinguda amb el test, això es degut a que aquesta vegada la predicció del test té el mateix tamany el diccionari que al K-Fold.

## Mateixa mida de diccionari amb diferent mida de Train
## La mida del diccionari sempre serà 100000 paraules
##### Utilitzant 80% de les dades al train


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.62710579 0.65578935 0.62053005 0.63822825]
    Mitjana: 0.6354133572259216
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.6235163180885352
    

##### Utilitzant 95% de les dades del train


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.62310035 0.62836102 0.60836345 0.6195525 ]
    Mitjana: 0.6198443322317237
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.6221928496265688
    

### Resultats

En aquest cas quan el Train era menor a millorat el resultat, per tant, com major sigui el Test més augmenta l'accuracy ja que els diccionaris tenien la mateixa mida.

## Apartat C
En aquest apartat seguirem el mateix procés que al B però aquesta vegada amb Laplace-Smoothing

### Ampliació del Train
##### Utilitzant 80% de les dades per al Train


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.7377056  0.7733159  0.76619613 0.76651739]
    Mitjana: 0.7609337563792613
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.777910313033947
    

##### Utilitzant 95% de les dades per al Train


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.73840242 0.77320464 0.76771463 0.76805585]
    Mitjana: 0.7618443850711966
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.7792015330565315
    

### Resultats

Podem observar que quan major es el train major major es l'accuracy, això es deu a que al augmentar el train també augmenta el tamany del diccionari. L'augment es un 0.2% de 70% a 80% de les dades al train i de 0.4% de 80% a 95% de les dades.

### Diferents mides de diccionari amb la mateixa mida de Train
##### 1000 paraules al diccionari


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.5008666  0.50064454 0.50068922 0.50066214]
    Mitjana: 0.5007156252209581
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.49846055573299264
    

##### 10000 paraules al diccionari


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.51929274 0.51414187 0.51791841 0.52718296]
    Mitjana: 0.5196339991966182
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.5166528265729798
    

##### 100000 al diccionari


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.62675961 0.63087597 0.60948337 0.62093339]
    Mitjana: 0.622013084629687
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.5965889492175024
    

### Resultats

Podem veure clarament que quan major sigui el diccionari més accuracy obtenim, també es pot observar com la precisió mitjana obtinguda amb K-Fold es major que la obtinguda amb el test, això es degut a que aquesta vegada la predicció del test té el mateix tamany el diccionari que al K-Fold.

## Mateixa mida de diccionari amb diferent mida de Train

## La mida del diccionari sempre serà 100000 paraules

##### Utilitzant 80% de les dades al train


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.631046   0.6590904  0.62049679 0.63903649]
    Mitjana: 0.6374174201337669
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.6038976111415584
    

##### Utilitzant 95% de les dades del train


```python
# K=4
accuracy = KFolds(train, 4)
print("Array:", accuracy)
print("Mitjana:", np.mean(accuracy))
```

    Array: [0.62675961 0.63087597 0.60948337 0.62093339]
    Mitjana: 0.622013084629687
    


```python
prediction = Predict(test['tweetText'].values, pos_t, neg_t, pos_prob, neg_prob, n_pos_tweets, n_neg_tweets)
correct = sum(prediction == test['sentimentLabel'].values)

print("Accuracy:", correct / prediction.size)
```

    Accuracy: 0.5965889492175024
    

### Resultats

Podem observar al cas on augmenta el train pero el diccionari no esta fixat que augmenta l'accuracy en un 4%, en canvi en la resta de casos no es nota la diferencia, això es degut a que Laplace-Smoothing ajuda quan les probabilitats de les paraules són 0, als casos on escullo la mida del diccionari utilitzo les N primeres paraules del diccionari per crear la taula de probabilitats, per tant es molt probable que les primeres paraules que apareguin siguin les més comunes i per tant s'utilitzin tant en tuits positius com en negatius, això provoca que no es noti l'efecte del Laplace-Smoothing
