#!/usr/bin/env python
# coding: utf-8

# # Platforma ray

# In[1]:


import ray
import time
# ray.init(dashboard_host='0.0.0.0')


# In[2]:


# ray.shutdown()


# Tablica *Dashboard* jest dostępna pod adresem http://localhost:8265

# ## 1. Zadania (Tasks)
# Cytując: *"Ray enables arbitrary functions to be executed asynchronously on separate Python workers. Such functions are called Ray remote functions and their asynchronous invocations are called Ray tasks."*

# In[3]:


@ray.remote
def suma(a,b):
    return a + b


# In[4]:


# Ta funkcja nie zostanie wywołana
# suma(1,2)


# In[5]:


# Sprawdż dashboard
suma.remote(1,2)


# In[6]:


result_ref = suma.remote(1,2)
# Pobierz wynik
ray.get(result_ref)


# ### 1.1 Składanie wywołań 

# In[7]:


result_ref = suma.remote(suma.remote(1,2),suma.remote(3,4))
ray.get(result_ref)


# Zadania ```suma.remote(1,2)``` i ```suma.remote(3,4)``` mogą być wykonywane współbieżnie na różnych węzłach obliczeniowych. Zadanie zwracające ostateczny wynik będzie oczekiwało na zakończenie poprzednich.
# 
# Ray analizuje zależności między funkcjami i tworzy graf zależności, który jest używany do efektywnego i równoległego wykonania zadań. 

# In[8]:


r1 = suma.remote(1,2)
r2 = suma.remote(3,4)
result_ref = suma.remote(r1,r2)
ray.get(result_ref)


# ### 1.2 Jeśli funkcja nie zwraca wartości

# In[9]:


@ray.remote
def void_foo(*params):
    for k in params:
        print(k,end=' ')
    print()

void_foo.remote(1,'ala',3.5,True)


# **TODO 1.2.1** Znajdź wydruk Dashboard:Jobs > void_foo > stdout

# ### 1.3 Ciąg Fibonacciego -  rekurencja
# Napisz funkcję obliczającą rekurencyjnie n-tą wartość ciągu Fibbonaciego.
# Sprawdź na tablicy Dashboard ile razy była wywołana funckja dla róznych wartości parametrów (ale raczej n <= 10)

# In[10]:


@ray.remote
def fibo(n):
    print(n)
    if n <= 1:
        return n
    else:
        return ray.get(fibo.remote(n-1)) + ray.get(fibo.remote(n-2))

obj_ref = fibo.remote(10)
ray.get(obj_ref)


# ### 1.4 Ciąg Fibonacciego -  iteracyjnie

# In[11]:


@ray.remote
def next_fibo_number(a,b):
    return a + b

obj_ref = next_fibo_number.remote(1,2)
ray.get(obj_ref)


# **TODO 1.4.1** Wykorzystaj funcję ```next_fibo_number``` do wzynaczenia elementu ciągu Fibboncciego o mnumerze 102 

# In[12]:


@ray.remote
def fibo_iterative(n):
    if n <= 2:
        return 1
        
    a = 1
    b = 1
    
    for _ in range(n-2):
        obj_ref = next_fibo_number.remote(a, b)
        next_num = ray.get(obj_ref)
        a = b
        b = next_num
    
    return b

result = ray.get(fibo_iterative.remote(102))
result



# Dlaczego w tej wersji działa znacznie szybciej? Jak (zapewne) wyglądają drzewa zależnoci w tych dwóch przypadkach?
# 
# 

# ### 1.5 Obliczanie pi

# **TODO 1.5.1** Napisz funkcję zgodnie ze specyfikacją

# In[13]:


import random
num_slices=100
slice_size=1000_000

@ray.remote
def compute_pi_on_slice(slice_size):
    """
    Params: slice_size - liczba powtórzeń
    Funkcja losuje dwie liczby x i y i sparwdza, czy mieszczą się w ćwiartce koła. Jesli tak, inkremetuje sumę.
    Returns: 4 * suma/ slice_size
    """
    sum_val = 0
    for _ in range(slice_size):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            sum_val += 1
    return 4 * sum_val / slice_size

object_ref = compute_pi_on_slice.remote(slice_size=slice_size)
ray.get(object_ref)


# In[14]:


# Dashboard
tab_ref = [compute_pi_on_slice.remote(slice_size=slice_size) for i in range(num_slices)]


# In[15]:


# Pobiera wszystkie elementy
# ray.get(tab_ref)


# In[16]:


tab_ref = [compute_pi_on_slice.remote(slice_size=slice_size) for i in range(num_slices)]


# Funkcja ```ray.wait()``` pobiera informacje o gotowych i oczekujących zadaniach na liście

# In[17]:


ready,remaining = ray.wait(tab_ref,num_returns=len(tab_ref),timeout=1)
print('-------------------------- ready ------------------------')
print(ready)
print('------------------------ remaining ----------------------')
print(remaining)


# ## 2. Aktorzy
# 
# Cytując: *"Actors extend the Ray API from functions (tasks) to classes. An actor is essentially a stateful worker (or a service). When a new actor is instantiated, a new worker is created, and methods of the actor are scheduled on that specific worker and can access and mutate the state of that worker."*

# ### 2.1 Aktor łączący teksty

# In[18]:


@ray.remote
class Concatenator:
    
    def __init__(self,initial_text=''):
        self.text = initial_text
    
    def append(self,text):
        print(f'[append called] with param: {text}')
        self.text += '\n'
        self.text += text

    def get(self):
        print(f'[get called]')
        return self.text

# wywołanie konstruktora
conc_ref = Concatenator.remote('Lokomotywa') 

# wywołanie zdalnych metod obiektu
conc_ref.append.remote('Stoi na stacji lokomotywa')
conc_ref.append.remote('Ciężka, ogromna i pot z niej spływa -')
conc_ref.append.remote('Tłusta oliwa.') 

text_ref = conc_ref.get.remote()
print(ray.get(text_ref))

# opcjonalnie można usunąć
# del conc_ref


# ### 2.2 Aktor zliczający słowa

# Counter przechowuje informacje o liczbie wystąpień obiektów

# In[19]:


from collections import Counter
counter = Counter()
counter.update([1,1,2,1,1,3,2,'a','a','b'])
counter.most_common(3)


# Wykorzystamy go do policznia wystąpień słów w tekscie

# In[20]:


url = 'https://wolnelektury.pl/media/book/txt/w-pustyni-i-w-puszczy.txt'
import requests
text = requests.get(url).text
words = text.split()
words[:30]


# Poprawimy wydzielanie symboli

# In[21]:


from bs4 import BeautifulSoup
import re
import requests

def tokenize(text): 
    text = BeautifulSoup(text, features="html.parser").get_text(' ')
    text = re.sub(r"[#\",!?;-<>/\\*\\&-]", " ", text) #znaki
    text = re.sub('\[[^\]]*\]',' ',text) #
    text = re.sub(r"[:\.\+\=()–©°′″•↑—]", " ", text) #reszta znaków
    text = re.sub(r"\d+\.?\d*", " ", text) #liczby
    words = text.split()
    return words



from collections import Counter
word_freq = Counter()

url = 'https://wolnelektury.pl/media/book/txt/w-pustyni-i-w-puszczy.txt'

text = requests.get(url).text

words = tokenize(text)
word_freq.update(words)
top_words=word_freq.most_common(10)
top_words


# In[22]:


word_freq.most_common()[-20:-1]


# Kolejne uaktualnienie

# In[23]:


text = requests.get('https://pl.wikipedia.org/wiki/Polska').text

words = tokenize(text)
word_freq.update(words)
top_words=word_freq.most_common(10)
top_words


# **TODO 2.2.1** Napisz klasę (aktora) ```WordCountingActor``` z metodami:
# * konstruktorem - inicjalizuje atrybut typu ```Counter```
# * tokenize() - wykorzystaj gotową funkcję
# * add_text(self,text:str) - dzieli tekst na słowa i uaktualnia licznik
# * get_top_words(self,n: int) - zwraca n najczęsciej występujących słów

# In[24]:


from collections import Counter
from bs4 import BeautifulSoup
import re

@ray.remote
class WordCountingActor:
    def __init__(self):
        self.counter = Counter()
        

    def tokenize(self, text:str): 
       return tokenize(text)

    def add_text(self, text:str):
       words = self.tokenize(text)
       self.counter.update(words)

    def get_top_words(self,n: int):
        return self.counter.most_common(n)


# **TODO 2.2.2** Utwórz aktora i dodawaj teksty pobrane ze stron internetowych. Możesz rozszerzyć przykładową listę. Po dodaniu teksty wypisz 10 najczęściej pojawiająych się słów.

# In[25]:


urls = ['https://pl.wikipedia.org/wiki/Akademia_G%C3%B3rniczo-Hutnicza_im._Stanis%C5%82awa_Staszica_w_Krakowie',
        'https://pl.wikipedia.org/wiki/Polska',
        'https://pl.wikipedia.org/wiki/Krak%C3%B3w',
        'https://www.agh.edu.pl/']


# In[26]:


wca_ref = WordCountingActor.remote()
for url in urls:
    text = requests.get(url).text
    wca_ref.add_text.remote(text)

top_words = wca_ref.get_top_words.remote(10)


# In[27]:


ray.get(top_words)


# ## 3. Obiekty (object)
# 
# Cytując: *"In Ray, tasks and actors create and compute on objects. We refer to these objects as remote objects because they can be stored anywhere in a Ray cluster, and we use object refs to refer to them. Remote objects are cached in Ray’s distributed shared-memory object store, and there is one object store per node in the cluster. In the cluster setting, a remote object can live on one or many nodes, independent of who holds the object ref(s).*
# 
# *An object ref is essentially a pointer or a unique ID that can be used to refer to a remote object without seeing its value. If you’re familiar with futures, Ray object refs are conceptually similar."*
# 
# Załadujemy obraz i utworzymy reprezentujący go obiekt

# In[28]:


from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (5, 5)

response = requests.get('https://images.pexels.com/photos/32277444/pexels-photo-32277444/free-photo-of-black-cat-on-geometric-steps-in-istanbul.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2')
img = Image.open(BytesIO(response.content))
img = np.array(img)

plt.imshow(img, cmap='gray')
plt.show()


# In[29]:


img_ref = ray.put(img)


# In[30]:


img_copy = ray.get(img_ref)
plt.imshow(img_copy,cmap='gray')
plt.show()


# ## 4. Biblioteka  Ray Data i Dataset
# Cytując: *"Ray Data is a scalable data processing library for ML workloads. It provides flexible and performant APIs for scaling Offline batch inference and Data preprocessing and ingest for ML training. Ray Data uses streaming execution to efficiently process large datasets.*
# 
# Podstawową klasą jest ```Dataset```. Podobue, jak w przypadku Sparka możliwy jest:
# * odczyt danych z róznych źródeł
# * zapis w róznych formatach
# * podstawowe transformacje (w tym konwersja do Pandas)
# * podział na wsady (batch)
# * automatyczne partycjonowanie i rozkładanie pomiędzy węzły
# * iteracja po pojedynczych danych lub z podziałem na wsady w formatach zgodnych z Tensorflow i Torch
# 

# In[31]:


ds_train = ray.data.read_csv("twitter_training.csv")
ds_test = ray.data.read_csv("twitter_validation.csv")


# In[32]:


ds_train


# In[33]:


ds_train.show(5)


# In[34]:


ds_train.stats()


# ### 4.1 Podstawowe transformacje

# #### to_pandas

# In[35]:


df_train = ds_train.to_pandas()
df_test = ds_test.to_pandas()
df_train.head(10)


# #### select_columns

# In[36]:


ds2 = ds_train.select_columns(['Entity','Sentiment']).show(5)


# #### groupby

# In[37]:


ds2 = ds_train.select_columns(['Entity','Sentiment']).groupby(['Entity','Sentiment']).count()


# In[38]:


ds2.to_pandas()


# #### filter

# In[39]:


ds2 = ds_train.filter(lambda row:row['Entity']=='Nvidia')


# In[40]:


ds2.show(5)


# #### map
# Funkcja konwertuje pojedyncze wiersze

# In[41]:


def split_content(row):
    row['words'] = row['TweetContent'].split()
    return row
    
ds_train.map(split_content).show(10)


# **TODO 4.1.1** 
# * dodaj kolumnę ```word_count``` zawierającą liczbę słów w ```TweetContent```
# * zgrupuj po kolumnach ```Entity``` i ```Sentiment```
# * za pomocą ```sum(col_name)``` policz ile bylo słów we wpisach należących do danej grupy
# * wyświetl jako ```pandas.DataFrame```

# In[42]:


def count_words(row):
    row["word_count"] = len(row["TweetContent"].split())
    return row
    
ds_train = ds_train.map(count_words)
ds_train.groupby(["Entity", "Sentiment"]).sum("word_count").to_pandas()


# #### flat_map
# Funkcja konweruje jeden wiersz na wiele wierszy (analogia do explode)

# In[43]:


def explode_content(row):
    words = row['TweetContent'].split()
    return [{'word':w} for w in words]

ds_train.flat_map(explode_content).show(10)


# **TODO 4.1.2** Przepisz funckję ```explode_content()```
# 
# W pętli po słowach:
# * utwórz kopię ```row```
# * dodaj słowo (w kolumnie ```word```)
# * dodaj zmodyfikowaną kopię do listy
# 
# Zwróć listę

# In[44]:


def explode_content(row):
    rows = []
    
    text_to_split = row.get('content', '') 
    
    words_list = text_to_split.split()

    for word_item in words_list:
        copied_row = row.copy() 
        copied_row['word'] = word_item 
        rows.append(copied_row)
    
    return rows


# #### random_shuffle

# In[45]:


ds_train.random_shuffle(seed=1).to_pandas()


# #### iteratory (w tym dzielące na wsady: batch)

# In[46]:


for i,row in enumerate(ds_train.iter_rows()):
    if i==3:
        break
    print(row)


# In[47]:


for i,row in enumerate(ds_train.iter_batches(batch_size=2)):
    if i==3:
        break
    print('-------------')    
    print(row)


# ## 5. Klasyfikacja

# ### 5.1 Metryki

# **TODO 5.1.1** napisz funkcję, która oblicza i zwraca metryki lub raport klasyfikacji

# In[48]:


from sklearn.metrics import precision_score, recall_score, classification_report, f1_score, accuracy_score

@ray.remote
def get_classification_results(y_true,y_pred,labels=None,return_scores=True):
    """
    Funkcja oblicza wetryki klasyfikacji
    Jeżeli return_scores==True funkcja zwraca słownik zawierający wartości accuracy, precision, recall (w wersji macro)
    Jeżeli return_scores==False funcka zwraca tekst będący wynikiem wywołania classification_report (z etykietami klas)
    """
    if return_scores:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    else:
        return classification_report(y_true, y_pred, labels=labels, zero_division=0)


# In[49]:


results_ref = get_classification_results.remote([1,1,2,2,3,3],[1,1,2,1,3,3],labels=['a','b','c'])
ray.get(results_ref)


# In[50]:


results_ref = get_classification_results.remote([1,1,2,2,3,3],[1,1,2,1,3,3],labels=['a','b','c'],return_scores=False)
print(ray.get(results_ref))


# ### 5.2 train_and_test

# **TODO 5.2.1** Napisz funckję zgodnie ze specyfikacją

# In[51]:


from sklearn import preprocessing

@ray.remote
def train_and_test(df_train, df_test, model, features_column, target_column, return_scores=True):
    """
    Funkcja (1) stosuje preprocessing.LabelEncoder() dla zbudowania wektorów liczbowych etykiet y_train oraz y_test
    (2) przeprowadza uczenie modelu
    (3) wyznacza wartości przewidywanych etykiet dla zbioru treningowego i testowego
    (4) wyznacza metryki predykcji dla zbioru uczącego i testowego wywołując get_classification_results
    
    Params: df_train - zbiór treningowy
            df_test  - zbiór testowy
            model - model klasyfikatora (algorytm) 
            features_column - kolumna lub lista kolumn zawierająca cechy
            target_column - kolumna z etykietami
            return_scores - analogicznie, jak get_classification_results
    Returns:
            Słownik zawierający następujące elementy
            * 'train' - metryki dla zbioru treningowego
            * 'test' - metryki dla zbioru testowego (powinno być "testowego", nie "uczącego" w opisie zwrotki)
            * 'model' - wytrenowany model
            * 'labels' - unikalne etykiety (klasy) po enkodowaniu
    """
    
    le = preprocessing.LabelEncoder()
    
    y_train_encoded = le.fit_transform(df_train[target_column])

    try:
        y_test_encoded = le.transform(df_test[target_column])
    except ValueError as e:
        
        print(f"Error encoding test labels: {e}. New labels found in test set not present in training set.")
        
        return {
            'train': {},
            'test': {},
            'model': model,
            'labels': []
        }

    X_train = df_train[features_column]
    X_test = df_test[features_column]
    
    model.fit(X_train, y_train_encoded)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
   
    encoded_labels = list(range(len(le.classes_)))

    results_train = get_classification_results.remote(y_train_encoded, y_pred_train, labels=encoded_labels, return_scores=return_scores)
    results_test = get_classification_results.remote(y_test_encoded, y_pred_test, labels=encoded_labels, return_scores=return_scores)
    
    return {
        'train': results_train,
        'test': results_test,
        'model': model,
        'labels': le.classes_ 
    }



# ### 5.3 Wywołanie

# In[52]:


ds_train = ray.data.read_csv("twitter_training.csv")
ds_test = ray.data.read_csv("twitter_validation.csv")
df_train = ds_train.to_pandas()
df_test = ds_test.to_pandas()
df_train_ref = ray.put(df_train)
df_test_ref = ray.put(df_test)


# **TODO 5.3.1** Wywołaj funkcję przekazując referencje do odpowiednich obiektów Ray

# In[53]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline([('vect', CountVectorizer()), ('cls', MultinomialNB())])
results_ref = train_and_test.remote(df_test, df_train, pipeline, features_column='TweetContent', target_column='Sentiment', return_scores=True)


# In[54]:


results = ray.get(results_ref)


# In[55]:


ray.get(results['train'])


# In[56]:


ray.get(results['test'])


# **TODO 5.3.2** Wywołaj funkcję dla innego klasyfikatora i odczytaj wyniki

# In[57]:


from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
pipeline = Pipeline([('vect', CountVectorizer()), ('cls', SGDClassifier(loss='perceptron', penalty='l2', alpha=1e-3))])
results_ref = train_and_test.remote(df_test, df_train, pipeline, features_column='TweetContent', target_column='Sentiment', return_scores=True)


# In[58]:


results = ray.get(results_ref)


# In[59]:


#train
ray.get(results['train'])


# In[60]:


# test
ray.get(results['test'])


# ## 6. Walidacja krzyżowa

# **TODO 6.1.1** Napisz funkcję zgodnie ze specyfikacją. 
# 
# Będzie realizowała pojedyncze zadanie wywoływane podczas walidacji krzyżowej (uczenie na k-1 podzbiorach, testowanie na jednym pozostawionym podzbiorze). 
# * Te zadania można zrównoleglić.
# * Zbiór, na którym będzie wykonywana walidacja krzyżowa będzie obiektem
# * Zadania będą wywoływane z róznymi zestawami indeksów odpowiadającymi podziałowi na podzbiory (*ang. fold*) uzyte do budowy modelu i testowania

# In[61]:


@ray.remote
def train_and_test_fold(df,train_indexes,test_indexes, model,features_column, target_column,return_scores=True):
    """
    Funkcja
    (1) dzieli zbiór na podzbiór dt_train treningowy i df_test testowy na podstawie indeksów wierszy 
    (2) stosuje preprocessing.LabelEncoder() dla zbudowania wektorów liczbowych etykiet y_train oraz y_test
    (3) przeprowadza uczenie modelu
    (4) wyznacza wartości przewidywanych etykiet dla zbioru treningowego i testowego
    (5) wyznacza metryki predykcji dla zbioru uczącego i testowego wywołując get_classification_results
    
    Params: df - zbiór poddany walidacji krzyżowej 
            train_indexes - indeksy wierszy podzbioru treningowego (z 
            test_indexes  - indeksy wierszy podzbioru testowego
            model - model klasyfikatora (algorytm) 
            features_column - kolumna zawierająca cechy
            target_column - kolumna z etykietami
            return_scores - analogicznie, jak get_classification_results
    Returns:
            Słownik zawierający następujące elementy
            * 'train' - metryki dla zbioru treningowego
            * 'test' - metryki dla zbioru uczącego
            * 'model' - wytrenowany model
            * 'labels' - etykiety
    """
    
    


# Przykład funkcji realizującej walidację krzyzową ze stratyfikacją (podziałem zachowującym prawdopodobieństwo a-priori klas)

# In[62]:


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

@ray.remote
def cross_validate_stratified(df,model, features_column, target_column ,n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits)
    result_refs=[]
    for i, (train_indexes, test_indexes) in enumerate(skf.split(df_train, df_train[target_column])):
        result_refs.append(train_and_test_fold.remote(df_train,train_indexes, test_indexes,clone(model),features_column, target_column))
    return result_refs


# **TODO 6.1.2** napisz analogiczną funkcję bez startyfikacji, posługująca się klasa ```KFold```

# In[63]:


from sklearn.model_selection import KFold
from sklearn.base import clone




# In[64]:


pipeline = Pipeline([('vect', CountVectorizer()), ('cls', SGDClassifier(loss='perceptron', penalty='l2', alpha=1e-3))])
results_ref = cross_validate_stratified.remote(df_train_ref, 
                                    pipeline,
                                    features_column='TweetContent', target_column='Sentiment')


# **TODO 6.1.3** Odczytaj wyniki

# In[66]:





# In[ ]:





# **TODO 6.1.4** Wywołaj funkcję ```cross_validate()``` i odczytaj wyniki

# In[65]:


pipeline = Pipeline([('vect', CountVectorizer()), ('cls', SGDClassifier(loss='perceptron', penalty='l2', alpha=1e-3))])
results_ref = cross_validate.remote(...)


# In[ ]:





# In[73]:





# ## 7. Modele dla Entity 

# Spróbujemy zbudować indywidaualne klasyfikatory dla ocenianych obiektów *Entity*

# In[74]:


df_train.Entity.unique()


# **TODO 7.1.1** W petli po podzbiorach danych:
# * utwórz Pipeline
# * wywołaj funkcję ```train_and_test()```
# * umieść wyniki (classification_report) w słowniku konwerując nazwę obiektu na małe litery
# * wyświetl wyniki dla testów

# In[75]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

results={}    
for ent in df_train.Entity.unique():
    ...


# In[76]:


for k in results:
    print(f'-------------- {k} ------------')
    ...


# **TODO 7.1.2** Doaj jeszcze do słownika wyniki dla pełnego zbioru (pod kluczem ```*```)

# In[77]:


pipeline = Pipeline([('vect', CountVectorizer()), ('cls', SGDClassifier(loss='perceptron', penalty='l2', alpha=1e-3))])
results['*'] = train_and_test.remote(df_train,df_test,pipeline,features_column='TweetContent', target_column='Sentiment', return_scores=False)


# **TODO 7.1.3** Wyekstrahuj z rezuultatów dwa słowniki: ```models```  z modelami oraz ```labels``` z etykietami.
# 
# Etykiety występują zawsze w tej samej kolejności, ale to specyfika zbioru danych. Gdyby zastosować permutację, przed uczeniem, kolejność byłaby inna, więc *better save than sorry...*

# In[78]:


models = ???
models


# In[79]:


labels = ???
labels


# Poniższa funkcja dokonuje klasyfikacji tekstu dla danego typu obiektu (entity). Jesli nie zostanie znaleziony, stosuje klucz ```*```
# 
# Wypróbuj jej działanie wpisując rózne wartości...

# In[80]:


def classify(models,labels,ent,text):
    k = ent.lower()
    if not k in models:
        k = '*'
    y_pred =  models[k].predict([text])
    return labels[k][y_pred[0]]
    
classify(models,labels,'*','low energy card')


# In[81]:


classify(models,labels,'nvidia','low energy card')


# ## 8. Ray serve

# Ray Serve to framework do zarządzania mikrousługami, który został stworzony na platformie Ray. Pozwala on na łatwe tworzenie, wdrażanie i skalowanie aplikacji opartych na mikrousługach. 
# 
# Cechy:
# 
# * **Prostota użycia:** Ray Serve zapewnia proste API do definiowania i wdrażania mikrousług. Programiści mogą definiować obsługę żądań za pomocą zwykłych funkcji Pythona.
# * **Elastyczność i skalowalność:** Ray Serve automatycznie zarządza skalowaniem i równoważeniem obciążenia mikrousług, co umożliwia obsługę wysokiego obciążenia aplikacji.
# * **Obsługa wielu typów modeli:** Ray Serve obsługuje wiele rodzajów modeli, w tym modele oparte na frameworkach uczenia maszynowego, takich jak TensorFlow, PyTorch czy Scikit-learn.
# * **Obsługa wielu typów interfejsów:** Obsługuje różne interfejsy API, takie jak REST, gRPC, HTTP oraz in-memory Python API, co umożliwia łatwe integrowanie z różnymi typami aplikacji.
# * **Rozproszenie i równoległość:** Dzięki wykorzystaniu platformy Ray, Ray Serve oferuje wbudowane wsparcie dla równoległego przetwarzania i rozproszenia, co pozwala na obsługę wysokich obciążeń i dużych ilości danych.

# In[82]:


from ray import serve
# serve.shutdown()
serve.start(http_options={'host':'0.0.0.0','port':8000})


# ### 8.1 Przykład wdrożenia mikrousługi

# Pokazemy prosty przykład usług REST. Dzięki (opcjonalnej) integracji z FastApi możliwe będzie wywołanie poprzez interfejs *Swagger*

# In[ ]:


import requests
from fastapi import FastAPI
from ray import serve
# import uvicorn

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@serve.deployment
@serve.ingress(app)
class FastAPIDeployment:
    # FastAPI will automatically parse the HTTP request for us.
    @app.get("/hello")
    def say_hello(self, firstname: str, surname: str) -> str:
        return f"Hello {firstname} {surname}!"


# 2: Deploy the deployment.
# serve.shutdown()
# serve.start(http_options={'host':'0.0.0.0','port':8000})
serve.run(FastAPIDeployment.bind(), route_prefix="/")

# otwórz http://localhost:8000/docs


# In[ ]:


print(requests.get("http://0.0.0.0:8000/hello", params={"firstname": "Jan",'surname':'Kowalski'}).text)


# Otwórz stronę http://localhost:8000/docs#/default/say_hello_hello_get i wypróbuj serwis
# 
# ![image.png](attachment:1b5ffc35-24bc-4e4e-ae8a-6a3e8c56a6d6.png)

# Otwórz stronę http://localhost:8000/docs#/default/say_hello_hello_get i wypróbuj serwis

# ### 8.2 Wdrożenie zbudowanego zbioru modeli jako mikroserwis

# **TODO 8.2.1** zaimplementuj klasę ```ClassifierDeployment```
# * W konstruktorze przekaż słowniki z modelami i etykietami
# * Zaimplementuj metodę ```calssify`` zgodnie ze specyfikacją

# In[ ]:


import requests
from fastapi import FastAPI
from ray import serve
# import uvicorn

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@serve.deployment
@serve.ingress(app)
class ClassifierDeployment:
    
    def __init__(self,models,labels):
       
        
    # FastAPI will automatically parse the HTTP request for us.
    @app.get("/classify")
    def classify(self, entity: str, text: str) -> str:
        """
        Params: entity:str nazwa obiektu
                text: treść wypowiedzi do sklasyfikowania
        Returns: 
                tekstową etykietę (Irrevelant, Positive, Neutral, Negative)
        
        """
        


# 2: Deploy the deployment.

app = ClassifierDeployment.bind(????)
serve.run(app, route_prefix="/")


# Wyświetlimy przykładowe wypowiedzi

# In[ ]:


ds_train.filter(lambda row:row['Entity']=='Nvidia').random_shuffle(seed=1).to_pandas().head(20)


# **TODO 8.2.2** Zamieśc kilka przykładowych wywołań mikroserwisu

# In[87]:


print(requests.get("http://0.0.0.0:8000/classify", params={"entity": "nvidia",'text':'so like, is there even a single game where'}).json())


# **TODO 8.2.3** Powtórz wywołanie w intefejsie Swaggera. Możesz wkleić bezpośrednio zrzuty ekranu
# 
# ![image.png](attachment:c2c5ee0b-2fad-4862-8524-ab82ca1c9653.png)
# 
# ![image.png](attachment:7a74064a-fc70-4038-89ec-5cb1220d726a.png)
# 
