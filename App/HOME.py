from pyparsing import col
import streamlit as st

st.set_page_config(
    page_title="Twitter Traffic Analysis",
    page_icon="🚦",
    layout="wide"
)



st.title("TRAFFIC ANALYSIS FROM TWITTER STREAMS")


st.markdown("<hr>", True)

col1, col2 = st.columns([3,1])

col1.markdown("""
Twitter is one of the most widely used Social Media Platform for users to share their toughts and ideas. However, Twitter can also be interpreted as a huge
storehouse of data, which on proper processing can serve as a huge source of information. This type of Data is known as **Big Data**.
<br><br>
This application streams tweets from **Twitter API** and uses **Machine Learning** to filter and classify the tweets in order to have an idea about the situation of traffic
in popular cities.
<br><br>
In order to increase its efficency, various models have been tried and tested and the best possible one can be seen from the scores and results.
<br><br>
The user can use **"Popular Search"** to quickly obtain traffic related tweets about some common cities based on pre-defined model.
They can also have complete control over the model-building in the **"Custom Search"** section.
""", True)

col2.markdown("<br>", True)
col2.image("twitter_home.png")

st.markdown("<hr>", True)


st.sidebar.markdown("""
### Documentation Links:
- <a href="https://docs.tweepy.org/en/stable/" style="color:white">TweePy</a>
- <a href="https://devdocs.io/scikit_learn/" style="color:white">Scikit-Learn</a>
- <a href="https://pandas.pydata.org/docs/" style="color:white">Pandas</a>
- <a href="https://seaborn.pydata.org/" style="color:white">Seaborn</a>
- <a href="https://www.nltk.org/" style="color:white">NLTK</a>
- <a href="https://docs.streamlit.io/library" style="color:white">StreamLit</a>
<hr/>

""", True)

st.sidebar.markdown("### Follow / Contact me for collaboration:",True)

side_col1, side_col2, side_col3, side_col4, side_col5 = st.sidebar.columns([1,1,1,1,1])
side_col1.markdown("""
<a href="https://www.linkedin.com/in/ayushmaan-das-635ab621a">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" style="height:30px"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path d="M416 32H31.9C14.3 32 0 46.5 0 64.3v383.4C0 465.5 14.3 480 31.9 480H416c17.6 0 32-14.5 32-32.3V64.3c0-17.8-14.4-32.3-32-32.3zM135.4 416H69V202.2h66.5V416zm-33.2-243c-21.3 0-38.5-17.3-38.5-38.5S80.9 96 102.2 96c21.2 0 38.5 17.3 38.5 38.5 0 21.3-17.2 38.5-38.5 38.5zm282.1 243h-66.4V312c0-24.8-.5-56.7-34.5-56.7-34.6 0-39.9 27-39.9 54.9V416h-66.4V202.2h63.7v29.2h.9c8.9-16.8 30.6-34.5 62.9-34.5 67.2 0 79.7 44.3 79.7 101.9V416z"/></svg>
</a>
""",True)
side_col2.markdown("""
<a href="https://www.github.com/ayushmaanFCB">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512" style="height:30px"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/></svg>
</a>
""",True)
side_col3.markdown("""
<a href="https://twitter.com/AyushMaan_10">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" style="height:30px"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path d="M459.37 151.716c.325 4.548.325 9.097.325 13.645 0 138.72-105.583 298.558-298.558 298.558-59.452 0-114.68-17.219-161.137-47.106 8.447.974 16.568 1.299 25.34 1.299 49.055 0 94.213-16.568 130.274-44.832-46.132-.975-84.792-31.188-98.112-72.772 6.498.974 12.995 1.624 19.818 1.624 9.421 0 18.843-1.3 27.614-3.573-48.081-9.747-84.143-51.98-84.143-102.985v-1.299c13.969 7.797 30.214 12.67 47.431 13.319-28.264-18.843-46.781-51.005-46.781-87.391 0-19.492 5.197-37.36 14.294-52.954 51.655 63.675 129.3 105.258 216.365 109.807-1.624-7.797-2.599-15.918-2.599-24.04 0-57.828 46.782-104.934 104.934-104.934 30.213 0 57.502 12.67 76.67 33.137 23.715-4.548 46.456-13.32 66.599-25.34-7.798 24.366-24.366 44.833-46.132 57.827 21.117-2.273 41.584-8.122 60.426-16.243-14.292 20.791-32.161 39.308-52.628 54.253z"/></svg>
</a>
""",True)
side_col4.markdown("""
<a href="https://www.facebook.com/AYmaan.AsesinoDeRey.10">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 512" style="height:30px"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path d="M279.14 288l14.22-92.66h-88.91v-60.13c0-25.35 12.42-50.06 52.24-50.06h40.42V6.26S260.43 0 225.36 0c-73.22 0-121.08 44.38-121.08 124.72v70.62H22.89V288h81.39v224h100.17V288z"/></svg>
</a>
""",True)
side_col5.markdown("""
<a href="https://www.instagram.com/ayushmaan.fcb/">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" style="height:30px"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path d="M224.1 141c-63.6 0-114.9 51.3-114.9 114.9s51.3 114.9 114.9 114.9S339 319.5 339 255.9 287.7 141 224.1 141zm0 189.6c-41.1 0-74.7-33.5-74.7-74.7s33.5-74.7 74.7-74.7 74.7 33.5 74.7 74.7-33.6 74.7-74.7 74.7zm146.4-194.3c0 14.9-12 26.8-26.8 26.8-14.9 0-26.8-12-26.8-26.8s12-26.8 26.8-26.8 26.8 12 26.8 26.8zm76.1 27.2c-1.7-35.9-9.9-67.7-36.2-93.9-26.2-26.2-58-34.4-93.9-36.2-37-2.1-147.9-2.1-184.9 0-35.8 1.7-67.6 9.9-93.9 36.1s-34.4 58-36.2 93.9c-2.1 37-2.1 147.9 0 184.9 1.7 35.9 9.9 67.7 36.2 93.9s58 34.4 93.9 36.2c37 2.1 147.9 2.1 184.9 0 35.9-1.7 67.7-9.9 93.9-36.2 26.2-26.2 34.4-58 36.2-93.9 2.1-37 2.1-147.8 0-184.8zM398.8 388c-7.8 19.6-22.9 34.7-42.6 42.6-29.5 11.7-99.5 9-132.1 9s-102.7 2.6-132.1-9c-19.6-7.8-34.7-22.9-42.6-42.6-11.7-29.5-9-99.5-9-132.1s-2.6-102.7 9-132.1c7.8-19.6 22.9-34.7 42.6-42.6 29.5-11.7 99.5-9 132.1-9s102.7-2.6 132.1 9c19.6 7.8 34.7 22.9 42.6 42.6 11.7 29.5 9 99.5 9 132.1s2.7 102.7-9 132.1z"/></svg>
</a>
""",True)

col_git1, col_git2, col_git3 = st.columns([7 ,1, 1])
col_git1.markdown("""
<h4 style="font-style:italic"> Visit the <u>GitHub</u> repository or <u>Google Drive </u> for source code and references: </h4>
<br><br>
""", True)
col_git2.markdown("""
<a href="https://github.com/B-TECH-2021-25-INT200/E-Traffic-Alert-System">
<img src="https://img.icons8.com/clouds/100/FFFFFF/github.png" style="height:60px; "/>
</a><br><br>
""", True)
col_git3.markdown("""
<a href="#">
<img src="https://img.icons8.com/clouds/100/FFFFFF/google-drive.png" style="height:60px;"/>
</a><br><br>
""", True)


expander1 = st.expander("Sample Python Code for Tweets Fetching")
expander1.code("""
import tweepy, configparser

consumer_key = #
consumer_secret = #
access_token = #
access_secret = #

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

query = ""
tweets = tweepy.Cursor(api.search_tweets, q=query1, lang='en', tweet_mode='extended', result_type='recent').items(500)
""")


expander2 = st.expander("Sample Python Code for Tweet Preprocessing")
expander2.code("""
from nltk.stem import WordNetLemmatizer
import re, nltk

def remove_emojis(data):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def preprocess_tweets(tweet):
    tweet = re.sub(r'https?://[^ ]+', '', tweet)
    tweet = re.sub(r'@[^ ]+', '', tweet)
    #tweet = re.sub(r'0 ', 'zero', tweet)
    tweet = re.sub(r'[^A-Za-z ]', '', tweet)
    tweet = tweet.lower()
    tweet = remove_emojis(tweet)
    #tweet = remove_stopwords(tweet)

    return tweet

def lemmatization(x):
    punctuations = '.?:!,;'
    words = nltk.word_tokenize(x)
    new_text=""
    for word in words:
        if word in punctuations:
            words.remove(word)
    word_net_lemmatizer = WordNetLemmatizer()
    for word in words:
        lem_word = word_net_lemmatizer.lemmatize(word,"v")
        new_text = new_text + lem_word+" "
    new_text.strip()
    return new_text
""")


expander3 = st.expander("Sample Python Code for Model Building")
expander3.code("""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# TRAIN-TEST-SPLIT:
x_train, x_test, y_train, y_test = train_test_split(samp_df.processed_text, samp_df.Classification, test_size=0.3, random_state=101)

# VECTORIZATION:
vectorizer = CountVectorizer()
x_train  = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# TRAINING:
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
""", language='python')


