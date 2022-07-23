from nltk.stem import WordNetLemmatizer
import re
import nltk

def convert(x):
    if x == 'traffic':
        return 1
    else:
        return 0

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