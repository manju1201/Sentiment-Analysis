from nltk import FreqDist,classify,NaiveBayesClassifier
from nltk.corpus import twitter_samples,stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string, random

#this function is to remove noise like links,punctuation and special characters 
#and twitter handles in replies, they are replaced with a empty string
#This function does both normalization(lemmatization) and removing the noise

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    
#pos_tag function gives tags to each token which assesses relative position of a word in text
    for token,tag in pos_tag(tweet_tokens):   
        token = re.sub('http[s]?://(?:[a-zA-Z]|[$-_@.&+#]|[!*\(\),]|'\
        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','',token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)  #.sub() method replaces the selected string with empty string
          

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        #lemmatization is form of normalization where the tokens are changed to their root words
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token,pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())

    return cleaned_tokens

#print('After Lemmatization,removing noises and stop words:',remove_noise(tweet_tokens[0],stop_words))

#To determine the word density
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

#The format of cleaned data is list, it is now changed to python dictionary with words as keys and True as values
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


positive_tweets = twitter_samples.strings('positive_tweets.json') #strings() method is to print the tweets as stings
nagative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')

stop_words = stopwords.words('english')

#.tokenzied() method will split the strings into smaller parts as tokens
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
#print('tokenized tweets:',tweet_tokens[0]) 

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens,stop_words))
for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens,stop_words))

#print(positive_tweet_tokens[500])
#print(positive_cleaned_tokens_list[500])


all_pos_words = get_all_words(positive_cleaned_tokens_list)

#.most_common() method lists the words which occur most frequently.
freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)


#spliting the dataset to train and test
positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset
random.shuffle(dataset)
train_data = dataset[:7000]
test_data = dataset[7000:]

#Training the data using naivebayesclassifer
classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(10))