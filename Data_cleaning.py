import json
from pathlib import Path
from os import path
import bz2
import langdetect
import langdetect.lang_detect_exception
import pickle

rel_emoji = '😂😒😩😭😍😔👌😊❤😏😁🎶😳💯😴😌🙌💕😑😅🙏😕😘♥😐💁😞🙈😫😎😡👍😢😪😋😤✋😷👏👀🔫😣😈😓💔🎧🙊😉💀😖😄😜😠🙅💪👊💜💖💙😬'


class Tweetloader:
    
    def __init__(self):
        self.emoji = {e for e in rel_emoji}
        self.tweets = []
        
    def load(self, filepath, filtered = True, cleaned = True, refiltered = True):
        if path.isdir(filepath):
            for fp in Path(filepath).iterdir():
                self.load(str(fp), filtered, cleaned, refiltered)
                print(fp)
                print(len(self.tweets))
            return
        if str(filepath).endswith('.json'):    
            with open(filepath, 'r') as file:
                datalist = file.readlines()
                
        elif str(filepath).endswith('.bz2'):
            with bz2.open(filepath, 'r') as file:
                datalist = file.readlines()
        else:
            return
        data = [json.loads(entry) for entry in datalist]
        tweets = [entry["text"] for entry in data if 'text' in entry.keys()]
        if filtered:
            tweets = filter_tweets(tweets)
        if cleaned:
            tweets = clean_tweets(tweets)
        self.tweets += tweets


def has_emoji(text):
    for character in text:
        if character in rel_emoji:
            return character.encode('UTF-32')[:8].decode('UTF-32')
    return False


def has_link(text):
    return 'http' in text


def clean_emojiword(word):
    wordlist = []
    if has_emoji(word) and len(word) > 1:
        wordlist.append(word.split(has_emoji(word),1)[0])
        wordlist.append(has_emoji(word))
        wordlist.append(word.split(has_emoji(word),1)[1])
        result = ""
        for word in wordlist:
            if word != "":
                result += clean_emojiword(word) + " "
        return result
    return word


def clean_tweet(tweet):
    words = tweet.split()
    result = ""
    resultji = []
    for word in words:
        try:
            word.encode('ascii')
        except UnicodeEncodeError:
            emj = has_emoji(word)
            if emj:
                resultji.append(emj)
        else:
            if not word.startswith('RT') and not word.startswith('@'):
                result += " " + word
    return [(result, target) for target in resultji]


def clean_tweets(tweets):
    result = []
    for tweet in tweets:
        cleaned = clean_tweet(tweet)
        if type(cleaned) == list:
            result += cleaned
        else:
            result.append(cleaned)
    return result


def filter_tweets(tweets):
    tweets = [tweet for tweet in tweets if has_emoji(tweet)]
    tweets = [tweet for tweet in tweets if not has_link(tweet)]
    tweets = [tweet for tweet in tweets if detect_language(tweet) == 'en']
    return tweets


def detect_language(tweet):
    try:
        return langdetect.detect(tweet)
    except langdetect.lang_detect_exception.LangDetectException:
        return None


if __name__ == "__main__":
    tl = Tweetloader()
    tweetdir = input('Input directory?:  ')
    targetdir = input('Output filename?:  ')
    tl.load(tweetdir)
    with open(targetdir, 'wb') as file:
        pickle.dump(tl.tweets, file)

