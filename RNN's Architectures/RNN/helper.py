import re 
import string

def remove_html_tags(text):
    pattern=re.compile('<.*?>')
    return pattern.sub(' ',text)

def remove_urls(text):
    pattern=re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(' ',text)
text = "Visit https://google.com or www.youtube.com for more information."

def remove_punctuations(text):
    exlude=string.punctuation
    for ch in exlude:
        text=text.replace(ch," ")
    return text
def remove_punctuation2(text):
    return text.translate(str.maketrans(' ',' ',string.punctuation))

chat_words = {
    "u": "you",
    "ur": "your",
    "urs": "yours",
    "r": "are",
    "b4": "before",
    "2day": "today",
    "2moro": "tomorrow",
    "2nite": "tonight",
    "4": "for",
    "4u": "for you",
    " gr8": "great",
    "gud": "good",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "lemme": "let me",
    "gimme": "give me",
    "dunno": "don't know",
    "thx": "thanks",
    "thanx": "thanks",
    "ty": "thank you",
    "tysm": "thank you so much",
    "yw": "you're welcome",
    "pls": "please",
    "plz": "please",
    "sry": "sorry",
    "np": "no problem",
    "omg": "oh my god",
    "omw": "on my way",
    "brb": "be right back",
    "btw": "by the way",
    "idk": "I don't know",
    "ik": "I know",
    "ikr": "I know, right",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "tbh": "to be honest",
    "ngl": "not gonna lie",
    "asap": "as soon as possible",
    "fyi": "for your information",
    "afaik": "as far as I know",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing",
    "smh": "shaking my head",
    "ttyl": "talk to you later",
    "cya": "see you",
    "cu": "see you",
    "wyd": "what are you doing",
    "wya": "where are you",
    "hbu": "how about you",
    "hru": "how are you",
    "sup": "what is up",
    "wassup": "what is up",
    "dm": "direct message",
    "irl": "in real life",
    "bff": "best friend forever",
    "bf": "boyfriend",
    "gf": "girlfriend",
    "bro": "brother",
    "sis": "sister",
    "fam": "family",
    "ppl": "people",
    "bc": "because",
    "bcz": "because",
    "cuz": "because",
    "coz": "because",
    "w/": "with",
    "w/o": "without",
    "abt": "about",
    "msg": "message",
    "pic": "picture",
    "pics": "pictures",
    "vid": "video",
    "app": "application",
    "tht": "that",
    "dat": "that",
    "dis": "this",
    "dey": "they",
    "n": "and",
    "nd": "and",
    "bday": "birthday",
    "congrats": "congratulations",
    "prob": "probably",
    "def": "definitely",
    "bcuz": "because",
    "sec": "second",
    "min": "minute",
    "tmr": "tomorrow",
    "nite": "night",
    "gn": "good night",
    "gm": "good morning",
    "tc": "take care",
    "jk": "just kidding",
    "ic": "I see",
    "ofc": "of course",
    "obv": "obviously",
    "legit": "legitimate",
    "tmi": "too much information",
    "fomo": "fear of missing out",
    "yolo": "you only live once",
    "goat": "greatest of all time",
    "afk": "away from keyboard",
    "gg": "good game",
    "wp": "well played",
    "nvm": "never mind",
    "msg me": "message me",
    "xoxo": "hugs and kisses"
}
def repalce_chatwords(text):
    new_text=[]
    for word in text.split():
        if word.lower() in chat_words:
            new_text.append(chat_words[word.lower()])
        else:
            new_text.append(word)
    return " ".join(new_text)

def replace_chatwords(text):
    return repalce_chatwords(text)

def check_spelling(text):
    from textblob import TextBlob

    textblob=TextBlob(text)
    return textblob.correct().string

def remove_stopwords(text):
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word.lower() not in stop_words)


def demojize_text(text):
    import emoji

    new_text=emoji.demojize(text)
    return new_text


def tokenize_data(text):
    import spacy

    return [token.text for token in spacy.blank('en').tokenizer(text)]

def numericalize(tokens):
    return [
        word2idx.get(word, word2idx["<UNK>"])
        for word in tokens
    ]