import numpy as np
import json 
from textblob import TextBlob
import re

from lxml import html  
import json
import requests
import re
import argparse
from dateutil import parser as dateparser
from time import sleep
from lxml import html as xmlhtml
import numpy as np
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}

def load_data(fname):
    X = []
    y = []
    with open(fname, 'r') as infile:
        for line in infile:
            # Add the line to our JSON block
            jfile = json.loads(line)
            reviewText = jfile["reviewText"]
            rating = jfile["overall"]

            X += [reviewText]
            y += [rating]

    X = np.array(X)
    y = np.array(y)

    return X, y
    
def clean_text(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|"
                           "([^0-9A-Za-z \t]) "
                           "|(\w+:\/\/\S+)", " ", text).split())

def ParseSerialNumbers(keyword="fifa"):
    amazon_url  = "http://www.amazon.com/s/field-keywords=%s" % keyword
    
    page = requests.get(amazon_url, headers = headers).text
    parser = html.fromstring(page)
    asin_list = parser.xpath('//li[@data-asin]/@data-asin')
    import pdb; pdb.set_trace()  # breakpoint 672579d1 //
    
    return asin_list
    
def get_sentiment(text):
    analysis = TextBlob(clean_text(text))
    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

