import numpy as np 
import utils as ut 
import urllib2
import json




def duplicate_reply(message):
    """ YOUR CODE HERE """
    return "%s" % message

def sentiment_reply(message):
    """ YOUR CODE HERE """
    return "The sentiment of your message is: YOUR CODE HERE"

def chuck_norris_reply(message):
    req = urllib2.Request("http://api.icndb.com/jokes/random/5")
    full_json = urllib2.urlopen(req).read()
    full = json.loads(full_json)

    n_jokes = len(full['value'])

    return full['value'][np.random.randint(0, n_jokes)]['joke']