from tweepy import Stream
from tweepy.streaming import StreamListener
import tweepy
from tweepy import OAuthHandler
 


import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import argparse
import string
#import config
import json

class MyListener(StreamListener):
    """Custom StreamListener for streaming data."""

    def __init__(self, data_dir, query):
        self.query_fname = format_filename(query)
        self.data_dir = data_dir
        self.outfile = "%s/stream_%s.json" % (data_dir, self.query_fname)
        self.geo_data = {
                "type": "FeatureCollection",
                "features": []
            }
        self.data = []
    def on_data(self, data):
        try:
            tweet = json.loads(data)
            print tweet["coordinates"]
            if tweet['coordinates']:
                geo_json_feature = {
                    "type": "Feature",
                    "geometry": tweet['coordinates'],
                    "properties": {
                        "text": tweet['text'],
                        "created_at": tweet['created_at']
                    }
                }
                self.geo_data['features'].append(geo_json_feature)
                outfile = "%s/%s_geoData.json" % (self.data_dir, self.query_fname)
                with open(outfile, 'w') as fout:
                    fout.write(json.dumps(self.geo_data, indent=4))

            if "text" in tweet:
                print("message\n=======\n")
                print(tweet["text"].encode('utf-8').strip())

                tweetData = {
                            #'ratings':ratings_dict,
                            #'reviews':reviews_list,
                            'text':tweet["text"]
                        }
                self.data.append(tweetData)        
                with open(self.outfile, 'w') as fout:
                    fout.write(json.dumps(self.data, indent=4))



        except BaseException as e:
            print("Error on_data: %s" % str(e))
            time.sleep(5)
        return True

    def on_error(self, status):
        print(status)
        return True


def format_filename(fname):
    return ''.join(convert_valid(one_char) for one_char in fname)


def convert_valid(one_char):
    valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return '_'

@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status

if __name__ == '__main__':
    """Get parser for command line arguments."""
    parser = argparse.ArgumentParser(description="Twitter Downloader")
    parser.add_argument("-q",
                        "--query",
                        dest="query",
                        help="Query/Filter",
                        default='-')
    parser.add_argument("-d",
                        "--data-dir",
                        dest="data_dir",
                        default="datasets",
                        help="Output/Data Directory")
    
    args = parser.parse_args()
    
    # GET THE KEYS FROM 
    # https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/
    
    consumer_key = 'ADD KEY'
    consumer_secret = 'ADD KEY'
    access_token = 'ADD KEY'
    access_secret = 'ADD KEY'
     
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth)

    twitter_stream = Stream(auth, MyListener(args.data_dir, args.query))
    twitter_stream.filter(track=[args.query])