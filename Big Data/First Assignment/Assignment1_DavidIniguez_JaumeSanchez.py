import requests
import pandas as pd
import pymongo

### Data reading

data = pd.read_csv('Top100-US.csv', header = 0, delimiter=';')

### URL and Keys of weatherAPI

url = "https://weatherapi-com.p.rapidapi.com/current.json"

headers = {
	"X-RapidAPI-Key": "2cb604fa2fmshd37115236f5451ep1840d0jsn65643604a4c4",
	"X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
}

### Name of the mongo database and collection, as well as the port

MONGO_NAME = 'mongo'
MONGO_PORT = 27017
MONGO_DB = "BigDataAssignment1"
MONGO_COLL = "Top100_filled"

myclient = pymongo.MongoClient(MONGO_NAME, MONGO_PORT)
mycol = myclient[MONGO_DB][MONGO_COLL]

for i in data.index:
    zipcode = str(data['Zip'][i])
    ### If zipcode has less than 5 characters, we have to add zeros at the beginning
    if len (zipcode) <= 5:
        s = ''
        for i in range (5-len(zipcode)):
            s += '0'
        zipcode = s + zipcode
    
    querystring = {"q":zipcode}
    ### Connection to thr API
    response = requests.get(url, headers=headers, params=querystring)
    response_json = response.json()
    #     print (response_json)
    # print (response_json['current']['condition']['text'])

    ### Extraction of the relevant features

    weather_condition = response_json['current']['condition']['text']
    local_time = response_json['location']['localtime']
    city = data['City'][i]

    ### Dictionary building in order to add element to the database
    
    mydict = { "zip": int(zipcode), 
              "city": city, 
              "created_at":local_time, 
              "weather":weather_condition}
    
    # print (mydict)

    x = mycol.insert_one(mydict)

