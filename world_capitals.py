import random
import re
import fuzzywuzzy
from fuzzywuzzy import fuzz
import Levenshtein
import nltk

capitals_dict = dict()
fhan =open("capitals_dict-1.txt")
for line in fhan:
    if line[0] != "'": continue
    indices = [i.start() for i in re.finditer("'", line)]
    country_temp = line[indices[0]+1:indices[1]]
    capital_temp = line[indices[2]+1:indices[3]]
    capitals_dict[country_temp] = capital_temp
fhan.close()

countries = list(capitals_dict.keys())

while True:
    country = random.choice(countries)
    capital_guess = input("What is the capital of " + country + "?")
    print (fuzz.ratio(capitals_dict[country] .lower(), capital_guess.lower()))
    if len(capital_guess)<1:
        break
    elif capitals_dict[country] .lower() == capital_guess.lower():
        print("you are right!! well done")
    elif fuzz.ratio(capitals_dict[country].lower(), capital_guess.lower())>80:
        print("you close enough!! well done", capitals_dict[country])
    else:
        print("that was not correct")
