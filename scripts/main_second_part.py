import json
d = {'a': 1}

with open(f'../First_450/prova.json', 'w') as fp:
    json.dump(d, fp)

# Opening JSON file
f = open('../First_450/prova.json')

# returns JSON object as
# a dictionary
data = json.load(f)
print(data)