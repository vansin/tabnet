import json

f_test = open('data/table/annotations/latex_train.json')

data_train = json.load(f_test)

print(data_train)
