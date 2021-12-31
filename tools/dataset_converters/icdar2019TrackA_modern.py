import json

f = open('data/icdar2019/train.json')
data = json.load(f)
print(data)
data_m = dict()
data_m['type'] = data['type']
data_m['categories'] = data['categories']
data_m['annotations'] = data['annotations']
images = []
for image in data['images']:
    if image['file_name'].startswith('cTDaR_t1'):
        images.append(image)
data_m['images'] = images
with open('data/icdar2019/modern_train.json', 'w') as f:
    json.dump(data_m, f)


f = open('data/icdar2019/test.json')
data = json.load(f)
print(data)
data_m = dict()
data_m['type'] = data['type']
data_m['categories'] = data['categories']
data_m['annotations'] = data['annotations']
images = []
for image in data['images']:
    if image['file_name'].startswith('cTDaR_t1'):
        images.append(image)
data_m['images'] = images
with open('data/icdar2019/modern_test.json', 'w') as f:
    json.dump(data_m, f)
