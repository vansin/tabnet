import json

# Opening JSON file
f_train = open('/home/tml/vansin/paper/tabnet/data/table/annotations/tablebank_word_train.json')
f_val = open('/home/tml/vansin/paper/tabnet/data/table/annotations/tablebank_word_val.json')
f_test = open('/home/tml/vansin/paper/tabnet/data/table/annotations/tablebank_word_test.json')

# returns JSON object as
# a dictionary
data_train = json.load(f_train)
data_val = json.load(f_val)
data_test = json.load(f_test)

train_image_set = set()
val_image_set = set()
test_image_set = set()

images_train = data_train['images']
images_val = data_val['images']
images_test = data_test['images']

for i,image in enumerate(images_train):
    train_image_set.add(image['file_name'])

for i,image in enumerate(images_val):
    val_image_set.add(image['file_name'])

for i,image in enumerate(images_test):
    test_image_set.add(image['file_name'])

print(train_image_set.intersection(val_image_set))
print(train_image_set.intersection(test_image_set))

# data = {}
data.pop('images')
images = []
for i,image in enumerate(images_train):
    if i % 100 == 0:
        images.append(image)
data['images'] = images
with open("/home/tml/vansin/paper/tabnet/data/table/annotations/tablebank_word_train_son.json", "w") as outfile:
    json.dump(data, outfile)
print(data)

