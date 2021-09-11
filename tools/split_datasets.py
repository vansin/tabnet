import json

# Opening JSON file
f = open('/home/tml/vansin/paper/tabnet/data/table/annotations/tablebank_word_train.json', )

# returns JSON object as
# a dictionary
data = json.load(f)



images_ori = data['images']
# data = {}
data.pop('images')

images = []
for i,image in enumerate(images_ori):
    if i % 100 == 0:
        images.append(image)

data['images'] = images


with open("/home/tml/vansin/paper/tabnet/data/table/annotations/tablebank_word_train_son.json", "w") as outfile:
    json.dump(data, outfile)


print(data)
