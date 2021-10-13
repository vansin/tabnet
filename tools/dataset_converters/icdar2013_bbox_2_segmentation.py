import json

train_in = open('data/icdar2013/annotations/table_ICDAR2013_train.json')
test_in = open('data/icdar2013/annotations/table_ICDAR2013_test.json')

data_train = json.load(train_in)
data_test = json.load(test_in)

for annotation in data_train['annotations']:

    bbox = annotation['bbox']
    segmentation = []
    # left_top
    segmentation.append(int(bbox[0]))
    segmentation.append(int(bbox[1]))
    # left_bottom
    segmentation.append(int(bbox[0]))
    segmentation.append(int(bbox[1] + bbox[3]))
    # right_bottom
    segmentation.append(int(bbox[0] + bbox[2]))
    segmentation.append(int(bbox[1] + bbox[3]))
    # right_top
    segmentation.append(int(bbox[0] + bbox[2]))
    segmentation.append(int(bbox[1]))

    annotation['segmentation'].append(segmentation)

for annotation in data_test['annotations']:

    bbox = annotation['bbox']
    segmentation = []
    # left_top
    segmentation.append(int(bbox[0]))
    segmentation.append(int(bbox[1]))
    # left_bottom
    segmentation.append(int(bbox[0]))
    segmentation.append(int(bbox[1] + bbox[3]))
    # right_bottom
    segmentation.append(int(bbox[0] + bbox[2]))
    segmentation.append(int(bbox[1] + bbox[3]))
    # right_top
    segmentation.append(int(bbox[0] + bbox[2]))
    segmentation.append(int(bbox[1]))

    annotation['segmentation'].append(segmentation)

with open('data/icdar2013/annotations/table_ICDAR2013_segm_train.json',
          'w') as outfile:
    json.dump(data_train, outfile)

with open('data/icdar2013/annotations/table_ICDAR2013_segm_test.json',
          'w') as outfile1:
    json.dump(data_test, outfile1)
