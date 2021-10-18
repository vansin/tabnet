import pickle

fr = open('work_dirs/table_v4-mask_rcnn_swin-t-p4-w7_fpn_1x_3000/result.pkl', 'rb')
data1 = pickle.load(fr)
print(data1)
print('read-result')
