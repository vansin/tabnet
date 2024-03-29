import json
import os

import matplotlib.pyplot as plt
import mmcv
import pandas as pd
import seaborn as sns

prefix = '/home/tml/Nutstore Files/ubuntu/paper/data/iou'


if __name__ == '__main__':

    work_dirs = os.listdir('work_dirs')

    results = []
    best_f1 = []

    for i, work_dir in enumerate(work_dirs):
        work_dir_files = os.listdir('work_dirs/' + work_dir)
        eval_files = []
        config_file = None
        for file_name in work_dir_files:
            if file_name.endswith('_eval.json'):

                name = 'work_dirs/' + work_dir + '/' + file_name
                data_origin = mmcv.load(name)

                epoch = int(name.split('/')[-1].split('_')[1])

                data = dict()
                data['epoch'] = epoch
                config_name = data_origin['config'].split('/')[-1]
                data['config'] = config_name
                data.update(data_origin['metric'])

                try:
                    iou_curves = data_origin['metric']['iou_infos']
                    df = pd.DataFrame.from_dict(iou_curves)
                    df.to_csv(prefix + '/' + work_dir + '=' + str(epoch) + '.csv')
                    # g = sns.lineplot(x='iou', y='f1_score', data=df, markers=True, dashes=False)
                    # g.legend(loc='right', bbox_to_anchor=(1.5, 0.5), ncol=1)
                    # plt.show()
                    # print(plt)
                    eval_files.append(data)
                except Exception as e:
                    print(e)
            if file_name.endswith('.py'):
                config_file = 'work_dirs/' + work_dir + '/' + file_name

        eval_files.sort(key=lambda x: x['epoch'])
        try:
            best_f1.append(
                (work_dir, max(eval_files, key=lambda x: x['f1_score'])['f1_score']))
        except Exception as e:
            print(e)
        results.append(eval_files)
    print(results)


intput_data = []


for result in results:

    intput_data.extend(result)

    pass

with open('/home/tml/Nutstore Files/ubuntu/paper/data/1.json', 'w') as f:
    json.dump(results, f)

df = pd.DataFrame.from_dict(intput_data)


df.to_csv('/home/tml/Nutstore Files/ubuntu/paper/data/1.csv')


g = sns.lineplot(x='epoch', y='bbox_mAP', data=df, hue='config',
                 style='config', markers=True, dashes=False)
# g.legend(loc='right', bbox_to_anchor=(1.5, 0.5), ncol=1)

plt.show()
print(plt)
# for result in results:
#
#     sns.set_theme(style='darkgrid')
#     # Load an example dataset with long-form data
#     df = pd.DataFrame.from_dict(result)
#
#     # Plot the responses for different events and regions
#     sns.lineplot(x='epoch', y='bbox_mAP',
#                  data=df)
#
#     plt.show()
