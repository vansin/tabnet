import os

import matplotlib.pyplot as plt
import mmcv
import pandas as pd
import seaborn as sns

if __name__ == '__main__':

    work_dirs = os.listdir('work_dirs')

    results = []

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
                eval_files.append(data)
            if file_name.endswith('.py'):
                config_file = 'work_dirs/' + work_dir + '/' + file_name

        eval_files.sort(key=lambda x: x['epoch'])
        results.append(eval_files)

    print(results)


intput_data = []


for result in results:

    intput_data.extend(result)

    pass

df = pd.DataFrame.from_dict(intput_data)

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
