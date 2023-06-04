import os
import shutil

import numpy as np
import pickle as pkl

from absl import app, flags

flags.DEFINE_string(
    'data_dir',
    '',
    ''
)
flags.DEFINE_string(
    'output_dir',
    '',
    ''
)

FLAGS = flags.FLAGS

def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    for f in os.listdir(FLAGS.data_dir):
        print(f)
        curr_path = os.path.join(FLAGS.data_dir, f)
        save_path = os.path.join(FLAGS.output_dir, f)

        if f[-4:] == '.pkl':
            with open(curr_path, 'rb') as f:
                data = pkl.load(f)
                new_data = []
                for i in range(len(data)):
                    data_i = data[i]
                    actions_i = data_i['actions']
                    new_data_i = {
                        k : [] for k in data_i.keys()
                    }
                    for j in range(actions_i.shape[0]):
                        actions = actions_i[j]
                        valid = False
                        for dim in [0, 1, 2, 4]:
                            if np.abs(actions[dim]) <= 0.5:
                                actions[dim] = 0.0
                            else:
                                if actions[dim] > 0:
                                    actions[dim] = 1.0
                                else:
                                    actions[dim] = -1.0

                            if np.abs(actions[dim]) > 0.1:
                                valid = True
                        
                        if valid:
                            for k in new_data_i:
                                new_data_i[k].append(data_i[k][j])
                        
                    if len(new_data_i['actions']) > 0:
                        for k in new_data_i.keys():
                            new_data_i[k] = np.stack(new_data_i[k])
                        new_data.append(new_data_i)
                        
                file = open(save_path, 'wb')
                pkl.dump(new_data, file)
                file.close()
        else:
            shutil.copytree(curr_path, save_path)

if __name__ == '__main__':
    app.run(main)