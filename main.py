import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from Upsampling.model import Model
from Upsampling.configs import FLAGS

import pprint
pp = pprint.PrettyPrinter()

def run():
    if FLAGS.phase=='train':
        FLAGS.train_file = os.path.join(FLAGS.data_dir, 'train/PPU_poisson_256_512_1024.h5')
        print('train_file:',FLAGS.train_file)
        if not FLAGS.restore:
            FLAGS.log_dir = os.path.join(FLAGS.log_dir,FLAGS.model_name)
            try:
                os.makedirs(FLAGS.log_dir)
            except os.error:
                pass
    else:
        FLAGS.test_data = os.path.join(FLAGS.data_dir, 'test/random_sample_2048/*.xyz')
        print(FLAGS.log_dir)
        file_name = FLAGS.log_dir.split('/')[-1]
        FLAGS.out_folder = os.path.join('./result/Ours/')
        if not os.path.exists(FLAGS.out_folder):
            os.makedirs(FLAGS.out_folder)
        print('test_data:',FLAGS.test_data)

    print('checkpoints:',FLAGS.log_dir)
    pp.pprint(FLAGS)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        model = Model(FLAGS,sess)
        if FLAGS.phase == 'train':
            model.train()
        else:
            model.test()


def main(unused_argv):
  run()

if __name__ == '__main__':
  tf.app.run()
