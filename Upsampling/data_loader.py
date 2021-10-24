

import numpy as np
import h5py
import queue
import threading
from Common import point_operation


def load_h5_data(h5_filename='', opts=None, skip_rate = 1, use_randominput=True):
    num_point = opts.num_point
    

    print("h5_filename : ",h5_filename)
    if use_randominput:
        print("use randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f['poisson_1024'][:]
        gt = f['poisson_1024'][:]
        gt_x2 = f['poisson_512'][:]
    else:
        print("Do not randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f['poisson_256'][:] 
        gt = f['poisson_1024'][:]
        gt_x2 = f['poisson_512'][:]

    print(input.shape, gt.shape, gt_x2.shape, '====++=====')

    #name = f['name'][:]
    assert len(input) == len(gt)

    print("Normalization the data")
    data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    

    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)


    gt_x2[:, :, 0:3] = gt_x2[:, :, 0:3] - centroid
    gt_x2[:, :, 0:3] = gt_x2[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    gt = gt[::skip_rate]
    input = input[::skip_rate]
    gt_x2 = gt_x2[::skip_rate]
    
    data_radius = data_radius[::skip_rate]

    print("total %d samples" % (len(input)))
    return input, gt, gt_x2, data_radius


class Fetcher(threading.Thread):
    def __init__(self, opts):
        super(Fetcher,self).__init__()
        self.queue = queue.Queue(50)
        self.stopped = False
        self.opts = opts
        self.use_random_input = self.opts.use_non_uniform
        self.input_data, self.gt_data, self.gt_x2_data, self.radius_data = load_h5_data(self.opts.train_file,opts=self.opts,use_randominput=self.use_random_input)
        self.batch_size = self.opts.batch_size
        self.sample_cnt = self.input_data.shape[0]
        self.patch_num_point = self.opts.patch_num_point
        self.num_batches = self.sample_cnt//self.batch_size
        print ("NUM_BATCH is %s"%(self.num_batches))

    def run(self):
        while not self.stopped:
            idx = np.arange(self.sample_cnt)
            np.random.shuffle(idx)
            self.input_data = self.input_data[idx, ...]
            self.gt_data = self.gt_data[idx, ...]
            self.gt_x2_data = self.gt_x2_data[idx, ...]
            self.radius_data = self.radius_data[idx, ...]

            for batch_idx in range(self.num_batches):
                if self.stopped:
                    return None
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                batch_input_data = self.input_data[start_idx:end_idx, :, :].copy()
                batch_data_gt = self.gt_data[start_idx:end_idx, :, :].copy()
                batch_data_gt_x2 = self.gt_x2_data[start_idx:end_idx, :, :].copy()

                radius = self.radius_data[start_idx:end_idx].copy()

                #print('======++++++++',batch_input_data.shape, batch_data_gt.shape)

                if self.use_random_input:
                    new_batch_input = np.zeros((self.batch_size, self.patch_num_point, batch_input_data.shape[2]))
                    for i in range(self.batch_size):
                        idx = point_operation.nonuniform_sampling(self.input_data.shape[1], sample_num=self.patch_num_point)
                        new_batch_input[i, ...] = batch_input_data[i][idx]
                    batch_input_data = new_batch_input
                if self.opts.augment:
                    batch_input_data = point_operation.jitter_perturbation_point_cloud(batch_input_data, sigma=self.opts.jitter_sigma, clip=self.opts.jitter_max)
                    batch_input_data, batch_data_gt, batch_data_gt_x2 = point_operation.rotate_point_cloud_and_gt_and_gt_x2(batch_input_data, batch_data_gt, batch_data_gt_x2)
                    batch_input_data, batch_data_gt, batch_data_gt_x2, scales = point_operation.random_scale_point_cloud_and_gt_and_gt_x2(batch_input_data,
                                                                                              batch_data_gt,
                                                                                              batch_data_gt_x2,
                                                                                              scale_low=0.8,
                                                                                              scale_high=1.2)
                   
                    radius = radius * scales
                
                self.queue.put((batch_input_data[:,:,:3],  batch_data_gt[:,:,:3], batch_data_gt_x2[:,:,:3], radius))
                
        return None

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        print ("Shutdown .....")
        while not self.queue.empty():
            self.queue.get()
        print ("Remove all queue data")


