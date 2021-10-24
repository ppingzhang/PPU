import tensorflow as tf
from Upsampling.generator import Generator
from Common.visu_utils import plot_pcd_three_views,point_cloud_three_views
from Common.ops import add_scalar_summary, add_hist_summary, add_train_view_summary
from Upsampling.data_loader import Fetcher
from Common import model_utils
from Common import pc_util
from Common.loss_utils import pc_distance,get_uniform_loss,get_repulsion_loss,discriminator_loss,generator_loss, view_loss, chamfer_for
from tf_ops.sampling.tf_sampling import farthest_point_sample
import logging
import os
from tqdm import tqdm
from glob import glob
import math
from time import time
from termcolor import colored
import numpy as np
from render.render_pro import render_views
from math import log

class Model(object):
	def __init__(self,opts,sess):
			self.sess = sess
			self.opts = opts

	def allocate_placeholders(self):
			self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
			self.global_step = tf.Variable(0, trainable=False, name='global_step')
			self.input_x = tf.placeholder(tf.float32, shape=[self.opts.batch_size,self.opts.num_point,3])
			self.input_gt = tf.placeholder(tf.float32, shape=[self.opts.batch_size, int(4*self.opts.num_point),3])
			self.input_gt_x2 = tf.placeholder(tf.float32, shape=[self.opts.batch_size, int(2*self.opts.num_point),3])
			self.pc_radius = tf.placeholder(tf.float32, shape=[self.opts.batch_size])    



	def build_model(self):
			self.G = Generator(self.opts,self.is_training,name='generator')
			self.up_x1, self.up_x2, self.up_x4 = self.G(self.input_x)
			loss_cd_x4  = self.opts.fidelity_w * pc_distance(self.up_x4, self.input_gt, dis_type='cd', radius=self.pc_radius)

			gt_views_x4 = render_views(self.input_gt, self.opts.batch_size)
			re_views_x4 = render_views(self.up_x4, self.opts.batch_size)
			loss_view_x4 = view_loss(gt_views_x4, re_views_x4)

			loss_repulsion_x4 = self.opts.repulsion_w*get_repulsion_loss(self.up_x4)
			all_loss_x4 = loss_cd_x4  + loss_view_x4 +  loss_repulsion_x4

			self.loss_x4 = {'all_loss':all_loss_x4, 
							'loss_cd':loss_cd_x4,
							'loss_view':loss_view_x4, 
							'loss_repulsion':loss_repulsion_x4, 
							'gt_views':gt_views_x4, 
							're_views':re_views_x4}
			loss_cd_x2  = self.opts.fidelity_w * pc_distance(self.up_x2, self.input_gt_x2, dis_type='cd', radius=self.pc_radius)

			gt_views_x2 = render_views(self.input_gt_x2, self.opts.batch_size)
			re_views_x2 = render_views(self.up_x2, self.opts.batch_size)
			loss_view_x2 = view_loss(gt_views_x2, re_views_x2)

			loss_repulsion_x2 = self.opts.repulsion_w * get_repulsion_loss(self.up_x2)
			all_loss_x2 = loss_cd_x2*0.3

			self.loss_x2 = { 'all_loss':all_loss_x2, 
							'loss_cd':loss_cd_x2, 
							'loss_view':loss_view_x2, 
							'loss_repulsion':loss_repulsion_x2, 
							'gt_views':gt_views_x2, 
							're_views':re_views_x2}



			self.total_gen_loss = self.loss_x4['all_loss'] + self.loss_x2['all_loss'] + tf.losses.get_regularization_loss()
			
			self.setup_optimizer()
			self.summary_all()

	


	def summary_all(self):
			add_scalar_summary('loss/loss_x2/all_loss', self.loss_x2['all_loss'], collections='gen')
			add_scalar_summary('loss/loss_x2/loss_cd', self.loss_x2['loss_cd'], collections='gen')
			add_scalar_summary('loss/loss_x2/loss_view', self.loss_x2['loss_view'], collections='gen')
			add_scalar_summary('loss/loss_x2/loss_repulsion', self.loss_x2['loss_repulsion'], collections='gen')


			add_scalar_summary('loss/loss_x4/all_loss', self.loss_x4['all_loss'], collections='gen')
			add_scalar_summary('loss/loss_x4/loss_cd', self.loss_x4['loss_cd'], collections='gen')
			add_scalar_summary('loss/loss_x4/loss_view', self.loss_x4['loss_view'], collections='gen')
			add_scalar_summary('loss/loss_x4/loss_repulsion', self.loss_x4['loss_repulsion'], collections='gen')


			add_scalar_summary('loss/', self.total_gen_loss, collections='gen')


			add_train_view_summary('loss/loss_x2/gt_views', self.loss_x2['gt_views'], collections='gen')
			add_train_view_summary('loss/loss_x2/re_views', self.loss_x2['re_views'], collections='gen')
			add_train_view_summary('loss/loss_x4/gt_views', self.loss_x4['gt_views'], collections='gen')
			add_train_view_summary('loss/loss_x4/re_views', self.loss_x4['re_views'], collections='gen')

			
			self.g_summary_op = tf.summary.merge_all('gen')

			self.visualize_ops = [self.input_x[0], self.up_x1[0], self.input_gt_x2[0], self.up_x2[0],self.input_gt[0], self.up_x4[0]]
			self.visualize_titles = ['input_x', 'up_x1', 'input_gt_x2','up_x2','input_gt', 'up_x4']

			self.image_x_merged = tf.placeholder(tf.float32, shape=[None, 1500, 3000, 1])
			self.image_x_summary = tf.summary.image('Upsampling', self.image_x_merged, max_outputs=1)

	def setup_optimizer(self):

			learning_rate_g = tf.where(
					tf.greater_equal(self.global_step, self.opts.start_decay_step),
					tf.train.exponential_decay(self.opts.base_lr_g, self.global_step - self.opts.start_decay_step,
																		 self.opts.lr_decay_steps, self.opts.lr_decay_rate, staircase=True),
					self.opts.base_lr_g
			)
			learning_rate_g = tf.maximum(learning_rate_g, self.opts.lr_clip)
			add_scalar_summary('learning_rate/learning_rate_g', learning_rate_g, collections='gen')

			# create pre-generator ops
			gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
			gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

			with tf.control_dependencies(gen_update_ops):
					self.G_optimizers = tf.train.AdamOptimizer(learning_rate_g, beta1=self.opts.beta).minimize(self.total_gen_loss, var_list=gen_tvars,
																																															colocate_gradients_with_ops=True,
																																															global_step=self.global_step,
																																															name='Adam_G')

	def train(self):

			self.allocate_placeholders()
			self.build_model()

			self.sess.run(tf.global_variables_initializer())

			fetchworker = Fetcher(self.opts)
			fetchworker.start()

			self.saver = tf.train.Saver(max_to_keep=None)
			self.writer = tf.summary.FileWriter(self.opts.log_dir, self.sess.graph)

			restore_epoch = 0
			if self.opts.restore:
					restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
					self.saver.restore(self.sess, checkpoint_path)
					self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'a')
					tf.assign(self.global_step, restore_epoch * fetchworker.num_batches).eval()
					restore_epoch += 1

			else:
					os.makedirs(os.path.join(self.opts.log_dir, 'plots'))
					self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'w')

			with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:
					for arg in sorted(vars(self.opts)):
							log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments

			step = self.sess.run(self.global_step)
			start = time()
			for epoch in range(restore_epoch, self.opts.training_epoch):
					logging.info('**** EPOCH %03d ****\t' % (epoch))
					for batch_idx in range(fetchworker.num_batches):

							batch_input_x,batch_data_gt, batch_data_gt_x2, batch_radius = fetchworker.fetch()
							
							feed_dict = {self.input_x: batch_input_x,
													 self.input_gt: batch_data_gt,
													 self.input_gt_x2: batch_data_gt_x2,
													 self.pc_radius: batch_radius,
													 self.is_training: True}
			 
							_, self.loss_x2_d, self.loss_x4_d, loss_all_, summary= self.sess.run(
									[self.G_optimizers, self.loss_x2, self.loss_x4, self.total_gen_loss, self.g_summary_op], feed_dict=feed_dict)

							self.writer.add_summary(summary, step)
							
							if step % self.opts.steps_per_print == 0:
									self.log_string('-----------EPOCH %d Step %d:-------------' % (epoch,step))
									self.log_string('  G_loss   : {}'.format(loss_all_))
									loss_x2_str = 'all:{:<12.5f},cd:{:<12.5f},view:{:<12.5f},repu:{:<12.5f}'.\
																format(self.loss_x2_d['all_loss'], self.loss_x2_d['loss_cd'], self.loss_x2_d['loss_view'], self.loss_x2_d['loss_repulsion'])

									loss_x4_str = 'all:{:<12.5f},cd:{:<12.5f},view:{:<12.5f},repu:{:<12.5f}'.\
																format(self.loss_x4_d['all_loss'], self.loss_x4_d['loss_cd'], self.loss_x4_d['loss_view'], self.loss_x4_d['loss_repulsion'])
									
									self.log_string(loss_x2_str)
									self.log_string(loss_x4_str)
									self.log_string(' Time Cost : {}'.format(time() - start))
									start = time()
									feed_dict = {self.input_x: batch_input_x, self.is_training: False}

									up_x4_p_, up_x2_p_, up_x1_p_ = self.sess.run([self.up_x4, self.up_x2, self.up_x1], feed_dict=feed_dict)


									up_x4_p = np.squeeze(up_x4_p_)
									up_x2_p = np.squeeze(up_x2_p_)
									up_x1_p = np.squeeze(up_x1_p_)

									image_input_x  = point_cloud_three_views(batch_input_x[0])
									image_up_x4 = point_cloud_three_views(up_x4_p[0])
									image_up_x2 = point_cloud_three_views(up_x2_p[0])
									image_up_x1 = point_cloud_three_views(up_x1_p[0])
									image_gt_x4 = point_cloud_three_views(batch_data_gt[0, :, 0:3])
									image_gt_x2 = point_cloud_three_views(batch_data_gt_x2[0, :, 0:3])
									
									image_x_merged = np.concatenate([image_input_x, image_up_x1, image_gt_x2, image_up_x2, image_gt_x4, image_up_x4], axis=1)
								
									image_x_merged = np.expand_dims(image_x_merged, axis=0)
									image_x_merged = np.expand_dims(image_x_merged, axis=-1)
									image_x_summary = self.sess.run(self.image_x_summary, feed_dict={self.image_x_merged: image_x_merged})
									self.writer.add_summary(image_x_summary, step)



							if self.opts.visulize and (step % self.opts.steps_per_visu == 0):
									feed_dict = {self.input_x: batch_input_x,
															self.input_gt: batch_data_gt,
															self.input_gt_x2: batch_data_gt_x2,
															self.is_training: False}
									pcds = self.sess.run([self.visualize_ops], feed_dict=feed_dict)
									pcds = np.squeeze(pcds)  # np.asarray(pcds).reshape([3,self.opts.num_point,3])
									plot_path = os.path.join(self.opts.log_dir, 'plots',
																					 'epoch_%d_step_%d.png' % (epoch, step))
									plot_pcd_three_views(plot_path, pcds, self.visualize_titles)

							step += 1
					if (epoch % self.opts.epoch_per_save) == 0:
							self.saver.save(self.sess, os.path.join(self.opts.log_dir, 'model'), epoch)
							print(colored('Model saved at %s' % self.opts.log_dir, 'white', 'on_blue'))

			fetchworker.shutdown()

	def patch_prediction(self, patch_point):
			# normalize the point clouds
			patch_point, centroid, furthest_distance = pc_util.normalize_point_cloud(patch_point)
			patch_point = np.expand_dims(patch_point, axis=0)
			pred = self.sess.run([self.pred_pc], feed_dict={self.inputs: patch_point})
			pred = np.squeeze(centroid + pred * furthest_distance, axis=0)
			return pred

	def pc_prediction(self, pc):
			## get patch seed from farthestsampling
			points = tf.convert_to_tensor(np.expand_dims(pc,axis=0),dtype=tf.float32)
			start= time()
			print('------------------patch_num_point:',self.opts.patch_num_point)
			seed1_num = int(pc.shape[0] / self.opts.patch_num_point * self.opts.patch_num_ratio)

			## FPS sampling
			seed = farthest_point_sample(seed1_num, points).eval()[0]
			seed_list = seed[:seed1_num]
			print("farthest distance sampling cost", time() - start)
			print("number of patches: %d" % len(seed_list))
			input_list = []
			up_point_list=[]

			patches = pc_util.extract_knn_patch(pc[np.asarray(seed_list), :], pc, self.opts.patch_num_point)

			for point in tqdm(patches, total=len(patches)):
						up_point = self.patch_prediction(point)
						up_point = np.squeeze(up_point,axis=0)
						input_list.append(point)
						up_point_list.append(up_point)

			return input_list, up_point_list

	def test(self):

			self.inputs = tf.placeholder(tf.float32, shape=[1, self.opts.patch_num_point, 3])
			is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
			Gen = Generator(self.opts, is_training, name='generator')

			'''
			2 = 0
			4 = 1
			8 = 1
			16 = 2

			'''
			times = int(log(self.opts.up_ratio,4))

			up_x = self.opts.up_ratio

			if times == 0:
					res_x = up_x
			else:
					res_x = up_x / (4**times)
					

			if res_x>1:
					_,_, self.pred_pc  = Gen(self.inputs, res_x)
			
					for ii in range(times):
							_,_, self.pred_pc  = Gen(self.pred_pc, 4)
			elif res_x == 1 :
					
					_, _, self.pred_pc = Gen(self.inputs, 4)
					for ii in range(times-1):
							print(ii,times-1)
							_,_, self.pred_pc  = Gen(self.pred_pc, 4)


			print(self.pred_pc.shape, 2048*self.opts.up_ratio)

			
			saver = tf.train.Saver()
			restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
			print(checkpoint_path)
			saver.restore(self.sess, checkpoint_path)

			samples = glob(self.opts.test_data)
			point = pc_util.load(samples[0])
			self.opts.num_point = point.shape[0]
			out_point_num = int(self.opts.num_point*self.opts.up_ratio)

			for point_path in samples:
					logging.info(point_path)
					start = time()
					pc = pc_util.load(point_path)[:,:3]


					pc, centroid, furthest_distance = pc_util.normalize_point_cloud(pc)

					if self.opts.jitter:
							pc = pc_util.jitter_perturbation_point_cloud(pc[np.newaxis, ...], sigma=self.opts.jitter_sigma,
																														 clip=self.opts.jitter_max)
							pc = pc[0, ...]

					input_list, pred_list = self.pc_prediction(pc)

					end = time()
					#print("total time: ", end - start)
					pred_pc = np.concatenate(pred_list, axis=0)
					#print(pred_pc.shape)
					pred_pc = (pred_pc * furthest_distance) + centroid
					#print(pred_pc.shape)

					pred_pc = np.reshape(pred_pc,[-1,3])

					if not os.path.exists(os.path.join(self.opts.out_folder, 'ply')):
						os.makedirs(os.path.join(self.opts.out_folder, 'ply'))
					if not os.path.exists(os.path.join(self.opts.out_folder, 'xyz')):
						os.makedirs(os.path.join(self.opts.out_folder, 'xyz'))

					path_ply = os.path.join(self.opts.out_folder, 'ply', point_path.split('/')[-1][:-4] +'_x'+ str(self.opts.up_ratio) +'.ply')
					path_xyz = os.path.join(self.opts.out_folder, 'xyz', point_path.split('/')[-1][:-4] +'_x'+ str(self.opts.up_ratio) +'.xyz')
					
					idx = farthest_point_sample(out_point_num, pred_pc[np.newaxis, ...]).eval()[0]
					pred_pc = pred_pc[idx, 0:3]
					#np.savetxt(path_xyz,pred_pc,fmt='%.6f')
					self.save_as_ply(pred_pc, path_ply)

					#path_ply = os.path.join(self.opts.out_folder, 'ply', point_path.split('/')[-1][:-4] +'.ply')
					path_xyz = os.path.join(self.opts.out_folder, point_path.split('/')[-1][:-4] +'.xyz')
					
					print('----',out_point_num, self.opts.num_point*self.opts.up_ratio)
					idx = farthest_point_sample(out_point_num, pred_pc[np.newaxis, ...]).eval()[0]
					pred_pc = pred_pc[idx, 0:3]
					np.savetxt(path_xyz,pred_pc,fmt='%.6f')

	def log_string(self,msg):
			#global LOG_FOUT
			logging.info(msg)
			self.LOG_FOUT.write(msg + "\n")
			self.LOG_FOUT.flush()

	def save_as_ply(self, vertices, filename):
	
	
			f = open(filename, 'wb')
			np.savetxt(f, vertices, fmt='%f %f %f')
		 
			f.close()

			ply_header = '''ply
				format ascii 1.0
				element vertex %(vert_num)d
				property float x
				property float y
				property float z
				end_header
			\n
			'''


			with open(filename, 'r+') as f:
				old = f.read()

				f.seek(0)
				f.write(ply_header % dict(vert_num=len(vertices)))
				f.write(old)












