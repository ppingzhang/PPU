
from render.quaternion import quaternion_rotate
from render.camera import intrinsic_matrix
from render.drc import drc_projection
import math
import numpy as np
import tensorflow as tf
from functools import reduce


def euler2mat(z=0, y=0, x=0):
	''' Return matrix for rotations around z, y and x axes

	Uses the z, then y, then x convention above

	Parameters
	----------
	z : scalar
	   Rotation angle in radians around z-axis (performed first)
	y : scalar
	   Rotation angle in radians around y-axis
	x : scalar
	   Rotation angle in radians around x-axis (performed last)

	Returns
	-------
	M : array shape (3,3)
	   Rotation matrix giving same rotation as for given angles

	Examples
	--------
	>>> zrot = 1.3 # radians
	>>> yrot = -0.1
	>>> xrot = 0.2
	>>> M = euler2mat(zrot, yrot, xrot)
	>>> M.shape == (3, 3)
	True

	The output rotation matrix is equal to the composition of the
	individual rotations

	>>> M1 = euler2mat(zrot)
	>>> M2 = euler2mat(0, yrot)
	>>> M3 = euler2mat(0, 0, xrot)
	>>> composed_M = np.dot(M3, np.dot(M2, M1))
	>>> np.allclose(M, composed_M)
	True

	You can specify rotations by named arguments

	>>> np.all(M3 == euler2mat(x=xrot))
	True

	When applying M to a vector, the vector should column vector to the
	right of M.  If the right hand side is a 2D array rather than a
	vector, then each column of the 2D array represents a vector.

	>>> vec = np.array([1, 0, 0]).reshape((3,1))
	>>> v2 = np.dot(M, vec)
	>>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
	>>> vecs2 = np.dot(M, vecs)

	Rotations are counter-clockwise.

	>>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
	>>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
	True
	>>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
	>>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
	True
	>>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
	>>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
	True

	Notes
	-----
	The direction of rotation is given by the right-hand rule (orient
	the thumb of the right hand along the axis around which the rotation
	occurs, with the end of the thumb at the positive end of the axis;
	curl your fingers; the direction your fingers curl is the direction
	of rotation).  Therefore, the rotations are counterclockwise if
	looking along the axis of rotation from positive to negative.
	'''
	Ms = []
	if z:
		cosz = math.cos(z)
		sinz = math.sin(z)
		Ms.append(np.array(
				[[cosz, -sinz, 0],
				 [sinz, cosz, 0],
				 [0, 0, 1]]))
	if y:
		cosy = math.cos(y)
		siny = math.sin(y)
		Ms.append(np.array(
				[[cosy, 0, siny],
				 [0, 1, 0],
				 [-siny, 0, cosy]]))
	if x:
		cosx = math.cos(x)
		sinx = math.sin(x)
		Ms.append(np.array(
				[[1, 0, 0],
				 [0, cosx, -sinx],
				 [0, sinx, cosx]]))
	if Ms:
		return reduce(np.dot, Ms[::-1])
	return np.eye(3)



def multi_expand(inp, axis, num):
	inp_big = inp
	for i in range(num):
		inp_big = tf.expand_dims(inp_big, axis)
	return inp_big




def pc_perspective_transform(point_cloud, transform, predicted_translation=None, focal_length=None):

	"""
	:param point_cloud: [B, N, 3]
	:param transform: [B, 4] if quaternion or [B, 4, 4] if camera matrix
	:param predicted_translation: [B, 3] translation vector
	:return:
	"""
	#camera_distance = 2.0  #default相机距离
	#focal_length = 1.875

	camera_distance = 2.0  #default相机距离
	focal_length = 1.0
	pose_quaternion = True

	transform = tf.constant(transform, dtype=tf.float32)




	
	if pose_quaternion:
		pc2 = quaternion_rotate(point_cloud, transform)

		if predicted_translation is not None:
			predicted_translation = tf.expand_dims(predicted_translation, axis=1)
			pc2 += predicted_translation

		xs = tf.slice(pc2, [0, 0, 2], [-1, -1, 1])
		ys = tf.slice(pc2, [0, 0, 1], [-1, -1, 1])
		zs = tf.slice(pc2, [0, 0, 0], [-1, -1, 1])

		# translation part of extrinsic camera
		zs += camera_distance
		# intrinsic transform
		xs *= focal_length
		ys *= focal_length
	else:
		xyz1 = tf.pad(point_cloud, tf.constant([[0, 0], [0, 0], [0, 1]]), "CONSTANT", constant_values=1.0)

		extrinsic = transform
		intr = intrinsic_matrix( focal_length, dims=4)
		intrinsic = tf.convert_to_tensor(intr)
		intrinsic = tf.expand_dims(intrinsic, axis=0)
		intrinsic = tf.tile(intrinsic, [tf.shape(extrinsic)[0], 1, 1])
		full_cam_matrix = tf.matmul(intrinsic, extrinsic)
		full_cam_matrix = tf.expand_dims(extrinsic,0)

		pc2 = tf.matmul(xyz1, tf.transpose(full_cam_matrix, [0, 2, 1]))

		# TODO unstack instead of split
		xs = tf.slice(pc2, [0, 0, 2], [-1, -1, 1])
		ys = tf.slice(pc2, [0, 0, 1], [-1, -1, 1])
		zs = tf.slice(pc2, [0, 0, 0], [-1, -1, 1])

	xs /= zs
	ys /= zs

	zs -= camera_distance
	if predicted_translation is not None:
		zt = tf.slice(predicted_translation, [0, 0, 0], [-1, -1, 1])
		zs -= zt

	xyz2 = tf.concat([zs, ys, xs], axis=2)
	return xyz2




def pointcloud2voxels3d_fast(pc, rgb = None):  # [B,N,3]
    vox_size = 64
    
    vox_size_z = 64

    pc_rgb_stop_points_gradient = False

    batch_size = pc.shape[0]
    num_points = tf.shape(pc)[1]

    has_rgb = rgb is not None

    grid_size = 1.0
    half_size = grid_size / 2

    filter_outliers = True
    valid = tf.logical_and(pc >= -half_size, pc <= half_size)
    valid = tf.reduce_all(valid, axis=-1)

    vox_size_tf = tf.constant([[[vox_size_z, vox_size, vox_size]]], dtype=tf.float32)
    pc_grid = (pc + half_size) * (vox_size_tf - 1)
    indices_floor = tf.floor(pc_grid)
    indices_int = tf.cast(indices_floor, tf.int32)
    batch_indices = tf.range(0, batch_size, 1)
    batch_indices = tf.expand_dims(batch_indices, -1)
    batch_indices = tf.tile(batch_indices, [1, num_points])
    batch_indices = tf.expand_dims(batch_indices, -1)

    indices = tf.concat([batch_indices, indices_int], axis=2)
    indices = tf.reshape(indices, [-1, 4])

    r = pc_grid - indices_floor  # fractional part
    rr = [1.0 - r, r]

    if filter_outliers:
        valid = tf.reshape(valid, [-1])
        indices = tf.boolean_mask(indices, valid)

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][:, :, 0] * rr[pos[1]][:, :, 1] * rr[pos[2]][:, :, 2]
        updates = tf.reshape(updates_raw, [-1])
        if filter_outliers:
            updates = tf.boolean_mask(updates, valid)

        indices_loc = indices
        indices_shift = tf.constant([[0] + pos])
        num_updates = tf.shape(indices_loc)[0]
        indices_shift = tf.tile(indices_shift, [num_updates, 1])
        indices_loc = indices_loc + indices_shift

        #print('--------',indices_loc, updates,  [batch_size, vox_size_z, vox_size, vox_size])
        voxels = tf.scatter_nd(indices_loc, updates, [batch_size, vox_size_z, vox_size, vox_size])
        if has_rgb:
            if pc_rgb_stop_points_gradient:
                updates_raw = tf.stop_gradient(updates_raw)
            updates_rgb = tf.expand_dims(updates_raw, axis=-1) * rgb
            updates_rgb = tf.reshape(updates_rgb, [-1, 3])
            if filter_outliers:
                updates_rgb = tf.boolean_mask(updates_rgb, valid)
            voxels_rgb = tf.scatter_nd(indices_loc, updates_rgb, [batch_size, vox_size_z, vox_size, vox_size, 3])
        else:
            voxels_rgb = None

        return voxels, voxels_rgb

    voxels = []
    voxels_rgb = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                vx, vx_rgb = interpolate_scatter3d([k, j, i])
                voxels.append(vx)
                if vx_rgb  != None:
                	voxels_rgb.append(vx_rgb)

    voxels = tf.add_n(voxels)
    voxels_rgb = tf.add_n(voxels_rgb) if has_rgb else None

    return voxels, voxels_rgb




def pointcloud_project_fast(point_cloud, transform, all_rgb = None):
    has_rgb = all_rgb is not None
    pc_rgb_divide_by_occupancies = False
    pc_rgb_clip_after_conv = False
    ptn_max_projection = False
    tr_pc = pc_perspective_transform(point_cloud, transform)
    voxels, _ = pointcloud2voxels3d_fast(tr_pc)
    voxels = tf.expand_dims(voxels, axis=-1)
    voxels_raw = voxels

    voxels = tf.clip_by_value(voxels, 0.0, 1.0)

    

    
    voxels_rgb = None
    if has_rgb:
        if pc_rgb_divide_by_occupancies:
            voxels_div = tf.stop_gradient(voxels_raw)
            voxels_div = smoothen_voxels3d(cfg, voxels_div, kernel)
            voxels_rgb = voxels_rgb / (voxels_div + cfg.pc_rgb_divide_by_occupancies_epsilon)

        if pc_rgb_clip_after_conv:
            voxels_rgb = tf.clip_by_value(voxels_rgb, 0.0, 1.0)

    if ptn_max_projection:
        proj = tf.reduce_max(voxels, [1])
        drc_probs = None
        proj_depth = None
    else:
        proj, drc_probs = drc_projection(voxels)
        drc_probs = tf.reverse(drc_probs, [2])
        #proj_depth = util.drc.drc_depth_projection(drc_probs, cfg)

    proj = tf.reverse(proj, [1])

    if voxels_rgb is not None:
        voxels_rgb = tf.reverse(voxels_rgb, [2])
        proj_rgb = util.drc.project_volume_rgb_integral(cfg, drc_probs, voxels_rgb)
    else:
        proj_rgb = None

    output_all = {
        "proj": proj,
        "voxels": voxels,
        "tr_pc": tr_pc,
        "voxels_rgb": voxels_rgb,
        "proj_rgb": proj_rgb,
        "drc_probs": drc_probs,
    }

    output = output_all['proj']
    voxels = output_all['voxels']
    tr_pc = output_all['tr_pc']
    return output, voxels, tr_pc


def render_views(pc, batch_size, p_rgb= None):

	skip = 50

	x_rot = np.reshape(np.array([ii for ii in range(-180, 180+1, skip)]+\
							[1 for _ in range(-180, 180+1, skip)]+\
							[2 for _ in range(-180, 180+1, skip)]+\
							[45,    -45, -135, 135, 45,  -45, -135, 135])/180.0 *np.pi, [-1,1])
	y_rot = np.reshape(np.array([3 for _ in range(-180, 180+1, skip)]+\
								[ii for ii in range(-180, 180+1, skip)]+\
								[4 for _ in range(-180, 180+1, skip)]+\
								[-45,  -135, -135, -45, 45,  135,  135,  45])/180.0 *np.pi, [-1, 1])
	z_rot = np.reshape(np.array([5 for _ in range(-180, 180+1, skip)]+\
								[6 for _ in range(-180, 180+1, skip)]+\
								[ii for ii in range(-180, 180+1, skip)]+\
								[-135, -135,  135, 135, -45, -45,  45,   45])/180.0 *np.pi, [-1,1])


	trams = np.concatenate([x_rot, y_rot, z_rot, np.ones(x_rot.shape)], 1)

	all_result = []
	for ii in range(batch_size):
		output, voxels, tr_pc  = pointcloud_project_fast(tf.expand_dims(pc[ii], 0), trams, all_rgb = p_rgb)
		if ii != 0:
			all_result = tf.concat([all_result, output], 0)
		else:
			all_result = output
	return all_result


