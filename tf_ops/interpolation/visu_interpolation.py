''' Visualize part segmentation '''
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('./PPU/tf_ops/interpolation')
#from show3d_balls import showpoints
import numpy as np
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf



import numpy as np
import ctypes as ct
import cv2
import sys
showsz=800
mousex,mousey=0.5,0.5
zoom=1.0
changed=True
def onmouse(*args):
    global mousex,mousey,changed
    y=args[1]
    x=args[2]
    mousex=x/float(showsz)
    mousey=y/float(showsz)
    changed=True
cv2.namedWindow('show3d')
cv2.moveWindow('show3d',0,0)
cv2.setMouseCallback('show3d',onmouse)

dll=np.ctypeslib.load_library('render_balls_so','./PPU/tf_ops/renderball/')

def showpoints(xyz,c_gt=None, c_pred = None ,waittime=0,showrot=False,magnifyBlue=0,freezerot=False,background=(0,0,0),normalizecolor=True,ballradius=10):
    global showsz,mousex,mousey,zoom,changed, idx
    idx = 0
    if len(xyz.shape) == 2:
        xyz = np.expand_dims(xyz, 0)

    num_samples = xyz.shape[0]

    for i in range(num_samples):
        xyz[i]=xyz[i]-xyz[i].mean(axis=0)
        radius=((xyz[i]**2).sum(axis=-1)**0.5).max()
        xyz[i]/=(radius*2.2)/showsz
    if c_gt is None:
        c0=np.zeros((len(xyz[idx]),),dtype='float32')+255
        c1=np.zeros((len(xyz[idx]),),dtype='float32')+255
        c2=np.zeros((len(xyz[idx]),),dtype='float32')+255
    else:
        c0=c_gt[:,0]
        c1=c_gt[:,1]
        c2=c_gt[:,2]


    if normalizecolor:
        c0/=(c0.max()+1e-14)/255.0
        c1/=(c1.max()+1e-14)/255.0
        c2/=(c2.max()+1e-14)/255.0


    c0=np.require(c0,'float32','C')
    c1=np.require(c1,'float32','C')
    c2=np.require(c2,'float32','C')

    show=np.zeros((showsz,showsz,3),dtype='uint8')
    def render():
        rotmat=np.eye(3)
        if not freezerot:
            xangle=(mousey-0.5)*np.pi*1.2
        else:
            xangle=0
        rotmat=rotmat.dot(np.array([
            [1.0,0.0,0.0],
            [0.0,np.cos(xangle),-np.sin(xangle)],
            [0.0,np.sin(xangle),np.cos(xangle)],
            ]))
        if not freezerot:
            yangle=(mousex-0.5)*np.pi*1.2
        else:
            yangle=0
        rotmat=rotmat.dot(np.array([
            [np.cos(yangle),0.0,-np.sin(yangle)],
            [0.0,1.0,0.0],
            [np.sin(yangle),0.0,np.cos(yangle)],
            ]))
        rotmat*=zoom
        nxyz=xyz[idx].dot(rotmat)+[showsz/2,showsz/2,0]

        ixyz=nxyz.astype('int32')
        show[:]=background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        if magnifyBlue>0:
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=0))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=0))
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=1))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=1))
        if showrot:
            cv2.putText(show,'xangle %d'%(int(xangle/np.pi*180)),(30,showsz-30),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'yangle %d'%(int(yangle/np.pi*180)),(30,showsz-50),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'zoom %d%%'%(int(zoom*100)),(30,showsz-70),0,0.5,cv2.cv.CV_RGB(255,0,0))
    changed=True
    while True:
        if changed:
            render()
            changed=False
        cv2.imshow('show3d',show)
        if waittime==0:
            cmd=cv2.waitKey(10)%256
        else:
            cmd=cv2.waitKey(waittime)%256
        if cmd==ord('q'):
            break
        elif cmd==ord('Q'):
            sys.exit(0)

        if cmd==ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                if c_gt is None:
                    c0=np.zeros((len(xyz[idx]),),dtype='float32')+255
                    c1=np.zeros((len(xyz[idx]),),dtype='float32')+255
                    c2=np.zeros((len(xyz[idx]),),dtype='float32')+255
                else:
                    c0=c_gt[:,0]
                    c1=c_gt[:,1]
                    c2=c_gt[:,2]
            else:
                if c_pred is None:
                    c0=np.zeros((len(xyz[idx]),),dtype='float32')+255
                    c1=np.zeros((len(xyz[idx]),),dtype='float32')+255
                    c2=np.zeros((len(xyz[idx]),),dtype='float32')+255
                else:
                    c0=c_pred[:,0]
                    c1=c_pred[:,1]
                    c2=c_pred[:,2]
            if normalizecolor:
                c0/=(c0.max()+1e-14)/255.0
                c1/=(c1.max()+1e-14)/255.0
                c2/=(c2.max()+1e-14)/255.0
            c0=np.require(c0,'float32','C')
            c1=np.require(c1,'float32','C')
            c2=np.require(c2,'float32','C')
            changed = True


        if cmd==ord('j'):
            idx = (idx + 1) % num_samples
            print(idx)
            changed=True

        if cmd==ord('k'):
            idx = (idx - 1) % num_samples
            print(idx)
            changed=True

        if cmd==ord('n'):
            zoom*=1.1
            changed=True
        elif cmd==ord('m'):
            zoom/=1.1
            changed=True
        elif cmd==ord('r'):
            zoom=1.0
            changed=True
        elif cmd==ord('s'):
            cv2.imwrite('show3d.png',show)
        if waittime!=0:
            break
    return cmd


def showpoints_frame(xyz,c_gt=None, c_pred = None ,waittime=0,showrot=False,magnifyBlue=0,freezerot=False,background=(0,0,0),normalizecolor=True,ballradius=10):
    #global showsz,mousex,mousey,zoom,changed, idx
    
    
    
    showsz=800
    mousex,mousey=0.5,0.5
    zoom=1.0
    changed=True
    idx = 0
    
    if len(xyz.shape) == 2:
        xyz = np.expand_dims(xyz, 0)

    num_samples = xyz.shape[0]

    for i in range(num_samples):
        xyz[i]=xyz[i]-xyz[i].mean(axis=0)
        radius=((xyz[i]**2).sum(axis=-1)**0.5).max()
        xyz[i]/=(radius*2.2)/showsz
    if c_gt is None:
        c0=np.zeros((len(xyz[idx]),),dtype='float32')+255
        c1=np.zeros((len(xyz[idx]),),dtype='float32')+255
        c2=np.zeros((len(xyz[idx]),),dtype='float32')+255
    else:
        c0=c_gt[:,0]
        c1=c_gt[:,1]
        c2=c_gt[:,2]


    if normalizecolor:
        c0/=(c0.max()+1e-14)/255.0
        c1/=(c1.max()+1e-14)/255.0
        c2/=(c2.max()+1e-14)/255.0


    c0=np.require(c0,'float32','C')
    c1=np.require(c1,'float32','C')
    c2=np.require(c2,'float32','C')

    show=np.zeros((showsz,showsz,3),dtype='uint8')
    def render():
        rotmat=np.eye(3)

        xangle=-np.pi/4
        rotmat=rotmat.dot(np.array([
            [1.0,0.0,0.0],
            [0.0,np.cos(xangle),-np.sin(xangle)],
            [0.0,np.sin(xangle),np.cos(xangle)],
            ]))

        yangle=-np.pi/4
        rotmat=rotmat.dot(np.array([
            [np.cos(yangle),0.0,-np.sin(yangle)],
            [0.0,1.0,0.0],
            [np.sin(yangle),0.0,np.cos(yangle)],
            ]))
        rotmat*=zoom
        nxyz=xyz[idx].dot(rotmat)+[showsz/2,showsz/2,0]

        ixyz=nxyz.astype('int32')
        show[:]=background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        if magnifyBlue>0:
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=0))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=0))
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=1))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=1))
        if showrot:
            cv2.putText(show,'xangle %d'%(int(xangle/np.pi*180)),(30,showsz-30),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'yangle %d'%(int(yangle/np.pi*180)),(30,showsz-50),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'zoom %d%%'%(int(zoom*100)),(30,showsz-70),0,0.5,cv2.cv.CV_RGB(255,0,0))
    
    render()
    return show



pts2 = np.array([[0,0,1],[1,0,0],[0,1,0],[1,1,0]]).astype('float32')
xyz1 = np.random.random((100,3)).astype('float32')
xyz2 = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,1]]).astype('float32')

def fun(xyz1,xyz2,pts2):
    with tf.device('/cpu:0'):
        points = tf.constant(np.expand_dims(pts2,0))
        xyz1 = tf.constant(np.expand_dims(xyz1,0))
        xyz2 = tf.constant(np.expand_dims(xyz2,0)) #
        dist, idx = three_nn(xyz1, xyz2)#xyz2点比较稀疏
        #weight = tf.ones_like(dist)/3.0
        #对距离进行处理

        #print(idx, dist)
        #norm1 = tf.reduce_sum(dist,axis=2,keep_dims=True)  #[x,y,,z]坐标求和
        #output = tf.nn.top_k(norm1, 2) #values/indices
    #3print('output:',output)




       
       

        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm, [1,1,3])
        
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points, idx, weight)
    with tf.Session('') as sess:
        tmp,pts1,d,w, id_ = sess.run([xyz1, interpolated_points, dist, weight, idx])
        #print w
        #print(d)
        #print(id_)
        pts1 = pts1.squeeze()
    return pts1



pts1 = fun(xyz1,xyz2,pts2) 
all_pts = np.zeros((104,3))
all_pts[0:100,:] = pts1
all_pts[100:,:] = pts2
all_xyz = np.zeros((104,3))
all_xyz[0:100,:]=xyz1
all_xyz[100:,:]=xyz2
showpoints(xyz2, pts2, ballradius=8)
showpoints(xyz1, pts1, ballradius=8)
showpoints(all_xyz, all_pts, ballradius=8)

