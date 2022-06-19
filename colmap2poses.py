# most of the code is refactored colmap parsing from LLFF, a GPLv3 project.
# this is meant to be comparable to instant-ngp's colmap2nerf.py, and uses the
# same arguments, with additional functions for masking and LLFF format poses.
# LLFF format .npy files will always be made, remove if you ONLY want NeRF format

#~~~provide a dataset folder path with another "image" subfolder with the pics~~~
# ~~~colmap can be installed to PATH or linked as an argument (--colmap_path)~~~

# usage: (for LLFF format, nvdiffrec)
# colmap2poses.py --mask "/path/to/dataset/"
# a "mask" folder is required, manual or passed with --mask as shown

# if you wish for NeRF format, pass --json
# masks must be applied to the image for NeRF format, (--alpha)
# nvdiffrec loads the .npy first, but for NeRF format datasets, the image 
# extentions must removed to prevent errors

# video and aabb_scale options are from instant-ngp, follow their guide to use
# https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md

import argparse
import os
import sys

import numpy as np
import json
import math
import cv2
import subprocess
import struct
import collections
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="convert colmap to NeRF or LLFF(NeRD) format, automask objects, and process video")
    parser.add_argument('--json', action="store_true", 
		    default='false', help='output json/instant-ngp format')
    parser.add_argument("--mask", action="store_true",
                    default="", help="use rembg to automatically mask the entire image folder. reccomended to use manual masks for better results in nvdiffrec.")
    parser.add_argument("--alpha", action="store_true",
                    default="", help="only works when --mask is enabled, applys a mask to the alpha channel of the image (instead of a separate folder). enables masking in instant-ngp.")
    parser.add_argument("--video_in", 
                    default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also. needs ffmpeg in path or installed.")
    parser.add_argument("--video_fps", 
                    default=3)
    parser.add_argument("--time_slice", 
                    default="", help="time (in seconds) in the format t1,t2 within which the images should be generated from the video. eg: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video")
    parser.add_argument('--match_type', type=str, 
                    default='exhaustive_matcher', help='type of matcher used.  Valid options: exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
    parser.add_argument("--database_db", 
                    default="database.db", help="colmap database filename")
    parser.add_argument("--aabb_scale", 
                    default=2, choices=["1","2","4","8","16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
    parser.add_argument("--out_json", 
                    default="transforms_test.json", help="output json filename")
    parser.add_argument("--images", 
                    default="images", help="name of the directory containing the images")
    parser.add_argument('--colmap_path', type=str,
                    default='', help='path to colmap batch file, if not in PATH')
    parser.add_argument('scenedir', type=str,
                    help='input scene directory, where the dataset will be stored, required to run')
    args = parser.parse_args()
    return args

args = parse_args()

if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
    print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
    sys.exit()

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])

def run_colmap(basedir, match_type, colmap_path="Colmap"):
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    feature_extractor_args = [
        colmap_path, 'feature_extractor', 
            '--database_path', os.path.join(basedir, args.database_db), 
            '--image_path', os.path.join(basedir, args.images),
            '--ImageReader.single_camera', '1',
            # '--SiftExtraction.use_gpu', '0',
    ]
    feat_output = ( subprocess.check_output(feature_extractor_args, universal_newlines=True) )
    logfile.write(feat_output)
    print('Features extracted')

    exhaustive_matcher_args = [
        colmap_path, match_type, 
            '--database_path', os.path.join(basedir, args.database_db), 
    ]

    match_output = ( subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
    logfile.write(match_output)
    print('Features matched')
    
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)
    
    mapper_args = [
        colmap_path, 'mapper',
            '--database_path', os.path.join(basedir, args.database_db),
            '--image_path', os.path.join(basedir, args.images),
            '--output_path', os.path.join(basedir, 'sparse'), # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
    ]

    map_output = ( subprocess.check_output(mapper_args, universal_newlines=True) )
    logfile.write(map_output)
    logfile.close()
    print('Sparse map created')
    
    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )

def run_ffmpeg(args):
    images = args.images
    video = args.video_in
    fps = float(args.video_fps) or 1.0
    print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
    if (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
        sys.exit(1)
    try:
        shutil.rmtree(f"{args.scenedir}/{args.images}/")
    except:
        pass
    os.mkdir(os.path.normpath(f"{args.scenedir}/{args.images}/"))
    
    time_slice_value = ""
    time_slice = args.time_slice
    if time_slice:
        start, end = time_slice.split(",")
        time_slice_value = f",select='between(t\,{start}\,{end})'"
    do_system(f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}, scale=-2:512\"  {os.path.join(args.scenedir,args.images)}\%04d.jpg")
 
def gen_poses(basedir, match_type, colmap_path, factors=None):
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print('Need to run COLMAP')
        run_colmap(basedir, match_type, colmap_path)
    else:
        print('Don\'t need to run COLMAP')
        
    print( 'Post-colmap')
    
    load_save_pose(basedir)

    # poses, pts3d, perm = load_colmap_data(basedir)

    # save_poses(basedir, poses, pts3d, perm)
    
    if factors is not None:
        print( 'Factors:', factors)
        minify(basedir, factors)
    
    print( 'Done with imgs2poses' )
    
    return True

def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm

def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        # print( i, close_depth, inf_depth )
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)
    
def save_views(realdir,names):
    with open(os.path.join(realdir,'view_imgs.txt'), mode='w') as f:
        f.writelines('\n'.join(names))
    f.close()


def load_save_pose(realdir):

    # load colmap data 
    model_path = os.path.join(realdir, 'sparse/0/')
    modeldata = read_model(model_path, ".bin")
    camdata = modeldata[0]
    
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', cam)

    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h,w,f]).reshape([3,1])

    
    imdata = modeldata[1]

    real_ids = [k for k in imdata]

    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])

    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))

    # if (len(names)< 32):
    #     raise ValueError(f'{realdir} only {len(names)} images register, need Re-run colmap or reset the threshold')


    perm = np.argsort(names)
    sort_names = [names[i] for i in perm]
    save_views(realdir,sort_names)

    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)

    
    pts3d = modeldata[2]

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    # save pose
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < real_ids.index(ind):
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[real_ids.index(ind)] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape)
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )

    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)

        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)

    np.save(os.path.join(realdir, 'poses_bounds.npy'), save_arr)
    
def convert_to_json (args):
    AABB_SCALE = int(args.aabb_scale)
    text = os.path.normpath(args.scenedir + '/text')
    OUT_PATH = os.path.normpath(args.scenedir+ '/' + args.out_json)
    sparce = os.path.normpath(args.scenedir + '/sparse')
    
    try:
        shutil.rmtree(text)
    except:
        pass
    do_system(f"mkdir {text}")
    do_system(f"colmap model_converter --input_path {sparce}/0 --output_path {text} --output_type TXT")
    
    print(f"outputting to {OUT_PATH}...")
    with open(os.path.join(text,"cameras.txt"), "r") as f:
        angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

    with open(os.path.join(text,"images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if  i % 2 == 1:
                elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                name = str(f"./{args.images}/{elems[9]}")
                b=sharpness(os.path.normpath(f"{args.scenedir}/{args.images}/{elems[9]}"))
                print(name, "sharpness=",b)
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                c2w[0:3,2] *= -1 # flip the y and z axis
                c2w[0:3,1] *= -1
                c2w = c2w[[1,0,2,3],:] # swap y and z
                c2w[2,:] *= -1 # flip whole world upside down

                up += c2w[0:3,1]

                frame={"file_path":name,"sharpness":b,"transform_matrix": c2w}
                out["frames"].append(frame)
    nframes = len(out["frames"])
    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1


    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3,:]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.01:
                totp += p*w
                totw += w
    totp /= totw
    print(totp) # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3,3] -= totp

    avglen = 0.
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
    avglen /= nframes
    print("avg camera distance from origin", avglen)
    for f in out["frames"]:
        f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes,"frames")
    print(f"writing {OUT_PATH}")
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)
        
def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
    
def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def read_points3d_binary(path_to_model_file):
    """
see: src/base/reconstruction.cc
    void Reconstruction::ReadPoints3DBinary(const std::string& path)
    void Reconstruction::WritePoints3DBinary(const std::string& path)
"""
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D

def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

if __name__=='__main__':
    if args.video_in != "":
            run_ffmpeg(args)
    if args.mask != "":
        from rembg import remove
        if args.alpha != "":
            print("applying alpha masks, images will be saved as _alpha.png (this is VERY slow)")
            try: os.mkdir(os.path.normpath(f"{args.scenedir}/images_alpha/"))
            except: pass
            for img in os.listdir(os.path.normpath(f"{args.scenedir}/{args.images}/")):
                with open(os.path.normpath(f"{args.scenedir}/{args.images}/{img}"), "rb") as i:
                    with open(os.path.normpath(f"{args.scenedir}/images_alpha/{img}_alpha.png"), 'wb') as o:
                        im = i.read()
                        try:
                            output = remove(im, alpha_matting=True)
                            o.write(output)
                        except:
                            print(f"matting failed for {img}, applying mask")
                            output = remove(im)
                            o.write(output)
                        print(f"{img}_alpha.png done")
            if (input(f"please filter images with bad alpha in images_alpha. continue? (Y/n)").lower().strip()+"n")[:1] != "n":
                sys.exit(1)
            print('images_alpha is the new default image folder, please do not rename it or the files inside after colmapping')
            args.images = "images_alpha"
        else:
            print("producing mask images")
            try: os.mkdir(os.path.normpath(f"{args.scenedir}/masks/"))
            except: pass
            for img in os.listdir(os.path.normpath(f"{args.scenedir}/{args.images}/")):
                im = cv2.imread(os.path.normpath(f"{args.scenedir}/{args.images}/{img}"))
                output = remove(im, only_mask=True)
                cv2.imwrite(os.path.normpath(f"{args.scenedir}/masks/{img}"), output)
                print(f"{img} done")
                
    if args.colmap_path != '':
        if os.name == 'nt' and "COLMAP.bat" not in args.colmap_path:
            sys.exit("colmap path fail")
        elif os.path.exists(args.colmap_path):
            print ('valid custom colmap path!')
            colmap_path = os.path.normpath(args.colmap_path)
            print("using colmap path:", colmap_path)
        else: sys.exit("colmap path fail")
    elif os.name == 'nt':
        colmap_path = "colmap.bat"
    gen_poses(args.scenedir, args.match_type, colmap_path)
    if args.json == 1:
        convert_to_json(args)
