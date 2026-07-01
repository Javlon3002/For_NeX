# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Distribute under MIT License
# Authors:
#  - Suttisak Wizadwongsa <suttisak.w_s19[-at-]vistec.ac.th>
#  - Pakkapon Phongthawee <pakkapon.p_s19[-at-]vistec.ac.th>
#  - Jiraphon Yenphraphai <jiraphony_pro[-at-]vistec.ac.th>
#  - Supasorn Suwajanakorn <supasorn.s[-at-]vistec.ac.th>

from __future__ import division
from __future__ import print_function

import argparse
import getpass

import torch as pt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

import os, sys, json, shutil, csv
import numpy as np
from skimage import io, transform
from datetime import datetime

from utils.utils import *
from utils.mpi_utils import *
from utils.mlp import *
from utils.colmap_runner import colmapGenPoses

parser = argparse.ArgumentParser()

#training schedule
parser.add_argument('-epochs', type=int, default=4000, help='total epochs to train')
parser.add_argument('-steps', type=int, default=-1, help='total steps to train. In our paper, we proposed to use epoch instead.')
parser.add_argument('-tb_saveimage', type=int, default=50, help='write an output image to tensorboard for every <tb_saveimage> epochs')
parser.add_argument('-tb_savempi', type=int, default=200, help='generate MPI (WebGL) and measure PSNR/SSIM of validation image for every <tb_savempi> epochs')
parser.add_argument('-checkpoint', type=int, default=100, help='save checkpoint for every <checkpoint> epochs. Be aware that! It will replace the previous checkpoint.')
parser.add_argument('-tb_toc',type=int, default=500, help="print output to terminal for every tb_toc epochs")

#lr schedule
parser.add_argument('-lrc', type=float, default=10, help='the number of times of lr using for learning rate of explicit basis (k0).')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate of a multi-layer perceptron')
parser.add_argument('-decay_epoch', type=int, default=1333, help='the number of epochs for decay learning rate')
parser.add_argument('-decay_rate', type=float, default=0.1, help='ratio of decay rate at every <decay_epoch> epochs')

#network (First MLP)
parser.add_argument('-ray', type=int, default=8000, help='the number of sampled ray that is used to train in each step')
parser.add_argument('-hidden', type=int, default=384, help='the number of hidden node of the main MLP')
parser.add_argument('-mlp', type=int, default=4, help='the number of hidden layer of the main MLP')
parser.add_argument('-pos_level', type=int, default=10, help='the number of positional encoding in terms of image size. We recommend to set 2^(pos_level) > image_height and image_width')
parser.add_argument('-depth_level', type=int, default=8,help='the number of positional encoding in terms number of plane. We recommend to set 2^(depth_level) > layers * subplayers')
parser.add_argument('-lrelu_slope', type=float, default=0.01, help='slope of leaky relu')
parser.add_argument('-sigmoid_offset', type=float, default=5, help='sigmoid offset that is applied to alpha before sigmoid')

#basis (Second MLP)
parser.add_argument('-basis_hidden', type=int, default=64, help='the number of hidden node in the learned basis MLP')
parser.add_argument('-basis_mlp', type=int, default=1, help='the number of hidden layer in the learned basis MLP')
parser.add_argument('-basis_order', type=int, default=3, help='the number of  positional encoding in terms of viewing angle')
parser.add_argument('-basis_out', type=int, default=8, help='the number of coeffcient output (N in equation 3 under seftion 3.1)')

#loss
parser.add_argument('-gradloss', type=float, default=0.05, help='hyperparameter for grad loss')
parser.add_argument('-tvc', type=float, default=0.03, help='hyperparameter for total variation regularizer')

#training and eval data
parser.add_argument('-scene', type=str, default="", help='directory to the scene')
parser.add_argument('-ref_img', type=str, default="",  help='reference image, camera parameter of reference image is use to create MPI')
parser.add_argument('-dmin', type=float, default=-1, help='first plane depth')
parser.add_argument('-dmax', type=float, default=-1, help='last plane depth')
parser.add_argument('-invz', action='store_true', help='place MPI with inverse depth')
parser.add_argument('-scale', type=float, default=-1, help='scale the MPI size')
parser.add_argument('-llff_width', type=int, default=1008, help='if input dataset is LLFF it will resize the image to <llff_width>')
parser.add_argument('-deepview_width', type=int, default=800, help='if input dataset is deepview dataset, it will resize the image to <deepview_width>')
parser.add_argument('-train_ratio', type=float, default=0.875, help='ratio to split number of train/test (in case dataset doesn\'t specify how to split)')
parser.add_argument('-val_image_interval', '--val_image_interval', type=int, default=5, help='validation split interval: zero-based image ids where id mod interval equals 0 are validation')
parser.add_argument('-random_split', action='store_true', help='random split the train/test set. (in case dataset doesn\'t specify how to split)')
parser.add_argument('-num_workers', type=int, default=8, help='number of pytorch\'s dataloader worker')
parser.add_argument('-cv2resize', action='store_true', help='apply cv2.resize instead of skimage.transform.resize to match the score in our paper (see note in github readme for more detail) ')

#MPI
parser.add_argument('-offset', type=int, default=200, help='the offset (padding) of the MPI.')
parser.add_argument('-layers', type=int, default=16, help='the number of plane that stores base color')
parser.add_argument('-sublayers', type=int, default=12, help='the number of plane that share the same texture. (please refer to coefficient sharing under section 3.4 in the paper)')

#predict
parser.add_argument('-no_eval', action='store_true', help='do not measurement the score (PSNR/SSIM/LPIPS) ')
parser.add_argument('-no_csv', action='store_true', help="do not write CSV on evaluation")
parser.add_argument('-no_video', action='store_true', help="do not write the video on prediction")
parser.add_argument('-no_webgl', action='store_true', help='do not predict webgl (realtime demo) related content.')
parser.add_argument('-predict', action='store_true', help='predict validation images')
parser.add_argument('-eval_path', type=str, default='runs/evaluation/', help='path to save validation image')
parser.add_argument('-web_path', type=str, default='runs/html/', help='path to output real time demo')
parser.add_argument('-web_width', type=int, default=16000, help='max texture size (pixel) of realtime demo. WebGL on Highend PC is support up to 16384px, while mobile phone support only 4096px')
parser.add_argument('-http', action='store_true', help='serve real-time demo on http server')
parser.add_argument('-render_viewing', action='store_true', help='genereate view-dependent-effect video')
parser.add_argument('-render_nearest', action='store_true', help='genereate nearest input video')
parser.add_argument('-render_depth', action='store_true', help='generate depth')

# render path
parser.add_argument('-nice_llff', action='store_true', help="generate video that its rendering path matches real-forward facing dataset")
parser.add_argument('-nice_shiny', action='store_true', help="generate video that its rendering path matches shiny dataset")


#training utility
parser.add_argument('-model_dir', type=str, default="scene", help='model (scene) directory stored in results/<model_dir>/')
parser.add_argument('-pretrained', type=str, default="", help='location of checkpoint file, if not provide will use model_dir instead')
parser.add_argument('-restart', action='store_true', help='delete old weight and retrain')
parser.add_argument('-clean', action='store_true', help='delete old weight without start training process')


#miscellaneous
parser.add_argument('-all_gpu',action='store_true',help="In multiple GPU training, We don't train MLP (data parallel) on the first GPU. This make training slower but we can utilize more VRAM on other GPU.")

args = parser.parse_args()

# =============================================================================
# STANDARD EXPERIMENT LOGGING HELPERS
# Logging-only patch. NeX model/training/rendering core is untouched.
# =============================================================================

METRICS_FIELDS = [
  "timestamp",
  "epoch",
  "phase",
  "train_loss_total",
  "train_mse",
  "train_psnr",
  "lr",
  "val_mean_mse",
  "val_mean_psnr",
  "val_mean_ssim",
  "val_mean_lpips",
  "is_best",
]

class TeeStream:
  def __init__(self, *streams):
    self.streams = streams

  def write(self, data):
    for stream in self.streams:
      stream.write(data)
      stream.flush()

  def flush(self):
    for stream in self.streams:
      stream.flush()

def nowIso():
  return datetime.now().isoformat(timespec="seconds")

def jsonDefault(obj):
  if isinstance(obj, np.ndarray):
    return obj.tolist()
  if isinstance(obj, np.integer):
    return int(obj)
  if isinstance(obj, np.floating):
    return float(obj)
  if pt.is_tensor(obj):
    if obj.numel() == 1:
      return float(obj.detach().cpu().item())
    return obj.detach().cpu().tolist()
  return str(obj)

def tensorToFloat(x):
  if x == "":
    return ""
  if pt.is_tensor(x):
    return float(x.detach().cpu().item())
  if isinstance(x, np.generic):
    return float(x)
  return float(x)

def psnrFromMse(mse):
  mse = tensorToFloat(mse)
  return float(-10.0 * np.log10(max(mse, 1e-12)))

def getGitCommit():
  try:
    return os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip()
  except Exception:
    return ""

def expPaths(runpath):
  exp_dir = os.path.join(runpath, args.model_dir)
  return {
    "exp_dir": exp_dir,
    "checkpoints_dir": os.path.join(exp_dir, "checkpoints"),
    "best_dir": os.path.join(exp_dir, "checkpoints", "best"),
    "periodic_dir": os.path.join(exp_dir, "checkpoints", "periodic"),
    "images_dir": os.path.join(exp_dir, "images"),
    "logs_dir": os.path.join(exp_dir, "logs"),
    "tensorboard_dir": os.path.join(exp_dir, "logs", "tensorboard"),
    "config_path": os.path.join(exp_dir, "config.json"),
    "run_info_path": os.path.join(exp_dir, "run_info.txt"),
    "metrics_path": os.path.join(exp_dir, "metrics.csv"),
    "summary_path": os.path.join(exp_dir, "logs", "summary.json"),
    "terminal_log_path": os.path.join(exp_dir, "logs", "terminal_output.log"),
    "last_ckpt": os.path.join(exp_dir, "checkpoints", "last.pt"),
    "best_ckpt": os.path.join(exp_dir, "checkpoints", "best", "best.pt"),
  }

def ensureExperimentDirs(runpath):
  paths = expPaths(runpath)
  os.makedirs(paths["exp_dir"], exist_ok=True)
  os.makedirs(paths["checkpoints_dir"], exist_ok=True)
  os.makedirs(paths["best_dir"], exist_ok=True)
  os.makedirs(paths["periodic_dir"], exist_ok=True)
  os.makedirs(paths["images_dir"], exist_ok=True)
  os.makedirs(paths["logs_dir"], exist_ok=True)
  os.makedirs(paths["tensorboard_dir"], exist_ok=True)
  return paths

def setupTerminalLogging(paths):
  if getattr(sys, "_nex_pipeline_tee_started", False):
    return
  log_f = open(paths["terminal_log_path"], "a", buffering=1)
  sys.stdout = TeeStream(sys.__stdout__, log_f)
  sys.stderr = TeeStream(sys.__stderr__, log_f)
  sys._nex_pipeline_tee_started = True

def readSplitInfo(dpath):
  split_path = os.path.join(dpath, "split_info.json")
  if os.path.exists(split_path):
    with open(split_path, "r") as f:
      return json.load(f)
  return {}

def appendMetricsRow(runpath, row):
  paths = ensureExperimentDirs(runpath)
  out = {k: row.get(k, "") for k in METRICS_FIELDS}
  write_header = not os.path.exists(paths["metrics_path"])
  with open(paths["metrics_path"], "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
    if write_header:
      writer.writeheader()
    writer.writerow(out)

def updateSummary(runpath, extra):
  paths = ensureExperimentDirs(runpath)
  summary = {}
  if os.path.exists(paths["summary_path"]):
    try:
      with open(paths["summary_path"], "r") as f:
        summary = json.load(f)
    except Exception:
      summary = {}

  base = {
    "experiment": args.model_dir,
    "model_name": "NeX",
    "branch": "nex_baseline",
    "scene": os.path.basename(args.scene.rstrip("/")),
    "scene_path": args.scene,
    "val_image_interval": int(args.val_image_interval),
    "git_commit": getGitCommit(),
    "updated_at": nowIso(),
  }
  summary.update(base)
  summary.update(extra)

  with open(paths["summary_path"], "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True, default=jsonDefault)

def writeConfigAndRunInfo(runpath, dataset, sampler_train, sampler_val, start_epoch):
  paths = ensureExperimentDirs(runpath)
  split_info = readSplitInfo(args.scene)

  config = {
    "experiment": args.model_dir,
    "model_name": "NeX",
    "branch": "nex_baseline",
    "created_at": nowIso(),
    "git_commit": getGitCommit(),
    "user": getpass.getuser(),
    "host": os.popen("hostname 2>/dev/null").read().strip(),
    "command": " ".join(sys.argv),
    "scene": os.path.basename(args.scene.rstrip("/")),
    "scene_path": args.scene,
    "args": vars(args),
    "start_epoch": int(start_epoch),
    "n_train_images": int(len(sampler_train)),
    "n_val_images": int(len(sampler_val)),
    "split_info": split_info,
    "no_validation_leakage": True,
    "logging_style": "pipeline_standard",
    "output_structure": {
      "checkpoints": "checkpoints/",
      "best_checkpoint": "checkpoints/best/best.pt",
      "last_checkpoint": "checkpoints/last.pt",
      "periodic_checkpoints": "checkpoints/periodic/epoch_XXXX.pt",
      "config": "config.json",
      "images": "images/epoch_XXXX/",
      "logs": "logs/",
      "summary": "logs/summary.json",
      "terminal_output": "logs/terminal_output.log",
      "metrics": "metrics.csv",
      "val_per_image_metrics": "val_per_image_metrics.csv",
    },
  }

  with open(paths["config_path"], "w") as f:
    json.dump(config, f, indent=2, sort_keys=True, default=jsonDefault)

  with open(paths["run_info_path"], "w") as f:
    f.write("experiment: {}\n".format(args.model_dir))
    f.write("model_name: NeX\n")
    f.write("branch: nex_baseline\n")
    f.write("created_at: {}\n".format(config["created_at"]))
    f.write("git_commit: {}\n".format(config["git_commit"]))
    f.write("user: {}\n".format(config["user"]))
    f.write("host: {}\n".format(config["host"]))
    f.write("scene: {}\n".format(args.scene))
    f.write("command: {}\n".format(config["command"]))
    f.write("val_image_interval: {}\n".format(args.val_image_interval))
    f.write("n_train_images: {}\n".format(config["n_train_images"]))
    f.write("n_val_images: {}\n".format(config["n_val_images"]))
    f.write("val_names: {}\n".format(", ".join(split_info.get("val_names", []))))

  updateSummary(runpath, {
    "status": "initialized",
    "pipeline_complete": False,
    "n_train_images": config["n_train_images"],
    "n_val_images": config["n_val_images"],
    "val_names": split_info.get("val_names", []),
    "best_epoch": None,
    "best_val_mean_psnr": None,
    "best_val_mean_ssim": None,
    "best_val_mean_lpips": None,
  })

def recordValidation(runpath, model, optimizer, epoch, val_metrics, best_state):
  if val_metrics is None:
    return best_state

  paths = ensureExperimentDirs(runpath)

  val_mse = val_metrics.get("MSE", "")
  val_psnr = val_metrics.get("PSNR", "")
  val_ssim = val_metrics.get("SSIM", "")
  val_lpips = val_metrics.get("LPIPS", "")

  is_best = False
  if val_psnr != "":
    val_psnr_f = tensorToFloat(val_psnr)
    if val_psnr_f > best_state["best_val_mean_psnr"]:
      is_best = True
      best_state["best_val_mean_psnr"] = val_psnr_f
      best_state["best_epoch"] = int(epoch)
      checkpoint(paths["best_ckpt"], model, optimizer, int(epoch))

  appendMetricsRow(runpath, {
    "timestamp": nowIso(),
    "epoch": int(epoch) if isinstance(epoch, int) else epoch,
    "phase": "val",
    "val_mean_mse": tensorToFloat(val_mse) if val_mse != "" else "",
    "val_mean_psnr": tensorToFloat(val_psnr) if val_psnr != "" else "",
    "val_mean_ssim": tensorToFloat(val_ssim) if val_ssim != "" else "",
    "val_mean_lpips": tensorToFloat(val_lpips) if val_lpips != "" else "",
    "is_best": int(is_best),
  })

  updateSummary(runpath, {
    "status": "running",
    "latest_eval_epoch": int(epoch) if isinstance(epoch, int) else epoch,
    "latest_val_mean_mse": tensorToFloat(val_mse) if val_mse != "" else "",
    "latest_val_mean_psnr": tensorToFloat(val_psnr) if val_psnr != "" else "",
    "latest_val_mean_ssim": tensorToFloat(val_ssim) if val_ssim != "" else "",
    "latest_val_mean_lpips": tensorToFloat(val_lpips) if val_lpips != "" else "",
    "best_epoch": best_state["best_epoch"],
    "best_val_mean_psnr": best_state["best_val_mean_psnr"],
    "best_val_mean_ssim": tensorToFloat(val_ssim) if is_best and val_ssim != "" else None,
    "best_val_mean_lpips": tensorToFloat(val_lpips) if is_best and val_lpips != "" else None,
  })

  return best_state

def computeHomographies(sfm, feature, planes):
  fx = feature['fx'][0]
  fy = feature['fy'][0]
  px = feature['px'][0]
  py = feature['py'][0]

  new_r = feature['r'][0] @ sfm.ref_rT
  new_t = (-new_r @ sfm.ref_t) + feature['t'][0]

  n = pt.tensor([[0.0, 0.0, 1.0]])
  Ha = new_r.t()
  Hb = Ha @ new_t @ n @ Ha
  Hc = (n @ Ha @ new_t)[0]

  ki = pt.tensor([[fx, 0, px],
                  [0, fy, py],
                  [0, 0, 1]], dtype=pt.float).inverse()

  tt = sfm.ref_cam
  ref_k = pt.tensor( [[tt['fx'], 0, tt['px']],
                      [0, tt['fy'], tt['py']],
                      [0,        0,       1]])

  planes_mat = pt.Tensor(planes).view(-1, 1, 1)
  return (ref_k @ (Ha + Hb / (-planes_mat - Hc))) @ ki

def computeHomoWarp(sfm, input_shape, input_offset,
                    output_shape, selection,
                    feature, planes, inv=False, inv_offset = False):

  selection = selection.cuda()
  # coords: (sel, 3)
  coords = pt.stack([selection % output_shape[1], selection // output_shape[1],
                    pt.ones_like(selection)], -1).float()

  # Hs: (n, 3, 3)

  Hs = computeHomographies(sfm, feature, planes)
  if inv: Hs = Hs.inverse()
  if inv_offset:
    coords[:, :2] += input_offset
  prod = coords @ pt.transpose(Hs, 1, 2).cuda()
  scale = pt.tensor([input_shape[1] - 1, input_shape[0] - 1]).cuda()

  ref_coords = prod[:, :, :2] / prod[:, :, 2:]
  if not inv_offset:
    warp = ((ref_coords + input_offset) / scale.view(1, 1, 2)) * 2 - 1
  else:
    warp = ((ref_coords) / scale.view(1, 1, 2)) * 2 - 1
  warp = warp[:, :, None]


  return warp, ref_coords

def totalVariation(images):
  pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
  pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
  sum_axis = [1, 2, 3]

  tot_var = (
      pt.sum(pt.abs(pixel_dif1), dim=sum_axis) +
      pt.sum(pt.abs(pixel_dif2), dim=sum_axis))

  return tot_var / (images.shape[2]-1) / (images.shape[3]-1)

def cumprod_exclusive(x):
  cp = pt.cumprod(x, 0)
  cp = pt.roll(cp, 1, 0)
  cp[0] = 1.0
  return cp

def getWarp3d(warp, interpolate = False):
  if not interpolate:
    depths = pt.repeat_interleave(pt.linspace(-1, 1, args.layers), args.sublayers).view(1, -1, 1, 1, 1).cuda()
  else:
    depths = pt.linspace(-1, 1, args.layers * args.sublayers).view(1, -1, 1, 1, 1).cuda()
  warp3d = warp[None] # 1, n, sel, 1, 2
  warp3d = pt.cat([warp3d, pt.ones_like(warp3d[:, :, :, :, :1]) * depths], -1)
  return warp3d

def normalized(v, dim):
  return v / (pt.norm(v, dim=dim, keepdim=True) + 1e-7)

class Basis(nn.Module):
  def __init__(self, shape, out_view):
    super().__init__()
    #choosing illumination model
    self.order = args.basis_order

    # network for learn basis
    self.seq_basis = nn.DataParallel(
      ReluMLP(
        args.basis_mlp, #basis_mlp
        args.basis_hidden, #basis_hidden
        self.order * 4,
        args.lrelu_slope,
        out_node = args.basis_out, #basis_out
      )
    )
    print('Basis Network:',self.seq_basis)

    # positional encoding pre compute
    self.pos_freq_viewing = pt.Tensor([(2 ** i) for i in range(self.order)]).view(1, 1, 1, 1, -1).cuda()

  def forward(self, sfm, feature, ref_coords, warp, planes, coeff = None):
    vi, xy = get_viewing_angle(sfm, feature, ref_coords, planes)
    n, sel = vi.shape[:2]

    # positional encoding for learn basis
    hinv_xy = vi[...,  :2, None] * self.pos_freq_viewing
    big = pt.reshape(hinv_xy, [n, sel, 1, hinv_xy.shape[-2] * hinv_xy.shape[-1]])
    vi = pt.cat([pt.sin(0.5*np.pi*big), pt.cos(0.5*np.pi*big)], -1)

    out2 = self.seq_basis(vi)
    out2 = pt.tanh(out2)

    vi = out2.view(n, sel, 1, 1, -1)

    coeff = coeff.view(coeff.shape[0], coeff.shape[1], coeff.shape[2], 3,  -1)
    coeff = pt.tanh(coeff)

    illumination = pt.sum(coeff * vi,-1).permute([0, 3, 1, 2])

    return illumination

def get_viewing_angle(sfm, feature, ref_coords, planes):
  camera = sfm.ref_rT.t() @ feature["center"][0] + sfm.ref_t

  # (n, rays, 2) -> (n, 2, rays)
  coords = ref_coords.permute([0, 2, 1])
  # (n, 2, rays) -> (n, 3, rays)
  coords = pt.cat([coords, pt.ones_like(coords[:, :1])], 1)

  # coords: (n, 3, rays)
  # viewed planes: (n, 1, 1)
  # xyz: (n, 3, rays)
  xyz = coords * pt.Tensor(planes).view(-1, 1, 1).cuda()

  ki = pt.tensor([[feature['fx'][0], 0, feature['px'][0]],
                  [0, feature['fy'][0], feature['py'][0]],
                  [0, 0, 1]], dtype=pt.float).inverse().cuda()

  xyz = ki @ xyz

  # camera: (3, 1) -> (1, 3, 1)
  # xyz: (n, 3, rays)
  # viewing_angle: (n, 3, rays)
  # viewing_angle = normalized(camera[None].cuda() - xyz, 1)
  inv_viewing_angle = normalized(xyz - camera[None].cuda(), 1)

  view = inv_viewing_angle.permute([0, 2, 1])
  xyz = xyz.permute([0, 2, 1])
  return view[:,:,None], xyz[:,:,None]

class Network(nn.Module):
  def __init__(self, shape, sfm):
    super(Network, self).__init__()
    self.shape = [shape[2], shape[3]]
    total_cuda = pt.cuda.device_count()
    mlp_first_device = 1 if (not args.all_gpu) and total_cuda > 1 else 0
    mlp_devices = list(range(mlp_first_device, total_cuda))
    #mpi_c (k0) as an explicit
    mpi_c = pt.empty((shape[0], 3, shape[2], shape[3]), device='cuda:0').uniform_(-1, 1)
    self.mpi_c = nn.Parameter(mpi_c)
    self.specular = Basis(shape, args.basis_out * 3).cuda()
    self.seq1 = nn.DataParallel(
      VanillaMLP(
        args.mlp,
        args.hidden,
        args.pos_level,
        args.depth_level,
        args.lrelu_slope,
        out_node = 1 + args.basis_out * 3,
        first_gpu = mlp_first_device
      ),
      device_ids = mlp_devices
    )

    self.seq1 = self.seq1.cuda("cuda:{}".format(mlp_first_device))
    self.pos_freq = pt.Tensor([0.5 * np.pi * (2 ** i) for i in range(args.pos_level)] * 2).view(1, 1, 1, 2, -1).cuda()
    self.depth_freq = pt.Tensor([0.5 * np.pi * (2 ** i) for i in range(args.depth_level)]).view(1, 1, 1, -1).cuda()

    self.z_coords = pt.linspace(-1, 1, args.layers * args.sublayers).view(-1, 1, 1, 1).cuda()
    if args.render_depth:
      self.rainbow_mpi = np.zeros((shape[0], 3, shape[2], shape[3]), dtype=np.float32)
      for i,s in enumerate(np.linspace(1, 0, shape[0])):
        color = Rainbow(s)
        for c in range(3):
          self.rainbow_mpi[i,c] = color[c]
      self.rainbow_mpi = pt.from_numpy(self.rainbow_mpi).to('cuda:0')
    else:
      self.rainbow_mpi = None

    if sfm.dmin < 0 or sfm.dmax < 0:
      raise ValueError("invalid dmin dmax")

    self.planes = getPlanes(sfm, args.layers * args.sublayers)
    print('Mpi Size: {}'.format(self.mpi_c.shape))
    print('All combined layers: {}'.format(args.layers * args.sublayers))
    print(self.planes)
    print('Using inverse depth: {}, Min depth: {}, Max depth: {}'.format(sfm.invz == 1, self.planes[0],self.planes[-1]))
    print('Layer of MLP: {}'.format(args.mlp + 2))
    print('Hidden Channel of MLP: {}'.format(args.hidden))
    print('Main Network',self.seq1)


  def forward(self, sfm, feature, output_shape, selection):
    ''' Rendering
    Args:
      sfm: reference camera parameter
      feature: target camera parameter
      output_shape: [h, w]. Desired rendered image
      selection: [ray]. pixel to train
    Returns:
      output: [1, 3, rays, 1] rendered image
    '''
    # (n, sel, 1, 2), (n, sel, 1, 1), (n, sel, 2)
    warp, ref_coords = computeHomoWarp(sfm,
                                 self.shape,
                                 sfm.offset,
                                 output_shape, selection,
                                 feature, self.planes)

    n = warp.shape[0]
    sel = warp.shape[1]
    # vxy: (n, sel, 1, 2, pos_level)
    vxy = warp[:, :, :, :, None] * self.pos_freq
    vxy = vxy.view(n, sel, 1, -1) # (n, sel, 1, pos_level*2)

    # vz: (n, sel, 1, depth_level)
    vz = pt.ones_like(warp[:, :, :, :1]) * self.z_coords * self.depth_freq

    vxyz = pt.cat([vxy, vz], -1)
    bigcoords = pt.cat([pt.sin(vxyz), pt.cos(vxyz)], -1)
    # (n, sel, 1, out_node)
    out = self.seq1(bigcoords).cuda()

    node = 0
    self.mpi_a = out[..., node:node + 1]
    node += 1
    # n, 1, sel, 1
    self.mpi_a = self.mpi_a.view(self.mpi_a.shape[0], 1, self.mpi_a.shape[1], self.mpi_a.shape[2])
    mpi_a_sig  = pt.sigmoid(self.mpi_a - args.sigmoid_offset)

    if args.render_depth:
      # generate Rainbow MPI instead of real mpi to visualize the depth
      # self.rainbow_mpi: n, 3, h, w   warp: (n, sel, 1, 2)
      # Need: N, C, Din, Hin, Win;  N, Dout, Hout, Wout, 3
      rainbow_3d = self.rainbow_mpi.permute([1, 0, 2, 3])[None]
      warp3d = getWarp3d(warp)
      #samples: N, C, Dout, Hout, Wout
      samples = F.grid_sample(rainbow_3d, warp3d, align_corners=True)
      # (layers, out_node, rays, 1)
      rgb = samples[0].permute([1, 0, 2, 3])
    else:
      mpi_sig = pt.sigmoid(self.mpi_c)
      # mpi_sig: n, 3, h, w   warp: (n, sel, 1, 2)
      # Need: N, C, Din, Hin, Win;    N, Dout, Hout, Wout, 3
      mpi_sig3d = mpi_sig.permute([1, 0, 2, 3])[None]
      warp3d = getWarp3d(warp)
      #samples: N, C, Dout, Hout, Wout
      samples = F.grid_sample(mpi_sig3d, warp3d, align_corners=True)
      # (layers, out_node, rays, 1)
      rgb = samples[0].permute([1, 0, 2, 3])
      cof = out[::args.sublayers, ..., node:]
      cof = pt.repeat_interleave(cof, args.sublayers, 0)
      self.illumination = self.specular(sfm, feature, ref_coords, warp, self.planes, coeff = cof)
      # rgb: (layers, 3, rays, 1)
      rgb = pt.clamp(rgb + self.illumination, 0.0, 1.0)

    weight = cumprod_exclusive(1 - mpi_a_sig)

    output = pt.sum(weight * rgb * mpi_a_sig, dim=0, keepdim=True)

    return output

def getMPI(model, sfm, m = 1, dataloader = None):
  ''' convert from neural network to MPI planes
    Args:
      model: Neural net model
      sfm: reference camera parameter
      m: target camera parameter
      dataloader:
    Returns:
      output: dict({
        'mpi_c':'explicit coefficient k0',
        'mpi_a':'alpha transparentcy',
        'mpi_b':'basis',
        'mpi_v':'Kn coefficient'
      })
    '''
  sh = sfm.ref_cam['height'] + sfm.offset * 2
  sw = sfm.ref_cam['width'] + sfm.offset * 2
  #print((sh, sw))

  y, x = pt.meshgrid([
    (pt.arange(0, sh, dtype=pt.float)) / (sh-1) * 2 - 1,
    (pt.arange(0, sw, dtype=pt.float)) / (sw-1) * 2 - 1])

  coords = pt.cat([x[:,:,None].cuda(), y[:,:,None].cuda()], -1)
  model.eval()
  sh_v = 400
  sw_v = 400
  rangex, rangey =  0.7, 0.6
  y_v, x_v = pt.meshgrid([
    (pt.linspace(-rangey, rangey, sh_v)),
    (pt.linspace(-rangex, rangex , sw_v))])
  #viewing [sh_v, sh_w, 2]
  viewing = pt.cat([x_v[:,:,None].cuda(), y_v[:,:,None].cuda()], -1)
  #hinv_xy [1, sh_v, sh_w, 2 * pos_lev]
  hinv_xy = viewing.view(1, sh_v, sw_v, 2, 1) * model.specular.pos_freq_viewing
  hinv_xy = hinv_xy.view(1, sh_v, sw_v, -1)
  #pe_view [1, sh_v, sw_v, 2 * 2 * pos_lev]
  pe_view = pt.cat([pt.sin(0.5*np.pi *hinv_xy), pt.cos(0.5*np.pi *hinv_xy)], -1)
  #out2 [1, sh, sw, num_basis]
  out2 = model.specular.seq_basis(pe_view)
  #imgs_b [num_basis, 1, sh_v, sw_v]
  imgs_b = pt.tanh(out2.permute([3, 0, 1, 2])).cpu().detach()

  n = args.layers * args.sublayers
  imgs_c, imgs_a, imgs_v = [], [], []

  with pt.no_grad():
    for i in range(0, n, m):

      #coords [sh, sw, 2] --> [1, sh, sw, 2, 1]
      #vxy [1, sh, sw, 2, pos_lev] -->  [1, sh, sw, 2*pos_lev]
      vxy = coords.view(1, sh, sw, 2, 1) * model.pos_freq
      vxy = vxy.view(1, sh, sw, -1)
      #vz [1, sh, sw, depth_lev]
      vz = pt.ones_like(coords.view(1, sh, sw, -1)[..., :1]) * model.z_coords[i:i+1] * model.depth_freq
      #vxyz [1, sh, sw, 2*pos_lev + depth_lev]
      vxyz = pt.cat([vxy, vz], -1)
      bigcoords = pt.cat([pt.sin(vxyz), pt.cos(vxyz)], -1)
      if sfm.offset > 270:
        out =  [model.seq1(bigy) for bigy in [bigcoords[:, :int(sh/2)], bigcoords[:, int(sh/2):]]]
        out = pt.cat(out, 1)
      else:
        out = model.seq1(bigcoords)
      node = 0

      mpi_a = out[..., node:node + 1].cpu()
      node +=1
      mpi_a = mpi_a.view(mpi_a.shape[0], 1, mpi_a.shape[1], mpi_a.shape[2])
      imgs_a.append(pt.sigmoid(mpi_a[0] - args.sigmoid_offset))

      if i % args.sublayers == 0:
        out = out[..., node:].cpu()
        mpi_v = out.view(out.shape[0], out.shape[1], out.shape[2], 3, -1)
        mpi_v = mpi_v.permute([0, 3, 1, 2, 4])
        mpi_v = mpi_v[0]
        mpi_v =  pt.tanh(mpi_v)
        imgs_v.append(mpi_v)

    mpi_c_sig = pt.sigmoid(model.mpi_c)
    mpi_a_sig = pt.stack(imgs_a, 0)
    mpi_v_tanh = pt.stack(imgs_v, 0)
    info = {
      'mpi_c': mpi_c_sig.cpu(),
      'mpi_a': mpi_a_sig.cpu(),
      'mpi_v' : mpi_v_tanh.cpu(),
      'mpi_b':  imgs_b.cpu()
    }

  pt.cuda.empty_cache()
  return info

def generateAlpha(model, dataset, dataloader, writer, runpath, suffix="", dataloader_train = None):
  ''' Prediction
    Args.
      model.   --> trained model
      dataset. --> validation dataset
      writer.  --> tensorboard
  '''
  suffix_str = "/%06d" % suffix if isinstance(suffix, int) else "/"+str(suffix)
  epoch_label = "epoch_%04d" % suffix if isinstance(suffix, int) else str(suffix if suffix != "" else "final")
  eval_epoch = suffix if isinstance(suffix, int) else epoch_label

  # create webgl only when using -predict or finish training
  if not args.no_webgl and suffix =="":
    info = getMPI(model, dataset.sfm, dataloader = dataloader_train)

    outputCompactMPI(info,
                   dataset.sfm,
                   model.planes,
                   runpath + args.model_dir + suffix_str,
                   args.layers,
                   args.sublayers,
                   dataset.sfm.offset,
                   args.invz,
                   webpath=args.web_path,
                   web_width= args.web_width)

  out = None
  if not args.no_eval and len(dataloader) > 0:
    out = evaluation(model,
                     dataset,
                     dataloader,
                     args.ray,
                     os.path.join(runpath, args.model_dir),
                     webpath=runpath,
                     write_csv = not args.no_csv,
                     epoch = eval_epoch,
                     epoch_label = epoch_label,
                     split_info = readSplitInfo(args.scene))
    if writer is not None and isinstance(suffix, int):
      for metrics, score in out.items():
        writer.add_scalar('METRICS/{}'.format(metrics), score, suffix)
  return out


def setLearningRate(optimizer, epoch):
  ds = int(epoch / args.decay_epoch)
  lr = args.lr * (args.decay_rate ** ds)

  optimizer.param_groups[0]['lr'] = lr
  if args.lrc > 0:
    optimizer.param_groups[1]['lr'] = lr * args.lrc

def train():
  pt.manual_seed(1)
  np.random.seed(1)

  runpath = "results/"

  if args.restart or args.clean:
    shutil.rmtree(os.path.join(runpath, args.model_dir), ignore_errors=True)
  if args.clean:
    exit()

  paths = ensureExperimentDirs(runpath)
  setupTerminalLogging(paths)

  print("================================================================================")
  print("NeX PIPELINE-STANDARD RUN")
  print("================================================================================")
  print("experiment:", args.model_dir)
  print("scene:", args.scene)
  print("output:", paths["exp_dir"])
  print("command:", " ".join(sys.argv))
  print("================================================================================")

  dpath = args.scene

  writeIntervalSplitFiles(dpath, args.val_image_interval)

  dataset = loadDataset(dpath)
  sampler_train, sampler_val, dataloader_train, dataloader_val = prepareDataloaders(
    dataset,
    dpath,
    random_split = args.random_split,
    train_ratio = args.train_ratio,
    num_workers = args.num_workers
  )

  mpi_h = int(dataset.sfm.ref_cam['height'] + dataset.sfm.offset * 2)
  mpi_w = int(dataset.sfm.ref_cam['width'] + dataset.sfm.offset * 2)
  model = Network((args.layers,
                 4,
                 mpi_h,
                 mpi_w,
                 ), dataset.sfm)

  if args.lrc > 0:
    #mlp use lower lr, while mpi_c, light dir intensity get higher lr
    my_list = [name for name, params in model.named_parameters() if 'seq1' in name]
    mlp_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    other_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
    optimizer = pt.optim.Adam([
      {'params': mlp_params, 'lr': 0},
      {'params': other_params, 'lr': 0}])
  else:
    optimizer = pt.optim.Adam(model.parameters(), lr=0)

  start_epoch = 0
  paths = ensureExperimentDirs(runpath)
  ckpt = paths["last_ckpt"]
  if os.path.exists(ckpt):
    start_epoch = loadFromCheckpoint(ckpt, model, optimizer)
  elif args.pretrained != "":
    pretrained_ckpt = os.path.join(runpath, args.pretrained, "checkpoints", "last.pt")
    if not os.path.exists(pretrained_ckpt):
      pretrained_ckpt = os.path.join("runs", args.pretrained, "ckpt.pt")
    start_epoch = loadFromCheckpoint(pretrained_ckpt, model, optimizer)

  step = start_epoch * len(sampler_train)

  if args.epochs < 0 and args.steps < 0:
    raise Exception("Need to specify epochs or steps")

  if args.epochs < 0:
    args.epochs = int(np.ceil(args.steps / len(sampler_train)))

  if args.predict:
    generateAlpha(model, dataset, dataloader_val, None, runpath, dataloader_train = dataloader_train)
    if not args.no_video:
      if args.render_nearest:
        vid_path = 'video_nearest'
        render_type = 'nearest'
      elif args.render_viewing:
        vid_path = 'viewing_output'
        render_type = 'viewing'
      elif args.render_depth:
        vid_path = 'video_depth'
        render_type = 'depth'
      else:
        vid_path = 'video_output'
        render_type = 'default'
      pt.cuda.empty_cache()
      render_video(model, dataset, args.ray, os.path.join(runpath, vid_path, args.model_dir),
                  render_type = render_type, dataloader = dataloader_train)
    if args.http:
      serve_files(args.model_dir, args.web_path)
    exit()


  backupConfigAndCode(runpath, dataset, sampler_train, sampler_val, start_epoch)
  ts = TrainingStatus(num_steps=args.epochs * len(sampler_train))
  writer = SummaryWriter(paths["tensorboard_dir"])
  writer.add_text('command',' '.join(sys.argv), 0)
  best_state = {"best_val_mean_psnr": -1e30, "best_epoch": None}
  updateSummary(runpath, {
    "status": "running",
    "pipeline_complete": False,
    "start_epoch": int(start_epoch),
    "total_epochs": int(args.epochs),
  })
  ts.tic()

   # shift by 1 epoch to save last epoch to tensorboard
  for epoch in range(start_epoch, args.epochs+1):
    epoch_loss_total = 0
    epoch_mse = 0

    model.train()

    for i, feature in enumerate(dataloader_train):
      #print("step: {}".format(i))
      setLearningRate(optimizer, epoch)
      optimizer.zero_grad()

      output_shape = feature['image'].shape[-2:]

      #sample L-shaped rays
      sel = Lsel(output_shape, args.ray)


      gt = feature['image']
      gt = gt.view(gt.shape[0], gt.shape[1], gt.shape[2] * gt.shape[3])
      gt = gt[:, :, sel, None].cuda()
      output = model(dataset.sfm, feature, output_shape, sel)

      mse = pt.mean((output - gt) ** 2)

      loss_total = mse

      #tvc regularizer
      tvc = args.tvc * pt.mean(totalVariation(pt.sigmoid(model.mpi_c[:, :3])))
      loss_total = loss_total + tvc

      # grad loss
      ox = output[:, :, 1::3,  :] - output[:, :, 0::3, :]
      oy = output[:, :, 2::3,  :] - output[:, :, 0::3, :]
      gx = gt[:, :, 1::3,  :] - gt[:, :, 0::3, :]
      gy = gt[:, :, 2::3, :] - gt[:, :, 0::3, :]
      loss_total = loss_total + args.gradloss * (pt.mean(pt.abs(ox - gx)) + pt.mean(pt.abs(oy - gy)))

      epoch_loss_total += loss_total
      epoch_mse += mse

      loss_total.backward()
      optimizer.step()

      step += 1
      toc_msg = ts.toc(step, loss_total.item())
      if step % args.tb_toc == 0:  print(toc_msg)
      ts.tic()

    avg_loss_total = epoch_loss_total / len(sampler_train)
    avg_mse = epoch_mse / len(sampler_train)
    train_psnr = psnrFromMse(avg_mse)

    writer.add_scalar('loss/total', avg_loss_total, epoch)
    writer.add_scalar('loss/mse', avg_mse, epoch)
    writer.add_scalar('loss/train_psnr', train_psnr, epoch)

    appendMetricsRow(runpath, {
      "timestamp": nowIso(),
      "epoch": int(epoch),
      "phase": "train",
      "train_loss_total": tensorToFloat(avg_loss_total),
      "train_mse": tensorToFloat(avg_mse),
      "train_psnr": train_psnr,
      "lr": float(optimizer.param_groups[0]["lr"]),
    })

    updateSummary(runpath, {
      "status": "running",
      "latest_epoch": int(epoch),
      "latest_step": int(step),
      "latest_train_loss_total": tensorToFloat(avg_loss_total),
      "latest_train_mse": tensorToFloat(avg_mse),
      "latest_train_psnr": train_psnr,
    })

    if args.tb_saveimage > 0 and epoch % args.tb_saveimage == 0:
      with pt.no_grad():
        render = patch_render(model, dataset.sfm, feature, args.ray)
        Spec = getMPI(model, dataset.sfm, m = args.sublayers, dataloader = None)

        writer.add_image('images/render', pt.cat([feature['image'].cuda(), render], 2)[0], epoch)
        writer.add_image('images/2_mpia', make_grid(F.interpolate(Spec['mpi_a'],
          (int(mpi_h * 0.3), int(mpi_w * 0.3)),
          mode='area'), 4), epoch)
        writer.add_image('images/2_mpic', make_grid(F.interpolate(Spec['mpi_c'],
          (int(mpi_h * 0.3), int(mpi_w * 0.3)),
          mode='area'), 4), epoch)
        pt.cuda.empty_cache()

    if epoch % args.tb_savempi == 0 and args.tb_savempi > 0 and epoch > 0:
      val_metrics = generateAlpha(model, dataset, dataloader_val, writer, runpath, epoch)
      best_state = recordValidation(runpath, model, optimizer, epoch, val_metrics, best_state)
      pt.cuda.empty_cache()

    var = pt.mean(pt.std(model.illumination, 2)** 2)
    mean = pt.mean(model.illumination)
    writer.add_scalar('loss/illumination_mean', mean, epoch)
    writer.add_scalar('loss/illumination_var', var, epoch)

    if (epoch+1) % args.checkpoint == 0 or epoch == args.epochs-1:
      if np.isnan(loss_total.item()):
        exit()
      save_epoch = epoch + 1
      checkpoint(paths["last_ckpt"], model, optimizer, save_epoch)
      periodic_path = os.path.join(paths["periodic_dir"], "epoch_%04d.pt" % save_epoch)
      checkpoint(periodic_path, model, optimizer, save_epoch)

  print('Finished Training')

  if not (args.tb_savempi > 0 and args.epochs % args.tb_savempi == 0):
    val_metrics = generateAlpha(model, dataset, dataloader_val, None, runpath, args.epochs, dataloader_train = dataloader_train)
    best_state = recordValidation(runpath, model, optimizer, args.epochs, val_metrics, best_state)

  checkpoint(paths["last_ckpt"], model, optimizer, args.epochs)

  updateSummary(runpath, {
    "status": "finished",
    "pipeline_complete": True,
    "finished_at": nowIso(),
    "final_epoch": int(args.epochs),
    "best_epoch": best_state["best_epoch"],
    "best_val_mean_psnr": best_state["best_val_mean_psnr"] if best_state["best_epoch"] is not None else None,
  })

  writer.close()

  if not args.no_video:
    render_video(model, dataset, args.ray, os.path.join(runpath, args.model_dir, 'video_output'))
  if args.http:
    serve_files(args.model_dir, args.web_path)

def checkpoint(file, model, optimizer, epoch):
  os.makedirs(os.path.dirname(file), exist_ok=True)
  print("Checkpointing Model @ Epoch %d -> %s ..." % (epoch, file))
  pt.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, file)

def loadFromCheckpoint(file, model, optimizer):
  checkpoint = pt.load(file)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  start_epoch = checkpoint['epoch']
  print("Loading %s Model @ Epoch %d" % (args.pretrained, start_epoch))
  return start_epoch

def backupConfigAndCode(runpath, dataset=None, sampler_train=None, sampler_val=None, start_epoch=0):
  if args.predict or args.clean:
    return
  writeConfigAndRunInfo(runpath, dataset, sampler_train, sampler_val, start_epoch)


def writeIntervalSplitFiles(dpath, val_image_interval):
  interval = int(val_image_interval)
  if interval <= 1:
    raise ValueError("val_image_interval must be >= 2")

  img_dir = os.path.join(dpath, "images")
  if not os.path.isdir(img_dir):
    return

  imgs = sorted([
    f for f in os.listdir(img_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
  ])

  if len(imgs) == 0:
    return

  ids = list(range(len(imgs)))
  val_ids = [i for i in ids if i % interval == 0]
  train_ids = [i for i in ids if i % interval != 0]

  if len(train_ids) == 0:
    raise ValueError("Empty train split. Increase val_image_interval.")
  if len(val_ids) == 0:
    raise ValueError("Empty validation split.")

  train_names = [imgs[i] for i in train_ids]
  val_names = [imgs[i] for i in val_ids]

  with open(os.path.join(dpath, "train_image.txt"), "w") as f:
    f.write("\n".join(train_names) + "\n")

  with open(os.path.join(dpath, "val_image.txt"), "w") as f:
    f.write("\n".join(val_names) + "\n")

  split_info = {
    "val_image_interval": interval,
    "val_split_rule": "start_from_first_image_zero_based_ids_mod_interval_eq_0",
    "n_images": len(imgs),
    "train_ids_0based": train_ids,
    "val_ids_0based": val_ids,
    "train_names": train_names,
    "val_names": val_names,
  }

  with open(os.path.join(dpath, "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)

  print("[SPLIT] interval={} total={} train={} val={}".format(
    interval, len(imgs), len(train_ids), len(val_ids)
  ))
  print("[SPLIT] val_names={}".format(", ".join(val_names)))


def loadDataset(dpath):
  # if dataset directory has only image, create LLFF poses
  colmapGenPoses(dpath)

  if args.scale == -1:
    args.scale = getDatasetScale(dpath, args.deepview_width, args.llff_width)

  if is_deepview(dpath) and args.ref_img == '':
    with open(dpath + "/ref_image.txt", "r") as fi:
      args.ref_img = str(fi.readline().strip())
  render_style = 'llff' if args.nice_llff else 'shiny' if args.nice_shiny else ''
  return OrbiterDataset(dpath, ref_img=args.ref_img, scale=args.scale,
                           dmin=args.dmin,
                           dmax=args.dmax,
                           invz=args.invz,
                           render_style=render_style,
                           offset=args.offset,
                           cv2resize=args.cv2resize,
                           val_image_interval=args.val_image_interval)


if __name__ == "__main__":
  sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
  train()
