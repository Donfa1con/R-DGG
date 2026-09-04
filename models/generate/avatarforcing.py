#!/usr/bin/env python
"""
Batch driver: run AvatarForcing over the corpus (GT-audio conditioned),
one <stem>.mp4 per stem, matching the GT/dystream layout for the R-DGG
evaluation pipeline.

Mapping (symmetric dyadic pairs, via data/pairs184.txt):
    target stem X (the animated avatar)  <->  partner Y ("user"/speaker)
      avatar_ref   = <reference_dir>/X.png     avatar_audio = <wav_dir>/X.wav (GT)
      user_audio   = <wav_dir>/Y.wav           user_video   = 25fps face crops of Y
      output       = <out_dir>/X.mp4           (25 fps, avatar_audio muxed)

Design (why streaming): this driver is fully streaming and bounded to ~one
block + the model, so a low-RAM host never holds the whole decoded video
(fp16 ~13.6 GB for a 346 s clip, plus a uint8 copy ~6.8 GB):
  * partner frames streamed from disk in chunks (encode_user_motion),
  * decode runs block-by-block and pipes each block straight into an ffmpeg
    encoder (raw rgb24 -> h264, audio muxed in the same process) — the full
    (T,3,512,512) video is NEVER materialised on GPU or in RAM.
Tone-map is fixed [-1,1]->[0,255] (avatar output is trained in [-1,1]); this
differs slightly from the demo's global-min/max stretch but is content-identical
and removes the need to hold all frames. Idempotent; ffmpeg/ffprobe must be on
PATH (asserted). No *_tmp orphans; a failed stem's partial output is removed.
"""
import os, sys, glob, argparse, subprocess, shutil, time, tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
# Model source = the pinned, in-repo git submodule at models/avatarforcing (override with
# AVATARFORCING_REPO). The reviewer's setup applies models/patches/avatarforcing.patch to it via
# ../../setup.sh (self.opt fix so init_network doesn't read a __main__-only global, 30s->600s audio
# cap, RoPE max_seq_len 1024->16384 for full-length SI clips; "No model/CFG change"), and the model's
# own download step populates <REPO>/pretrained_dir/ with the weights + wav2vec2-base-960h.
REPO = os.environ.get("AVATARFORCING_REPO", os.path.abspath(os.path.join(_HERE, os.pardir, "avatarforcing")))
_DATA = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "data"))
sys.path.insert(0, REPO)

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from inference import InferenceAgent


# ----------------------------------------------------------------------------- helpers

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def require_tools():
    for t in ("ffmpeg", "ffprobe"):
        if shutil.which(t) is None:
            raise SystemExit(
                f"FATAL: {t} not on PATH. Activate the model's conda env first, e.g.:\n"
                f"  conda activate avatarforcing")


def read_pairs(pairs_path):
    """listener_stem -> speaker_stem (no extension). splitlines() so SI's final
    no-trailing-newline entry is included."""
    mp = {}
    with open(pairs_path) as f:
        for line in f.read().splitlines():
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            mp[a[:-4] if a.endswith('.wav') else a] = b[:-4] if b.endswith('.wav') else b
    return mp


def ffprobe_stream_types(path):
    """codec_type lines, or None if ffprobe itself failed (so callers can tell a
    'bad file' from a 'tool error' and NOT delete a good file on tool error)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


# --------------------------------------------- streaming ffmpeg writer (bounded memory)

class FFmpegWriter:
    """Pipe raw rgb24 frames to an ffmpeg process that encodes h264 (yuv420p) and
    muxes the avatar audio in the same process. Memory-bounded: nothing is held."""
    def __init__(self, out_path, audio_path, fps, size):
        self.out = out_path
        self.broken = False
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}",
               "-r", str(fps), "-i", "pipe:0",
               "-i", audio_path, "-shortest",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "16000",
               out_path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bytes):
        # -shortest finalizes ffmpeg at the (avatar) audio end and closes the
        # read end of the pipe; trailing video frames (video len =
        # T=max(avatar,user) audio, but muxed audio = avatar wav) then raise
        # BrokenPipe. That is the intended trim (same as stock save_video's
        # -shortest), NOT an error — stop feeding and let ffmpeg finalize.
        if self.broken:
            return
        try:
            self.p.stdin.write(frame_bytes)
        except BrokenPipeError:
            self.broken = True

    def close(self):
        try:
            self.p.stdin.close()
        except BrokenPipeError:
            pass
        try:
            rc = self.p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self.p.kill(); rc = self.p.wait()
        # rc==0 with a valid file is success even when -shortest trimmed the
        # video (the BrokenPipe path above); only a real encode failure raises.
        if rc != 0 or not os.path.exists(self.out):
            raise RuntimeError(f"ffmpeg encode/mux failed (rc={rc}) for {self.out}")

    def abort(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.kill(); self.p.wait(timeout=10)
        except Exception:
            pass
        if os.path.exists(self.out):
            os.remove(self.out)


# --------------------------------------------------- partner (user) video preprocessing

def _mean_face_bbox(frame_paths, fa, sample_stride):
    bsy, bsx, my, mx = [], [], [], []
    for p in frame_paths[::sample_stride]:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        mult = 360. / img.shape[0]
        rs = cv2.resize(img, (0, 0), fx=mult, fy=mult,
                        interpolation=cv2.INTER_AREA if mult < 1 else cv2.INTER_CUBIC)
        bb = fa.face_detector.detect_from_image(rs)
        bb = [(int(x1/mult), int(y1/mult), int(x2/mult), int(y2/mult), s)
              for (x1, y1, x2, y2, s) in bb if s > 0.95]
        if not bb:
            continue
        x1, y1, x2, y2, _ = bb[0]
        bsy.append((y2-y1)/2); bsx.append((x2-x1)/2)
        my.append((y1+y2)/2);  mx.append((x1+x2)/2)
    if not bsy:
        raise RuntimeError("no face detected in partner video sample")
    return np.mean(bsx), np.mean(bsy), np.mean(mx), np.mean(my)


def preprocess_partner(video_path, out_dir, fa, input_size, fps, pad_ratio=1.0, sample_stride=25):
    """25fps frames, face-crop (mean bbox, AvatarForcing formula), resize to
    input_size, save BGR jpgs. Cached via .done; partial (crashed) dir retried."""
    done = os.path.join(out_dir, ".done")
    if os.path.exists(done):
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    tmp = out_dir + "_raw"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                        "-r", str(fps), "-qscale:v", "2", os.path.join(tmp, "%06d.jpg")], check=True)
        raw = sorted(glob.glob(os.path.join(tmp, "*.jpg")))
        if not raw:
            raise RuntimeError(f"ffmpeg produced no frames for {video_path}")
        bsx, bsy, mx, my = _mean_face_bbox(raw, fa, sample_stride)
        h, w = cv2.imread(raw[0]).shape[:2]
        bs = int(max(bsy, bsx) * (1 + pad_ratio))
        x1, y1 = max(int(mx - bs), 0), max(int(my - bs), 0)
        x2, y2 = min(int(mx + bs), w), min(int(my + bs), h)
        cbsx, cbsy = x2 - x1, y2 - y1
        cmx, cmy = int(x1 + cbsx // 2), int(y1 + cbsy // 2)
        cbs = int(min(cbsx, cbsy) // 2)
        for p in raw:
            face = cv2.imread(p)[cmy-cbs:cmy+cbs, cmx-cbs:cmx+cbs]
            face = cv2.resize(face, (input_size, input_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), face)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    open(done, "w").close()
    return out_dir


# ------------------------------------------------- memory-safe user-motion streaming enc

def make_streaming_encode(G, rank):
    chunk = G.num_frames_for_clip

    @torch.inference_mode()
    def _enc(frames_dir):
        paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        outs = []
        for i in range(0, len(paths), chunk):
            batch = []
            for p in paths[i:i + chunk]:
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.).sub_(0.5).div_(0.5)
                batch.append(t.unsqueeze(0))
            ft = torch.cat(batch, 0).to(rank)
            r_d = G.motion_autoencoder.enc.enc_motion(ft)
            r_d = G.motion_autoencoder.dec.direction(r_d)
            outs.append(r_d)
            del ft
        return torch.cat(outs, 0).unsqueeze(0)

    return _enc


# ------------------------------- streaming decode -> ffmpeg (never holds the full video)

def make_streaming_decode(G, get_writer):
    """Replacement for G.decode_latent_into_image: decode block-by-block, fixed
    [-1,1]->uint8 tone-map, write each block straight to the current ffmpeg
    writer. Returns {'d_hat': None} — frames are already encoded. GPU holds at
    most one block; RAM holds nothing large."""
    bs = G.num_frames_per_block

    @torch.inference_mode()
    def _dec(s_r, r_s, s_r_feats, r_d):
        T = r_d.shape[1]; B = r_d.shape[0]
        assert B == 1, "streaming writer assumes batch 1"
        s_r = s_r.unsqueeze(1)
        feats = [f.repeat_interleave(bs, dim=0) for f in s_r_feats]
        w = get_writer()
        for i in range(0, T, bs):
            j = min(i + bs, T); L = j - i
            blk = r_d[:, i:j]
            if L < bs:
                blk = F.pad(blk, (0, 0, 0, bs - L), mode='replicate')
            x = (s_r + blk).reshape(B * bs, -1)
            img, _ = G.motion_autoencoder.dec(x, alpha=None, feats=feats)   # (B*bs,3,H,W)
            img = img.reshape(B, bs, *img.shape[1:])[0, :L]                 # (L,3,H,W)
            u = ((img.float().clamp(-1, 1) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            u = u.permute(0, 2, 3, 1).contiguous().cpu().numpy()            # (L,H,W,3) rgb
            w.write(u.tobytes())
            del x, img, u
        return {'d_hat': None}

    return _dec


# ------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Batch AvatarForcing (dyadic listener generation) -- SI-184 reproduction")
    ap.add_argument("--infer_config", default=os.path.join(REPO, "configs/inference.yaml"))
    ap.add_argument("--mae_ckpt_path", default=os.path.join(REPO, "pretrained_dir/motion_autoencoder.pth"))
    ap.add_argument("--ckpt_path", default=os.path.join(REPO, "pretrained_dir/flow_transformer.pth"))
    ap.add_argument("--reference_dir", default=os.path.join(_DATA, "references"),
                    help="dir of <stem>.png avatar portraits")
    ap.add_argument("--wav_dir", default=os.path.join(_DATA, "GT", "LS_AL_wav"),
                    help="dir of <stem>.wav (avatar own audio + partner audio, keyed by --pairs)")
    ap.add_argument("--gt_video_dir", default=os.path.join(_DATA, "GT", "LS_AL"),
                    help="dir of partner <stem>.mp4 (partner face motion is encoded from these frames)")
    ap.add_argument("--pairs", default=os.path.join(_DATA, "pairs184.txt"),
                    help="'<listener>.wav <speaker>.wav' per line (dyadic pairing)")
    ap.add_argument("--out_dir",
                    default=os.path.join(_DATA, "AvatarForcing", "LS_AL"))
    ap.add_argument("--work_dir",
                    default=os.path.join(tempfile.gettempdir(), "rdgg_avatarforcing_work"),
                    help="scratch for partner frames; keep OUTSIDE the corpus")
    ap.add_argument("--stems", default="all",
                    help="'all' (use --stems_file if present, else glob --reference_dir) or a comma list")
    ap.add_argument("--stems_file", default=os.path.join(_DATA, "pairs184.txt"),
                    help="pairing list (col1 = listener stems) used when --stems all; falls back to globbing --reference_dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--a_cfg_scale", type=float, default=2.0)
    ap.add_argument("--u_cfg_scale", type=float, default=1.0)
    ap.add_argument("--ext", default="mp4")
    ap.add_argument("--max_consec_fail", type=int, default=3)
    args = ap.parse_args()

    require_tools()
    infer_config = OmegaConf.load(args.infer_config)
    opt = OmegaConf.merge(infer_config, OmegaConf.create({
        "mae_ckpt_path": args.mae_ckpt_path, "ckpt_path": args.ckpt_path,
        # config ships wav2vec_model_path as a cwd-relative "pretrained_dir/wav2vec2-base-960h"
        # (loaded local_files_only); anchor it under REPO so it resolves regardless of cwd.
        "wav2vec_model_path": os.path.join(REPO, "pretrained_dir", "wav2vec2-base-960h"),
        "result_dir": args.out_dir, "rank": 0, "ngpus": 1,
        "avatar_ref_path": None, "avatar_audio_path": None,
        "user_audio_path": None, "user_video_path": None,
    }))

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    log("loading AvatarForcing model ...")
    agent = InferenceAgent(opt)
    agent.G.eval()
    dp = agent.data_processor
    rank, input_size, fps = opt.rank, opt.input_size, opt.fps

    agent.G._cur_writer = None
    agent.G.encode_user_motion = make_streaming_encode(agent.G, rank)
    agent.G.decode_latent_into_image = make_streaming_decode(agent.G, lambda: agent.G._cur_writer)

    pairs = read_pairs(args.pairs)
    if args.stems == "all":
        if args.stems_file and os.path.exists(args.stems_file):
            # listener stem = first whitespace token of each non-blank non-#-comment line, trailing .wav
            # stripped; dedup first-seen.
            stems, _seen = [], set()
            for s in open(args.stems_file):
                s = s.strip()
                if not s or s.startswith("#"):
                    continue
                tok = s.split()[0]
                if tok.endswith(".wav"):
                    tok = tok[:-4]
                if tok not in _seen:
                    _seen.add(tok)
                    stems.append(tok)
        else:
            stems = sorted(os.path.splitext(os.path.basename(p))[0]
                           for p in glob.glob(os.path.join(args.reference_dir, "*.png")))
    else:
        stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    if args.limit:
        stems = stems[:args.limit]

    log(f"{len(stems)} stems -> {args.out_dir}")
    ok = skip = fail = consec = 0
    for i, X in enumerate(stems, 1):
        out_path = os.path.join(args.out_dir, f"{X}.{args.ext}")
        if os.path.exists(out_path) and (ffprobe_stream_types(out_path) or "").find("video") >= 0:
            skip += 1; consec = 0; log(f"[{i}/{len(stems)}] skip {X}"); continue

        Y = pairs.get(X)
        if Y is None:
            fail += 1; consec += 1; log(f"[{i}/{len(stems)}] FAIL {X}: no partner in pairs.txt")
        else:
            ref_path = os.path.join(args.reference_dir, f"{X}.png")
            avatar_wav = os.path.join(args.wav_dir, f"{X}.wav")
            user_wav = os.path.join(args.wav_dir, f"{Y}.wav")
            partner_video = os.path.join(args.gt_video_dir, f"{Y}.{args.ext}")
            missing = next((p for p in (ref_path, avatar_wav, user_wav, partner_video)
                            if not os.path.exists(p)), None)
            if missing:
                fail += 1; consec += 1; log(f"[{i}/{len(stems)}] FAIL {X}: missing {missing}")
            else:
                writer = None
                try:
                    t0 = time.time()
                    frames_dir = preprocess_partner(
                        partner_video, os.path.join(args.work_dir, Y), dp.fa, input_size, fps)
                    avatar_ref = dp.transform(image=dp.preprocess_face(ref_path))["image"].unsqueeze(0)
                    avatar_a = dp.default_aud_loader(avatar_wav).unsqueeze(0)
                    user_a = dp.default_aud_loader(user_wav).unsqueeze(0)
                    data = {"avatar_ref": avatar_ref, "avatar_a": avatar_a,
                            "user_a": user_a, "user_frame": frames_dir}

                    writer = FFmpegWriter(out_path, avatar_wav, fps, input_size)
                    agent.G._cur_writer = writer
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        agent.G.inference(data=data, a_cfg_scale=args.a_cfg_scale,
                                          u_cfg_scale=args.u_cfg_scale, nfe=args.nfe,
                                          use_kv_cache=True)  # frames streamed to writer
                    writer.close(); writer = None
                    agent.G._cur_writer = None
                    torch.cuda.empty_cache()
                    st = ffprobe_stream_types(out_path)
                    if st is not None and "video" not in st:
                        raise RuntimeError("output has no video stream")
                    ok += 1; consec = 0
                    log(f"[{i}/{len(stems)}] OK {X} (partner {Y}, {time.time()-t0:.0f}s)")
                except Exception as e:
                    fail += 1; consec += 1
                    if writer is not None:
                        writer.abort()
                    elif os.path.exists(out_path):
                        os.remove(out_path)
                    agent.G._cur_writer = None
                    torch.cuda.empty_cache()
                    log(f"[{i}/{len(stems)}] FAIL {X}: {type(e).__name__}: {e}")

        if consec > args.max_consec_fail:
            log(f"ABORTING: {consec} consecutive failures — fix before continuing")
            break

    log(f"DONE  ok={ok} skip={skip} fail={fail}  out={args.out_dir}")


if __name__ == "__main__":
    main()
