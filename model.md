# pose-gaze

Reads video → RTMW3D 17-joint pose → metric 3-D (pinhole + De Leva) → track → velocity (OLS+EMA) → ground plane (SVD) → overlay + MP4/NDJSON recording.

```
config.py  (no project imports)
  └─ backend.py
       └─ frontend.py  ← python frontend.py
```

---

## config.py

```python
# Paths
SCRIPT_DIR      = Path(__file__).parent.resolve()
MODEL_DIR       = SCRIPT_DIR / "models"
OUTPUT_DIR      = SCRIPT_DIR / "recordings"
CAMERA_JSON_DIR = SCRIPT_DIR / "media"
VIDEO_PATH      = "media/video_1808x1392_mvBlueCOUGAR-X109b_crop205-391-2013-1783.mp4"
FPS_FALLBACK    = 13.2
K_FALLBACK      = np.array([[4637.7,0.,2056.],[0.,4637.7,1088.],[0.,0.,1.]], f64)

# Detection / keypoint
MIN_KEYPOINT_CONFIDENCE       = 0.25
MIN_VISIBLE_KEYPOINTS         = 6
MIN_PIXEL_HEIGHT_SPAN         = 5.0
DETECTION_SCORE_THRESHOLD     = 0.30   # YOLOX
MIN_DETECTION_MEAN_CONFIDENCE = 0.20

# Tracker
TRACKER_IOU_THRESHOLD        = 0.25
TRACKER_MAX_INACTIVE_FRAMES  = 90
TRACKER_MIN_HIT_STREAK       = 2
TRACKER_CENTROID_DIST_NORM   = 100.0
TRACKER_CENTROID_DIST_WEIGHT = 0.3

# Body geometry — DO NOT CHANGE (De Leva 1996)
BODY_HEIGHT_PRIOR_M        = 1.75
RTMW3D_HEAD_ANKLE_Z_RATIO  = 0.96
VISIBLE_BODY_RATIO_ANKLES  = 1.00
VISIBLE_BODY_RATIO_KNEES   = 0.75
VISIBLE_BODY_RATIO_HIPS    = 0.52
JOINT_DEPTH_MIN_M          = 0.1
JOINT_DEPTH_MAX_M          = 60.0
BODY_HEIGHT_MIN_M          = 0.5
BODY_HEIGHT_MAX_M          = 2.5
HEIGHT_BUFFER_WINDOW_FRAMES  = 60
HEIGHT_BUFFER_MIN_SAMPLES    = 5
HEIGHT_OUTLIER_MAX_DEVIATION = 0.30

DE_LEVA_SEGMENTS = [   # (joint_a, joint_b, body_height_fraction)
    (0,15,1.00),(0,16,1.00),(0,11,0.52),(0,12,0.52),
    (0, 5,0.19),(0, 6,0.19),(5,11,0.29),(6,12,0.29),
    (11,15,0.48),(12,16,0.48),(11,13,0.25),(12,14,0.25),
    (13,15,0.23),(14,16,0.23),(3,5,0.09),(4,6,0.09),
]

# EMA
EMA_ALPHA_PIXEL_HEIGHT = 0.30
EMA_ALPHA_DISPLAY      = 0.35

# Velocity
VELOCITY_BUFFER_S     = 2.0   # OLS window [s]
VELOCITY_MIN_SPAN_S   = 0.5   # min span before output [s]
VELOCITY_MAX_DT_S     = 3.0   # flush buffer on gap > this [s]
VELOCITY_MAX_SPEED_MS = 8.0   # hard cap before EMA [m/s]
VELOCITY_EMA_TAU_S    = 2.0   # time-constant [s], fps-invariant
VELOCITY_BUFFER_SIZE  = 60    # deque maxlen

# Display
LABEL_SMOOTH_WINDOW_S  = 2.0
DISPLAY_MAX_WIDTH_PX   = 1580
DISPLAY_MAX_HEIGHT_PX  = 840
SEEK_STEP_SECONDS      = 5
GAZE_RAY_LENGTH_M      = 3.0

# Joint indices (COCO-17)
JOINT_NOSE=0  JOINT_LEFT_EAR=3    JOINT_RIGHT_EAR=4
JOINT_LEFT_SHOULDER=5   JOINT_RIGHT_SHOULDER=6
JOINT_LEFT_HIP=11       JOINT_RIGHT_HIP=12
JOINT_LEFT_KNEE=13      JOINT_RIGHT_KNEE=14
JOINT_LEFT_ANKLE=15     JOINT_RIGHT_ANKLE=16

# Skeleton (1-based indices → subtract 1 before indexing)
SKELETON_EDGES = [
    (16,14),(14,12),(17,15),(15,13),(12,13),(6,12),(7,13),(6,7),
    (6,8),(7,9),(8,10),(9,11),(2,3),(1,2),(1,3),(2,4),(3,5),(4,6),(5,7),
]
SKELETON_EDGE_COLORS = [   # BGR rainbow
    (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),
    (0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),
    (0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85),
]

# Colors (BGR)
COLOR_KEYPOINT_NOSE=(0,0,255)  COLOR_KEYPOINT_EAR=(255,0,0)  COLOR_KEYPOINT_OTHER=(0,255,0)
COLOR_EAR_CONNECTOR=(150,150,150)  COLOR_GAZE_ORIGIN_DOT=(0,255,0)
COLOR_GAZE_LINE=(0,255,255)  COLOR_GAZE_ENDPOINT=(0,0,255)
COLOR_LABEL_DIST=(255,255,0)  COLOR_LABEL_ANGLES=(255,0,255)  COLOR_LABEL_GAZE=(0,255,255)
COLOR_BEARING_RAY=(255,200,0)  COLOR_GROUND_LONG_LINES=(80,80,80)
COLOR_GROUND_PLANE_FILL=(30,80,30)
COLOR_HUD_RECORDING=(0,0,255)  COLOR_HUD_PAUSED=(0,220,255)  COLOR_HUD_RUNNING=(0,255,0)
COLOR_HUD_CAM_INFO=(200,200,200)  COLOR_HUD_HINT=(255,255,255)
BEARING_ENDPOINT_COLOR=(255,200,0)

# Drawing sizes
KEYPOINT_RADIUS=2  GAZE_ORIGIN_DOT_RADIUS=2  GAZE_ENDPOINT_RADIUS=3
SKELETON_LINE_THICKNESS=1  GAZE_LINE_THICKNESS=2  TEXT_OUTLINE_THICKNESS=2
BEARING_RAY_THICKNESS=1  BEARING_ENDPOINT_RADIUS=4
LABEL_LINE_SPACING_PX=26  LABEL_FONT_SCALE_SMALL=0.55
HUD_FONT_SCALE_STATUS=0.7  HUD_FONT_SCALE_CAM=0.45

# Ground plane
GROUND_PLANE_ENABLED=True  GROUND_PLANE_MAX_DEPTH_M=20.0
GROUND_PLANE_SAMPLE_INTERVAL_S=0.5  GROUND_PLANE_SAMPLE_BUFFER=200
GROUND_PLANE_MIN_SAMPLES=20  GROUND_PLANE_LINE_THICKNESS=1
GROUND_PLANE_FILL_ALPHA=0.25  GROUND_PLANE_X_STEP_M=1.0  GROUND_PLANE_X_HALF_M=4.0
GROUND_PLANE_NEAR_FRAC=0.75  GROUND_PLANE_FAR_FRAC=0.25
```

### Dataclasses
```python
@dataclass class GroundPlane:
    centroid: ndarray(3,)f32   # mean of joint samples, camera frame [m]
    normal:   ndarray(3,)f32   # unit normal; invariant: normal[1] < 0
    n_samples: int = 0

@dataclass class MessageHeader:
    sec:int; nanosec:int; frame_id:str

@dataclass class ObjectSize:
    l:float; w:float; h:float

@dataclass class ObjectPose:
    x:float; y:float; yaw:float
    z:float=0.; roll:float=0.; pitch:float=0.
    covariance:List[float]=field(default_factory=list)

@dataclass class ObjectVelocity:   # vx=fwd, vy=lat(right+), vz=up [m/s]
    vx:float=0.; vy:float=0.; vz:float=0.
    rx:float=0.; ry:float=0.; rz:float=0.
    covariance:List[float]=field(default_factory=list)

@dataclass class ObjectLabel:
    text:str=""; value:float=0.

@dataclass class DetectedObject:
    id:int; object_class:str; score:float
    pose:ObjectPose; size:ObjectSize
    velocity:Optional[ObjectVelocity]=None
    label:Optional[ObjectLabel]=None
    def to_dict(self):
        d=asdict(self); d["class"]=d.pop("object_class")
        return {k:v for k,v in d.items() if v is not None}

@dataclass class DetectionMessage:
    header:MessageHeader; objects:List[DetectedObject]
    def to_dict(self): return {"header":asdict(self.header),"objects":[o.to_dict() for o in self.objects]}
    def to_json(self): return json.dumps(self.to_dict(),separators=(",",":"))
```

---

## Coordinate Systems
| Frame | X | Y | Z | Origin |
|---|---|---|---|---|
| Pixel | right | down | — | top-left |
| Camera-3D | right | down | forward | lens |
| Message | forward | right | up | lens |

`msg.x=Z_cam  msg.y=X_cam  msg.z=−Y_cam`

---

## backend.py

### Imports / Startup
```python
_inference_model = Wholebody3d(mode="balanced", backend="onnxruntime", device=DEVICE)
_inference_model.det_model.score_thr = DETECTION_SCORE_THRESHOLD
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # torch optional

# Precomputed from DE_LEVA_SEGMENTS
_DELEVA_JOINT_A = ndarray(16,)int32
_DELEVA_JOINT_B = ndarray(16,)int32
_DELEVA_RATIO   = ndarray(16,)f32
_TORSO_JOINTS   = [5, 6, 11, 12]   # module-level const, never recomputed per call
```

### PersonData (dataclass)
```python
person_id:int  pixel_coords:ndarray(17,2)f32  joint_scores:ndarray(17,)f32
joints_meters:ndarray(17,3)f32  joint_visible:ndarray(17,)bool
distance_m:float  elevation_deg:float  azimuth_deg:float  height_m:float
gaze_origin_m:ndarray(3,)|None  gaze_direction:ndarray(3,)unit|None
velocity_m_per_s:ObjectVelocity|None
```

### Module State
```python
_tracker        = PersonTracker()
_profiles       = PersonProfiles()
_camera_params:  dict|None = None
_ground_ref_buf: deque(maxlen=200)
_last_ground_t:  float = -GROUND_PLANE_SAMPLE_INTERVAL_S
_ground_plane_g: GroundPlane|None = None
_last_det_msg:   DetectionMessage|None = None
```

### Public API
```
init_camera(w,h,fps)                    → dict  (sets _camera_params)
process_frame(bgr,t_s=0.)              → List[PersonData]
get_ground_plane()                      → GroundPlane|None
build_detection_message(persons,t_s)   → DetectionMessage  (sets _last_det_msg)
get_last_detection_message()            → DetectionMessage|None
get_camera()                            → dict|None
```

### init_camera(w,h,fps) → dict
Camera JSON search (in order, stops at first match):
1. glob `camera_{w}x{h}_*.json` in `[CAMERA_JSON_DIR, parent, cwd, SCRIPT_DIR]` ±1px
2. Content-scan: parse JSON WIDTH/HEIGHT ±1
3. Same-aspect-ratio JSON → scale FX/FY/CX/CY by sx=w/jw, sy=h/jh
4. K_FALLBACK

Dict keys: `FX FY CX CY W H FPS K:ndarray(3,3)f64 D:ndarray(5,)f64 source:str crop:dict|None`

### process_frame(bgr, t_s) → List[PersonData]
```
kp3d_all(N,17,3)f32, scores_all(N,17)f32, _, kp2d_all(N,17,2)f32 = _inference_model(bgr)

bboxes, confs, det_data = [], [], []
for each detection i:
    px=kp2d_all[i,:17,:2].f32  k3d=kp3d_all[i,:17].f32  sc=scores_all[i,:17].f32
    skip if visible_count < MIN_VISIBLE_KEYPOINTS
    skip if mean(sc[sc>0.25]) < MIN_DETECTION_MEAN_CONFIDENCE
    bbox = [vis_px.min(axis=0), vis_px.max(axis=0)]  # [x1,y1,x2,y2]
    append to bboxes/confs/det_data

for det_idx, (pid, _) in enumerate(_tracker.update(bboxes, confs)):
    # det_idx aligns directly with det_data / bboxes order
    px, k3d, sc = det_data[det_idx]
    raw_ph = _pixel_body_height_span(px, sc)
    sm_ph  = _profiles.smooth_pixel_height(pid, raw_ph) if raw_ph else None
    result = compute_3d_pose(px,k3d,sc, FX,FY,CX,CY, h_ref=get_height_reference(pid), px_h_hint=sm_ph)
    if result[0] is None: continue
    jm, h_m, centroid_3d, (dist,elev,azim) = result
    _profiles.push_height_sample(pid, h_m, head_vis, ankle_vis)
    sm_dist,sm_elev,sm_azim,sm_h = _profiles.smooth_display_values(pid,dist,elev,azim,h_ref)
    gaze_origin, gaze_dir = compute_gaze_ray_3d(jm, sc)
    velocity = _profiles.compute_velocity(pid, centroid_3d, t_s)
    append PersonData(...)

if GROUND_PLANE_ENABLED and t_s - _last_ground_t >= 0.5:
    per person: collect lowest visible joints (ankles>knees>hips), append to _ground_ref_buf
    _fit_ground_plane(); _last_ground_t = t_s
```

### compute_3d_pose(px,k3d,sc,fx,fy,cx,cy,h_ref,px_h_hint) → (jm,h_m,centroid,geo)|fail
fail = `(None, 0., None, None)`

```python
# units_to_metres via confidence-weighted De Leva
sa=sc[JOINT_A]; sb=sc[JOINT_B]; span=|k3d[A,2]-k3d[B,2]|; cp=sa*sb
valid = (sa>0.25)&(sb>0.25)&(span>1e-3)
u2m = sum(RATIO[valid] * H_PRIOR/span[valid] * cp[valid]) / sum(cp[valid])

# measured height (only if head [0-4] and ankle [15,16] both have visible joints)
h_m = clip(|mean(k3d[head,2]) - mean(k3d[ankle,2])| * u2m / 0.96, 0.5, 2.5)

# pixel height: use hint if > 5.0 else compute
# top priority:  [0] → [1,2] → [3,4]   (first group with visible joints)
# bottom priority: [15,16] → [13,14] → [11,12]

# depth anchor — visible_body_ratio check order: ankles→hips→knees
ratio = ANKLES(1.00) if any([15,16] visible) else
        HIPS(0.52)   if any([11,12] visible) else
        KNEES(0.75)
z_root = fy * h_ref * ratio / px_height
depth[j] = clip(z_root + k3d[j,2]*u2m, 0.1, 60.0).f32

jm[:,2]=depth; jm[:,0]=(px[:,0]-cx)*depth/fx; jm[:,1]=(px[:,1]-cy)*depth/fy

# centroid: torso joints [5,6,11,12]; fallback all-visible if <2 torso visible
centroid = mean(jm[torso_vis]) or mean(jm[all_vis])
r = ||centroid||; hd = hypot(centroid[0], centroid[2])
return (jm, h_m, centroid,
        (r, deg(atan2(-centroid[1], hd)), deg(atan2(centroid[0], centroid[2]))))
```

### _pixel_body_height_span(px,sc) → float|None
```
top_y:  mean(px[[0],1]) if any visible, else [[1,2]], else [[3,4]]
bot_y:  mean(px[[15,16],1]) if any visible, else [[13,14]], else [[11,12]]
return |bot_y - top_y| or None
```

### compute_gaze_ray_3d(jm,sc) → (origin,unit)|(None,None)
Requires joints 0,3,4 all > 0.25.
`unit = normalise(jm[0] - (jm[3]+jm[4])*0.5)`

### gaze_angles_3d(d) → (az_deg, el_deg)
`az = deg(atan2(d[0], d[2]))`  0°=camera forward, +°=right
`el = deg(atan2(-d[1], hypot(d[0],d[2])))`  +°=up

### proj3d2d(pt,fx,fy,cx,cy) → ndarray(2,)f32|None
None if pt[2]≤0. `[pt[0]/pt[2]*fx+cx, pt[1]/pt[2]*fy+cy]`

### PersonProfiles
```python
# Internal dicts keyed by person_id
_height_history:   {pid: deque(maxlen=60)[float]}
_pixel_height_ema: {pid: float}
_display_ema:      {pid: (dist,elev,azim,h)}
_pos_history:      {pid: deque(maxlen=60)[(t_s, ndarray(3,))]}
_vel_ema:          {pid: ndarray(3,)}  # stores [cam_X, cam_Y, cam_Z]

smooth_pixel_height(pid,v) → EMA(α=0.30)

smooth_display_values(pid,dist,elev,azim,h):
    first call → initialise, return raw (no lag)
    azim wrap-around: delta>180 → azim-=360; delta<-180 → azim+=360
    EMA(α=0.35) on all four scalars

push_height_sample(pid,h_m,head_vis,ankle_vis):
    skip unless both visible AND 0.5≤h_m≤2.5
    skip if |h_m-median| > 0.30*median (after ≥5 samples)

get_height_reference(pid) → median(history) if len≥5 else 1.75
```

### PersonProfiles.compute_velocity(pid,centroid,t_s) → ObjectVelocity
```python
buf = _pos_history[pid]   # deque[(t, ndarray(3,))]

# Flush on seek/pause
if buf and t_s - buf[-1][0] > 3.0:
    buf.clear(); del _vel_ema[pid]

buf.append((t_s, centroid.copy()))
samples = [(t,p) for t,p in buf if t >= t_s - 2.0]
prev = _vel_ema.get(pid)

if len(samples)<3 or span<0.5:
    return ObjectVelocity() if prev is None
    else ObjectVelocity(vx=prev[2], vy=prev[0], vz=-prev[1])

T=array(times,f64); P=array(positions,f64)  # (N,3)
Tc=T-mean(T); Pc=P-mean(P,axis=0)
raw = (Tc@Pc) / sum(Tc²)   # shape (3,)

if ||raw||>8.0: raw *= 8.0/||raw||

dt = buf[-1][0] - buf[-2][0]
α = 1-exp(-dt/2.0) if dt>1e-4 else 0.1
ema = α*raw + (1-α)*prev  if prev else raw.copy()
_vel_ema[pid] = ema   # stored as [cam_X, cam_Y, cam_Z]

return ObjectVelocity(vx=ema[2], vy=ema[0], vz=-ema[1])
#                        Z_cam      X_cam     -Y_cam
```

### PersonTracker
```python
_tracks: {pid: {bbox:[x1,y1,x2,y2], inactive_frames:int, conf:float, hit_streak:int}}
_next_person_id: int  # monotonic, never reused

update(bboxes,confs) → List[(pid,conf)]:
    increment inactive_frames for all; drop if > 90
    per detection:
        score = IoU + (1-min(1,centroid_dist/100))*0.3
        match if score>best AND (IoU>0.25 OR dist<100)
        new track: hit_streak=1, inactive_frames=0
    return [(pid,conf) for result_ids if hit_streak>=2]
    # result order matches input bboxes order (det_idx alignment)
```

### _fit_ground_plane()
```python
pts = array(_ground_ref_buf, f64)
centroid = pts.mean(axis=0)
_,_,Vt = svd(pts-centroid)
normal = Vt[-1].f32; if normal[1]>0: normal=-normal
_ground_plane_g = GroundPlane(centroid.f32, normal, len(_ground_ref_buf))
```

### build_detection_message / _person_to_obj
```python
def _person_to_obj(p) → DetectedObject:
    er=rad(p.elevation_deg); ar=rad(p.azimuth_deg)
    hd = p.distance_m * cos(er)
    pose.x = hd*cos(ar)   # message forward = Z_cam = dist*cos(el)*cos(az)
    pose.y = hd*sin(ar)   # message lateral = X_cam
    pose.z = p.distance_m*sin(er)   # message up = -Y_cam
    pose.yaw = atan2(gaze_dir[0],gaze_dir[2]) if gaze_dir else ar
    size = ObjectSize(l=0.25, w=0.45, h=p.height_m)
    label.text = f"H:{h*100:.0f}cm spd:{speed_kmh:.1f}km/h"
    label.value = round(p.height_m,3)
```

---

## frontend.py

### LabelSmoother
```python
_buf: {(pid,field): deque[(t_s,v)]}
push(pid,field,v,t_s) → rolling_mean(last LABEL_SMOOTH_WINDOW_S=2.0s)
# speed additionally: round(v*2)/2  (0.5 km/h quantisation in draw_person)
```

### _text(img,text,pos,scale,color)
```python
cv2.putText(img,text,pos,FONT_HERSHEY_SIMPLEX,scale,(0,0,0),TEXT_OUTLINE+2)
cv2.putText(img,text,pos,FONT_HERSHEY_SIMPLEX,scale,color,TEXT_OUTLINE)
```

### draw_ground_plane(img,gp,fx,fy,cx,cy,ds)
```python
y_near=int(h*0.75); y_far=int(h*0.25)

def row_to_z(y):
    if gp and |gp.normal[1]|>1e-3:
        d=-dot(n,c); ray=[0,(y-cy)/fy,1]; denom=dot(n,ray)
        if |denom|>1e-6 and -d/denom>0.01: return -d/denom
    dy=y-cy; return max(0.3, 1.2*fy/dy) if dy>2 else MAX_DEPTH_M

z_near=row_to_z(y_near); z_far=row_to_z(y_far)
if z_near<=0 or z_far<=z_near*1.05: z_near,z_far=1.5,20.0

# Trapezoid: xl_n=(-4/z_near)*fx+cx ... clipped to [0,w-1]
overlay=img.copy(); fillPoly(overlay, poly, FILL_COLOR)
addWeighted(overlay,0.25, img,0.75, 0, img)
# Grid lines: x_m in arange(-4,4+eps,1.0), line (xn,y_near)→(xf,y_far)
```

### draw_person(img,p,fx,fy,cx,cy,ds,t_s)
```python
# Projection
safe_d = where(vis, jm[:,2], 1.0)
px_x = (jm[:,0]/safe_d)*fx+cx  .astype(int32)
px_y = (jm[:,1]/safe_d)*fy+cy  .astype(int32)

# Bearing ray: (cx,cy) → projected torso centroid (from dist/elev/azim)
torso_3d = [hd*sin(az), -dist*sin(el), hd*cos(az)]  # hd=dist*cos(el)
line (cx,cy)→torso_px; circle at torso_px; distance label at midpoint

# Skeleton: for i,(a1,b1) in enumerate(SKELETON_EDGES): a=a1-1; b=b1-1
# Gaze ray: origin=gaze_origin_m; ep=origin+dir*3.0; proj both
# Ear connector: line jm[3]→jm[4] if both visible
# Joint dots: nose=red, ears=blue, others=green

# Label block (smoothed values, ax/ay = above topmost visible joint)
ax = px_x[topmost] - round(30*ds)
ay = px_y[topmost] - round(12*ds) - g    # g=round(26*ds)
# bottom→top:
#   ay:     #{pid} {conf:.0%}              COLOR_LABEL_GAZE
#   ay-g:   gaze:{az:+.0f}/{el:+.0f}      COLOR_LABEL_GAZE
#   ay-2g:  ver:{elev:+.1f}               COLOR_LABEL_ANGLES
#   ay-3g:  hor:{azim:+.1f}               COLOR_LABEL_ANGLES
#   ay-4g:  spd:{round(sm*2)/2:.1f}km/h   COLOR_LABEL_ANGLES
#   ay-5g:  H:{h*100:.0f}cm               COLOR_LABEL_DIST
```

### draw_hud(img,n,t_s,total_s,rec,paused,src,h,ds)
```python
row1=max(20,round(30*ds)): f"Persons:{n} {t_s:.1f}s/{total_s:.1f}s [REC][PAUSED]"
row2=max(36,round(56*ds)): f"CAM:{src}"
bottom=h-max(8,round(12*ds)): "[A]<<5s [D]>>5s [Space]Pause [R]Rec+NDJSON [Q]Quit"
color = REC_RED|PAUSED_YELLOW|RUNNING_GREEN
```

### main()
```python
cap = VideoCapture(VIDEO_PATH)
fw,fh,fps,n_frames,total_s = from cap.get(...)
cam = init_camera(fw,fh,fps)
fx,fy,cx,cy = cam["FX"],cam["FY"],cam["CX"],cam["CY"]
ds    = 1.0 + (sqrt(fw*fh / (1808*1392)) - 1) * 0.6   # size scaling
scale = min(1580/fw, 840/fh, 1.0)                       # display downscale
dw,dh = int(fw*scale), int(fh*scale)

namedWindow("RTMPose 3D",WINDOW_NORMAL); resizeWindow(dw,dh); moveWindow(0,0)

loop:
  if not paused: cap.read() → process_frame(frame, cur_frame/fps)
                 log FPS every 1s to stdout
  canvas = frame.copy() | last_canvas.copy() | zeros
  if not paused: draw_ground_plane(); draw_person() per p
  draw_hud()
  if rec: vw.write(canvas); nw.write(msg.to_json()+"
")
  imshow; waitKey(100 if paused else 1)

Keys:
  Space → paused=not paused
  R     → toggle rec; on start: VideoWriter(mp4v,fps,(fw,fh)) + open(ndjson,"w")
           on stop: vw.release(); nw.close()
  A/D   → cap.set(POS_FRAMES, pos ± SEEK_STEP*fps); paused=False
  Q/Esc → flush rec; break
```

---

## Invariants
1. `init_camera()` before `process_frame()` — else RuntimeError.
2. `compute_3d_pose` returns 4-tuple; check `result[0] is None`.
3. `velocity_m_per_s` is None until ≥0.5s of data — never zero-filled before that.
4. `joint_visible[j]` ≡ `joint_scores[j] > 0.25` — always.
5. `joints_meters` always (17,3)f32; invisible rows are projected garbage — mask before use.
6. `gaze_direction` is unit-length or None.
7. `GroundPlane.normal[1] < 0` always.
8. Tracker IDs: monotonic int from 0, never reused.
9. `SKELETON_EDGES` indices are 1-based.
10. `_TORSO_JOINTS=[5,6,11,12]` module-level — never recomputed per call.
11. `_vel_ema[pid]` stores `[cam_X, cam_Y, cam_Z]`; vx=ema[2], vy=ema[0], vz=-ema[1].
12. `_tracker.update()` result order matches input bbox/det_data order; use `det_idx`.
13. `visible_body_ratio` check order: ankles→**hips**→knees (hips checked before knees).
14. `build_detection_message()` called only when recording is active.
15. `gaze_angles_3d`: az=atan2(d[0],d[2]) not atan2(d[0],−d[2]).
