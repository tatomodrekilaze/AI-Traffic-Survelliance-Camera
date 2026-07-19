import cv2
import numpy as np
import time
import os
import csv
import json
import random
import queue
import threading
import torch
from datetime import datetime, timezone, timedelta
from collections import deque, Counter
from ultralytics import YOLO
import ipywidgets as widgets
from IPython.display import display

# Has to be set before cv2 opens any capture, or ffmpeg just ignores it.
# Landed on this exact combo after the reader kept silently dying on the
# rtsp.me relay instead of reconnecting:
#   reconnect / reconnect_streamed / reconnect_at_eof - without these three,
#     ffmpeg treats one dropped HLS segment as end-of-stream and just stops.
#   reconnect_delay_max;2 - default backoff is way too patient for a feed
#     that's already cycling every ~6s on its own (see FreshestFrameReader).
#   analyzeduration/probesize;150000 - lower than default so ffmpeg commits
#     to a format guess fast instead of stalling the first open.
#   timeout;1200000 - 1.2s in microseconds. Long enough to survive one bad
#     segment, short enough that a genuinely dead link fails fast instead
#     of hanging the capture thread.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "reconnect;1|reconnect_streamed;1|reconnect_at_eof;1|"
    "reconnect_delay_max;2|analyzeduration;150000|probesize;150000|"
    "timeout;1200000"
)

RUN_CALIBRATION = False

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


def _init_persistent_storage():
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        base = '/content/drive/MyDrive/traffic_surveillance'
        os.makedirs(base, exist_ok=True)
        return base, True, None
    except Exception as e:
        return '/content', False, f"{type(e).__name__}: {e}"


_PERSIST_DIR, DRIVE_MOUNTED, _DRIVE_MOUNT_ERROR = _init_persistent_storage()

STREAM_URL = 'https://lon.rtsp.me/TGUFM63daUYHHMaouODCqg/1784324215/hls/4sBHNkR5.m3u8?'
FRAME_W, FRAME_H = 850, 480

# ---
# note to future me (or whoever else opens this file at 2am):
# this feed comes through rtsp.me's HLS relay, not a raw stream, and
# going by the actual timestamps in the admin log from the last run,
# it looks like that relay is chopping the live feed into ~6 second
# chunks. that lines up almost exactly with the periodic freeze people
# keep seeing -- it happens roughly every 6s, like clockwork, not
# randomly. that smells like a property of the source, not a bug we
# can code our way out of on this end. if rtsp.me ever gives you a
# plain rtsp:// link instead of the .m3u8 one, use that instead --
# real RTSP doesn't get chopped into segments like this and the
# latency should drop a lot. until then, ~5s hiccups every ~6s are
# probably close to the floor for this specific relay.
# ---

SPEED_LIMIT_MPH = 25.0
CALIBRATION_FACTOR = 0.12
AFK_SECONDS = 40.0
CONGESTION_CAR_THRESHOLD = 8
CONGESTION_SUSTAIN_SEC = 8.0
VIOLATION_COOLDOWN_SEC = 5.0
SAVE_VIOLATION_SNAPSHOTS = True
SNAPSHOT_DIR = f"{_PERSIST_DIR}/violation_snapshots"
LOG_CSV_PATH = f"{_PERSIST_DIR}/violation_log.csv"
TRACK_STALE_SECONDS = 15.0
TRACK_PRUNE_INTERVAL_SEC = 2.0
CALIBRATION_FILE = f"{_PERSIST_DIR}/zone_calibration.json"
MAX_SNAPSHOT_FILES = 2000
SNAPSHOT_CLEANUP_INTERVAL_SEC = 300.0

ROUNDABOUT_JAM_STATIONARY_SEC = 20.0
ROUNDABOUT_JAM_CAR_THRESHOLD = 2

BATUMI_TZ = timezone(timedelta(hours=4))

DETECT_IMGSZ_LEVELS = [768, 640]
DETECTION_STRIDE_MAX = 2
GOVERNOR_WINDOW = 30
GOVERNOR_LOW_FPS = 10.0
GOVERNOR_HIGH_FPS = 22.0
PANIC_LOW_FPS = 4.0
STREAM_STALE_TIMEOUT_SEC = 12.0
STALE_WARNING_REPEAT_SEC = 10.0

# --- Stream-URL health tracking (new) ---
RECONNECT_STORM_WINDOW_SEC = 90.0
RECONNECT_STORM_THRESHOLD = 6
READER_WATCHDOG_INTERVAL_SEC = 5.0

# Per-class thresholds, not one global conf value - COCO classes 1/3
# (bicycle/motorcycle) get missed constantly at this camera's distance and
# angle, so those run lower to catch more of them at the cost of some noise.
# Class 0 (person) runs higher than the rest specifically because false
# pedestrian detections are what trigger the blur toggle and afk/idle
# logging, and a shadow or a parked scooter getting tagged as a loitering
# person was noisy enough in early runs to be worth trading recall for.
CLASS_CONF = {0: 0.30, 1: 0.15, 2: 0.25, 3: 0.18, 5: 0.25, 7: 0.25}
TRACKED_CLASSES = {0: 'Human', 1: 'Scooter/Bicycle', 2: 'Car/SUV', 3: 'Scooter/Motorcycle', 5: 'Bus', 7: 'Truck'}

VEHICLE_CLASSES = {2, 5, 7}
SCOOTER_CLASSES = {1, 3}
PEDESTRIAN_CLASSES = {0}

DEFAULT_ZONES = {
    "RESTRICTED_GRASS": [
        [[240, 245], [250, 215], [270, 195], [300, 185], [350, 178], [400, 178], [450, 180], [500, 185], [530, 192], [550, 210], [558, 235], [550, 260], [535, 275], [500, 290], [450, 302], [380, 305], [310, 298], [265, 285], [245, 270], [240, 255]],
        [[510, 142], [550, 140], [600, 142], [645, 146], [675, 138], [710, 115], [750, 88], [790, 64], [818, 50], [770, 68], [720, 88], [660, 105], [600, 120], [545, 133], [520, 139]]
    ],
    "SIDEWALK": [
        [[200, 93], [250, 93], [300, 93], [350, 90], [400, 85], [450, 78], [500, 68], [550, 60], [595, 55], [613, 50], [550, 48], [500, 40], [450, 36], [400, 38], [350, 48], [300, 60], [260, 75], [225, 85]],
        [[675, 29], [720, 20], [800, 4], [850, 0], [850, 15], [810, 17], [750, 27]],
        [[395, 480], [430, 435], [500, 425], [600, 415], [700, 405], [800, 395], [850, 390], [850, 480]]
    ],
    "PARKING": [
        [[0, 115], [200, 93], [415, 103], [0, 185]]
    ]
}

def load_zones():
    source = DEFAULT_ZONES
    calibrated = False
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r") as f:
                loaded = json.load(f)
            if any(loaded.get(k) for k in ("RESTRICTED_GRASS", "SIDEWALK", "PARKING")):
                source = loaded
                calibrated = True
        except Exception:
            source = DEFAULT_ZONES
    zones = {}
    for name in ("RESTRICTED_GRASS", "SIDEWALK", "PARKING"):
        zones[name] = [np.array(poly, np.int32) for poly in source.get(name, [])]
    return zones, calibrated

ZONES, USING_CALIBRATION_FILE = load_zones()

ZONE_DRAW_COLOR = {
    "RESTRICTED_GRASS": (0, 0, 220),
    "SIDEWALK": (0, 0, 220),
    "PARKING": (200, 200, 200),
}

def expand_polygon(poly, factor=1.8):
    centroid = poly.mean(axis=0)
    return ((poly - centroid) * factor + centroid).astype(np.int32)

ROUNDABOUT_VICINITY = None
if ZONES["RESTRICTED_GRASS"]:
    _roundabout_poly = max(ZONES["RESTRICTED_GRASS"], key=cv2.contourArea)
    ROUNDABOUT_VICINITY = expand_polygon(_roundabout_poly, factor=1.8)


def check_stream_url_freshness(url):
    """Heuristic check: rtsp.me-style links embed a numeric token in the path.
    If that number looks like a Unix timestamp already in the past, the link
    is almost certainly a dead/expired session link, not a live-camera bug."""
    import re
    now = time.time()
    notices = []
    for match in re.findall(r'(\d{9,10})', url):
        val = int(match)
        # plausible unix-timestamp range: within ~3 years of now either side
        if abs(val - now) < 3 * 365 * 24 * 3600:
            age_hours = (now - val) / 3600.0
            if val < now:
                notices.append((
                    f"STREAM_URL contains a numeric token ({val}) that reads as a timestamp "
                    f"already {age_hours:.1f}h in the past. This URL may be an expired/one-time "
                    f"rtsp.me session link, not your camera. If the feed won't connect or keeps "
                    f"reconnecting, get a FRESH embed link from rtsp.me before assuming the script is broken.",
                    "WARNING"
                ))
            elif (val - now) < 3600:
                notices.append((
                    f"STREAM_URL contains a timestamp-like token ({val}) expiring in "
                    f"{(val - now)/60:.0f} min. If the stream dies around then, that's the cause.",
                    "WARNING"
                ))
    return notices

_stream_freshness_notices = check_stream_url_freshness(STREAM_URL)


class FreshestFrameReader:
    # Cheap same-connection retries before we stop bothering re-trying it
    # every loop tick (most drops are one bad/late segment, not a dead link).
    QUICK_RETRY_SLEEP = 0.08
    QUICK_RETRY_MAX_FAILS = 6
    # okay so I actually did the math on this instead of guessing again.
    # pulled the timestamps straight out of the admin log from the last
    # run and the "gap" events landed at 12:47:29, :35, :41, :47, :54,
    # 48:00, 48:06 -- that's a gap roughly every 6 seconds, basically on
    # the dot, for the whole session. that's not "the network is flaky
    # sometimes." that's a fixed cycle. this relay is almost certainly
    # publishing the live feed in ~6s HLS segments, and ffmpeg just has
    # to sit there and wait for the next one to exist before it can hand
    # us a frame. that part isn't something retry-tuning fixes, because
    # it's not our connection being slow, it's the source not having a
    # new segment ready yet.
    # what WAS actually our fault: this trigger used to be 3.0s, which
    # is less than that real 6s cycle. so we were kicking off a whole
    # second parallel connection on literally every normal segment wait,
    # before the source had even had a chance to hand over the next
    # frame on its own. that's pure overhead -- extra requests, extra
    # CPU, zero benefit, since the "stuck" connection wasn't actually
    # stuck, it just hadn't hit 6s yet. bumping this above the real
    # cycle length so we only fire a backup connection when something's
    # genuinely gone wrong, not every single time the source does its
    # normal thing.
    PARALLEL_RECONNECT_TRIGGER_SEC = 7.5
    # Give up on any one reconnect attempt after this long and let the next
    # trigger start a fresh one.
    ASYNC_RECONNECT_DEADLINE_SEC = 8.0

    def __init__(self, url, frame_w, frame_h):
        self.url = url
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.last_read_time = time.time()
        self.stopped = False
        self.reconnects = 0
        self.stream_fps = 25.0
        self._reconnect_in_progress = False

        self._open_initial()

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open_initial(self):
        try:
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if self.cap and self.cap.isOpened():
                fps_val = self.cap.get(cv2.CAP_PROP_FPS)
                if 10.0 <= fps_val <= 60.0:
                    self.stream_fps = fps_val
        except Exception:
            self.cap = None

    def _maybe_start_async_reconnect(self):
        with self.lock:
            if self._reconnect_in_progress:
                return
            self._reconnect_in_progress = True
        threading.Thread(target=self._async_reconnect_worker, daemon=True).start()

    def _async_reconnect_worker(self):
        # Opens and warms up a brand-new connection WITHOUT touching the
        # existing self.cap (which might be just fine and about to recover
        # on its own, or might be genuinely dead -- we don't need to know
        # which). This runs fully in parallel with the primary read loop, so
        # whichever connection produces a fresh frame first is simply the
        # one the render loop sees next. This is what collapses "freeze
        # while we tear down and reopen" down to "freeze only for as long as
        # the network genuinely takes to hand back a frame."
        try:
            new_cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            deadline = time.time() + self.ASYNC_RECONNECT_DEADLINE_SEC
            got_frame = False
            while time.time() < deadline and not self.stopped:
                if new_cap.isOpened():
                    try:
                        ok, frame = new_cap.read()
                    except Exception:
                        ok, frame = False, None
                    if ok and frame is not None:
                        # The new connection's probe/open step can leave a
                        # few segments already fetched and ready to decode
                        # instantly. If we swap this in as-is, the render
                        # loop ends up racing through that backlog frame by
                        # frame -- which is what looked like the video (and
                        # computed car speed) suddenly speeding up. Drain
                        # anything that's ready RIGHT NOW (cheap, no network
                        # wait) and keep only the last one.
                        drain_deadline = time.time() + 0.3
                        while time.time() < drain_deadline:
                            ok2, frame2 = new_cap.read()
                            if not ok2 or frame2 is None:
                                break
                            frame = frame2
                        resized = cv2.resize(frame, (self.frame_w, self.frame_h))
                        with self.lock:
                            self.cap = new_cap
                            self.ok = True
                            self.frame = resized
                            self.last_read_time = time.time()
                            self.reconnects += 1
                        got_frame = True
                        break
                time.sleep(0.15)
            if not got_frame:
                # Didn't pan out in time -- release our own unused capture
                # and let the next trigger spin up another attempt. We never
                # touch the *old* self.cap here, so we can't race a read
                # that's still in flight on it.
                try:
                    new_cap.release()
                except Exception:
                    pass
        finally:
            with self.lock:
                self._reconnect_in_progress = False

    def _loop(self):
        consecutive_fail = 0
        while not self.stopped:
            try:
                ok, frame = (False, None) if self.cap is None else self.cap.read()
            except Exception:
                ok, frame = False, None

            if ok and frame is not None:
                consecutive_fail = 0
                resized = cv2.resize(frame, (self.frame_w, self.frame_h))
                with self.lock:
                    self.ok = True
                    self.frame = resized
                    self.last_read_time = time.time()
            else:
                consecutive_fail += 1
                with self.lock:
                    self.ok = False
                time.sleep(self.QUICK_RETRY_SLEEP if consecutive_fail <= self.QUICK_RETRY_MAX_FAILS else 0.2)

            # Regardless of what the primary connection is doing, once it's
            # genuinely been too long since a real frame, start warming up a
            # fresh connection in parallel. This never blocks this loop --
            # it just kicks off a thread and moves on.
            if time.time() - self.last_read_time > self.PARALLEL_RECONNECT_TRIGGER_SEC:
                self._maybe_start_async_reconnect()

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def seconds_since_last_frame(self):
        with self.lock:
            return time.time() - self.last_read_time

    def get_last_frame_time(self):
        with self.lock:
            return self.last_read_time

    def is_healthy_thread(self):
        return self.thread.is_alive()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2)
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


# Calibration is click-to-draw-polygon rather than typed coordinates because
# the zone polygons in DEFAULT_ZONES below were themselves produced by this
# tool - eyeballing pixel coordinates for an 18-point sidewalk polygon by
# hand isn't realistic, and the camera's mounting angle/zoom drifts slightly
# whenever the housing gets bumped, so re-calibrating needs to be fast enough
# to actually happen instead of getting skipped.
if RUN_CALIBRATION:
    import matplotlib.pyplot as plt
    get_ipython().system('pip install ipympl -q')
    get_ipython().run_line_magic('matplotlib', 'widget')

    cap = cv2.VideoCapture(STREAM_URL)
    ret, frame = False, None
    for _ in range(25):
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.3)
    cap.release()
    if not ret or frame is None:
        raise RuntimeError('Stream offline')

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    zones_calib = {"RESTRICTED_GRASS": [], "SIDEWALK": [], "PARKING": []}
    current_points = []
    scatter_artists = []
    line_artist = [None]

    zone_picker = widgets.Dropdown(options=list(zones_calib.keys()), description='Zone:')
    status = widgets.HTML(value='Calibrator Ready')
    undo_btn = widgets.Button(description='Undo')
    finish_btn = widgets.Button(description='Finish Shape')
    clear_btn = widgets.Button(description='Clear', button_style='danger')
    save_btn = widgets.Button(description='Save', button_style='success')

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.imshow(frame_rgb)
    ax.grid(color='white', alpha=0.25, linewidth=0.5)

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        current_points.append((int(event.xdata), int(event.ydata)))
        redraw_current()

    def redraw_current():
        for a in scatter_artists:
            a.remove()
        scatter_artists.clear()
        if line_artist[0] is not None:
            line_artist[0].remove()
            line_artist[0] = None
        if current_points:
            xs = [p[0] for p in current_points]
            ys = [p[1] for p in current_points]
            scatter_artists.append(ax.scatter(xs, ys, c='red', s=30, zorder=5))
            if len(current_points) > 1:
                closed_x = xs + [xs[0]]
                closed_y = ys + [ys[0]]
                line_artist[0], = ax.plot(closed_x, closed_y, c='red', linewidth=1.5)
        fig.canvas.draw_idle()

    def on_undo(_):
        if current_points:
            current_points.pop()
            redraw_current()

    def on_finish(_):
        if len(current_points) < 3:
            return
        zones_calib[zone_picker.value].append([list(p) for p in current_points])
        current_points.clear()
        redraw_current()

    def on_clear(_):
        zones_calib[zone_picker.value] = []

    def on_save(_):
        if current_points:
            status.value = 'Finish or undo the current shape first.'
            return
        if not any(zones_calib.values()):
            status.value = 'Nothing to save yet — draw at least one zone.'
            return
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(zones_calib, f)
        if DRIVE_MOUNTED:
            status.value = f"Saved to {CALIBRATION_FILE} (on Google Drive — survives Colab restarts automatically)."
        else:
            status.value = (f"Saved to {CALIBRATION_FILE}. Drive isn't mounted, so this lives on the Colab VM's "
                             f"local disk only — click 'Download Backup' now, since a runtime restart wipes /content entirely.")

    def on_download(_):
        try:
            from google.colab import files
            files.download(CALIBRATION_FILE)
        except Exception as e:
            status.value = f"Download failed ({type(e).__name__}: {e}). Are you running this in Colab?"

    def on_restore(_):
        try:
            from google.colab import files
            uploaded = files.upload()
            for name, data in uploaded.items():
                with open(CALIBRATION_FILE, 'wb') as f:
                    f.write(data)
                status.value = f"Restored '{name}' to {CALIBRATION_FILE}. Set RUN_CALIBRATION = False and rerun this cell to use it."
                break
        except Exception as e:
            status.value = f"Restore failed ({type(e).__name__}: {e})."

    download_btn = widgets.Button(description='Download Backup', button_style='info')
    restore_btn = widgets.Button(description='Restore Backup', button_style='warning')

    fig.canvas.mpl_connect('button_press_event', on_click)
    undo_btn.on_click(on_undo)
    finish_btn.on_click(on_finish)
    clear_btn.on_click(on_clear)
    save_btn.on_click(on_save)
    download_btn.on_click(on_download)
    restore_btn.on_click(on_restore)
    display(zone_picker, widgets.HBox([undo_btn, finish_btn, clear_btn, save_btn, download_btn, restore_btn]), status)

else:
    _startup_notices = []
    _startup_notices.extend(_stream_freshness_notices)

    model = YOLO("yolov8s.pt")
    USE_HALF = False
    if torch.cuda.is_available():
        model.to('cuda')
        # torch.cuda.is_available() only proves a GPU exists, not that this
        # particular Colab GPU behaves under FP16 - on the T4/K80 rotation
        # Colab hands out, half precision has silently thrown NaNs on some
        # ops rather than raising, which is worse than just erroring. So
        # instead of trusting the flag, actually run one real frame through
        # half precision at startup and only commit to it if that succeeds.
        try:
            model.half()
            _warm = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            model.predict(_warm, imgsz=DETECT_IMGSZ_LEVELS[0], verbose=False)
            USE_HALF = True
            _startup_notices.append((f"GPU: {torch.cuda.get_device_name(0)} — running FP16.", "INFO"))
        except Exception as e:
            try:
                model.float()
            except Exception:
                pass
            USE_HALF = False
            _startup_notices.append((f"FP16 self-test failed ({type(e).__name__}: {e}) — using FP32 instead.", "WARNING"))
    else:
        _startup_notices.append(("NO GPU DETECTED. Runtime > Change runtime type > GPU (T4), then Restart session and rerun this cell. CPU will be very slow.", "CRITICAL"))

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    if not os.path.exists(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "track_id", "class", "zone", "violation_type", "details", "snapshot"])

    IS_NIGHT_MODE = False
    imgsz_level_idx = 0 if torch.cuda.is_available() else len(DETECT_IMGSZ_LEVELS) - 1
    DETECT_IMGSZ = DETECT_IMGSZ_LEVELS[imgsz_level_idx]
    detection_stride = 1

    _night_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    _color_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    _night_color_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    _gamma_lut = (np.linspace(0, 1, 256, dtype=np.float32) ** (1.0 / 2.0) * 255).astype(np.uint8)

    def gamma_brighten(img):
        return cv2.LUT(img, _gamma_lut)

    def classify_zone_with_distance(gx, gy, class_id):
        best_zone = "ROAD"
        max_dist = -9999.0
        for name in ("RESTRICTED_GRASS", "SIDEWALK", "PARKING"):
            for poly in ZONES[name]:
                dist = cv2.pointPolygonTest(poly, (float(gx), float(gy)), True)
                if dist >= 0:
                    if dist > max_dist:
                        max_dist = dist
                        best_zone = name
        return best_zone, max_dist

    def update_day_night_status(frame):
        global IS_NIGHT_MODE
        hour = datetime.now(BATUMI_TZ).hour
        if hour >= 21 or hour < 6:
            IS_NIGHT_MODE = True
        elif 9 <= hour < 18:
            IS_NIGHT_MODE = False
        else:
            small_img = cv2.resize(frame, (100, 100))
            hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
            median_v = np.median(hsv[:, :, 2])
            IS_NIGHT_MODE = median_v < 80

    def enhance_for_detection(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _night_clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def hue_to_label(h_val, s_val, v_val):
        # Value/saturation get checked before hue on purpose, not just because
        # that's the "correct" HSV order. Most vehicles crossing this camera
        # are white, silver, black, or gray, and on the compressed HLS feed
        # those all land with washed-out saturation regardless of what hue
        # channel noise says. Checking hue first misclassified a lot of
        # silver cars as green/blue during testing, since low-saturation
        # pixels have an essentially random hue.
        if v_val < 55:
            return "Black/Dark"
        if s_val < 50:
            return "White/Silver" if v_val > 155 else "Gray"

        # Below here s_val is high enough that hue is actually meaningful.
        # These band edges aren't the textbook 30/60/90/etc split - they're
        # nudged from testing against snapshots pulled off this specific feed,
        # where JPEG compression pushes reds a few degrees warm and blues a
        # few degrees toward purple compared to a clean sensor.
        if h_val < 10 or h_val > 170:
            return "Red"
        if h_val < 25:
            return "Orange"
        if h_val < 40:
            return "Yellow"
        if h_val < 85:
            return "Green"
        if h_val < 135:
            return "Blue"
        return "Purple"

    def extract_vehicle_color_sample(clean_frame, x1, y1, x2, y2):
        w, h = x2 - x1, y2 - y1
        if w < 22 or h < 18:
            return "TOO_FAR"
        margin_frac = 0.25 if w > 80 else (0.15 if w > 45 else 0.05)
        cx1, cx2 = int(x1 + w * margin_frac), int(x2 - w * margin_frac)
        cy1, cy2 = int(y1 + h * 0.35), int(y2 - h * 0.15)
        crop = clean_frame[max(0, cy1):cy2, max(0, cx1):cx2]
        if crop.size == 0:
            return None

        if IS_NIGHT_MODE:
            crop = gamma_brighten(crop)
            clahe = _night_color_clahe
            v_lo, v_hi = 16, 250
        else:
            clahe = _color_clahe
            v_lo, v_hi = 35, 245

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        crop = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        v_ch = hsv[:, :, 2]
        valid_mask = (v_ch > v_lo) & (v_ch < v_hi)
        min_pixels = 30 if IS_NIGHT_MODE else 20
        if valid_mask.sum() < min_pixels:
            return None
        pixels = crop[valid_mask].astype(np.float32)
        if len(pixels) < (15 if IS_NIGHT_MODE else 10):
            return None

        k = 2 if len(pixels) > 40 else 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)
        _, counts = np.unique(labels, return_counts=True)
        dominant_bgr = palette[np.argmax(counts)]
        hsv_color = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        return hue_to_label(*hsv_color)

    def resolve_track_color(track):
        votes = [c for c in track["color_samples"] if c]
        min_votes = 6 if IS_NIGHT_MODE else 3
        min_agreement = 0.5 if IS_NIGHT_MODE else 0.4
        if len(votes) < min_votes:
            return "Analyzing..."
        label, count = Counter(votes).most_common(1)[0]
        if count / len(votes) < min_agreement:
            return "Uncertain (Night)" if IS_NIGHT_MODE else "Analyzing..."
        return f"{label} (Night Est.)" if IS_NIGHT_MODE else label

    def get_direction(dx, dy):
        # 5px dead zone matters more than it looks like it should - without it,
        # a car sitting at the roundabout stop line flickers between whatever
        # directions its detection box jitters toward frame to frame, which
        # made the on-screen label unreadable for anything not actually moving.
        if abs(dx) < 5 and abs(dy) < 5:
            return "IDLE"

        # image y grows downward, so this is screen-space angle, not compass
        # angle - it only maps onto real N/S/E/W because the camera happens
        # to be mounted looking roughly north over the roundabout. If this
        # rig ever gets pointed at a different intersection, these labels
        # need re-deriving against that camera's actual orientation.
        angle = np.degrees(np.arctan2(dy, dx))
        compass_bins = (
            (-22.5, 22.5, "E"), (22.5, 67.5, "SE"), (67.5, 112.5, "S"),
            (112.5, 157.5, "SW"), (-67.5, -22.5, "NE"), (-112.5, -67.5, "N"),
            (-157.5, -112.5, "NW"),
        )
        for lo, hi, label in compass_bins:
            if lo <= angle < hi:
                return label
        return "W" if (angle >= 157.5 or angle < -157.5) else "UKN"

    # Video feed and admin log side by side rather than log-below-video: on
    # the Colab notebook width this project actually gets viewed at, stacking
    # them vertically pushed the log panel below the fold, which defeats the
    # point of it being live. Toggles live under the log column specifically
    # because they're admin-facing controls, not part of the feed itself.
    video_screen = widgets.Image(format='jpeg', width=750, height=420, layout=widgets.Layout(border='3px solid #2c2c2e'))
    admin_logs = widgets.Textarea(value="System Active\n", layout=widgets.Layout(width='350px', height='360px', border='2px solid #555', padding='5px'))
    stats_panel = widgets.HTML(value="Counting initialized")
    status_label = widgets.Label(value="Surveillance Live")
    heatmap_toggle = widgets.ToggleButton(value=False, description='Show Heatmap', icon='fire')
    blur_toggle = widgets.ToggleButton(value=False, description='Blur Pedestrians', icon='eye-slash')
    minimap_toggle = widgets.ToggleButton(value=True, description='Show Minimap', icon='map')
    ai_vision_toggle = widgets.ToggleButton(value=False, description='AI Vision (night)', icon='low-vision')

    log_col = widgets.VBox([admin_logs, widgets.HBox([heatmap_toggle, blur_toggle, minimap_toggle, ai_vision_toggle])])
    ui_layout = widgets.HBox([video_screen, log_col])
    display(status_label, stats_panel, ui_layout)

    severity_counter = Counter()
    violation_type_counter = Counter()

    VIOLATION_STORM_WINDOW_SEC = 60.0
    VIOLATION_STORM_THRESHOLD = 15
    VIOLATION_STORM_MUTE_SEC = 120.0
    violation_timestamps = deque()
    violation_storm_active = False
    violation_storm_until = 0.0
    suppressed_violation_count = 0

    def log_event(msg, level="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        current_logs = admin_logs.value.split('\n')
        if len(current_logs) > 60:
            current_logs = current_logs[:60]
        admin_logs.value = f"[{timestamp}] {level}: {msg}\n" + '\n'.join(current_logs)
        severity_counter[level] += 1

    _snapshot_queue = queue.Queue(maxsize=50)

    def _snapshot_worker():
        while True:
            item = _snapshot_queue.get()
            if item is None:
                break
            path, frame_to_save = item
            try:
                cv2.imwrite(path, frame_to_save)
            except Exception:
                pass
            _snapshot_queue.task_done()

    threading.Thread(target=_snapshot_worker, daemon=True).start()

    def queue_snapshot_save(path, frame_to_save):
        try:
            _snapshot_queue.put_nowait((path, frame_to_save))
        except queue.Full:
            pass

    # --- CSV writes moved off the main render loop (NEW) ---
    # Every violation used to do a synchronous open()/write() to LOG_CSV_PATH,
    # which lives on the Google-Drive FUSE mount. During a violation storm
    # (e.g. bad zone calibration flagging cars repeatedly) that turned into
    # dozens of blocking Drive writes per minute on the same thread that
    # renders frames -> that is what produced the "freeze, continue, freeze"
    # pattern. This background thread absorbs that I/O instead.
    _csv_queue = queue.Queue(maxsize=500)

    def _csv_worker():
        while True:
            row = _csv_queue.get()
            if row is None:
                break
            try:
                with open(LOG_CSV_PATH, "a", newline="") as f:
                    csv.writer(f).writerow(row)
            except Exception:
                pass
            _csv_queue.task_done()

    threading.Thread(target=_csv_worker, daemon=True).start()

    def queue_csv_row(row):
        try:
            _csv_queue.put_nowait(row)
        except queue.Full:
            pass

    def log_violation(frame, track_id, v_class, zone, vtype, details):
        global violation_storm_active, violation_storm_until, suppressed_violation_count
        now = time.time()

        if violation_storm_active:
            if now < violation_storm_until:
                suppressed_violation_count += 1
                return
            violation_storm_active = False
            log_event(f"Violation logging resumed after the storm mute ({suppressed_violation_count} "
                      f"suppressed). If this keeps happening, the zone calibration almost certainly doesn't "
                      f"match the current camera framing — recalibrate (RUN_CALIBRATION=True).", level="WARNING")
            suppressed_violation_count = 0

        violation_timestamps.append(now)
        while violation_timestamps and now - violation_timestamps[0] > VIOLATION_STORM_WINDOW_SEC:
            violation_timestamps.popleft()

        if len(violation_timestamps) > VIOLATION_STORM_THRESHOLD:
            violation_storm_active = True
            violation_storm_until = now + VIOLATION_STORM_MUTE_SEC
            violation_timestamps.clear()
            log_event(f"VIOLATION STORM: {VIOLATION_STORM_THRESHOLD}+ CRITICAL events in "
                      f"{VIOLATION_STORM_WINDOW_SEC:.0f}s. This is almost always a sign the zone calibration "
                      f"doesn't match the camera view (e.g. defaults loaded after a restart), not a genuine "
                      f"mass violation. Muting violation logging/snapshots for {VIOLATION_STORM_MUTE_SEC:.0f}s — "
                      f"detection keeps running, go recalibrate if this trips repeatedly.", level="CRITICAL")
            return

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        snap_path = ""
        if SAVE_VIOLATION_SNAPSHOTS:
            snap_path = f"{SNAPSHOT_DIR}/{track_id}_{vtype}_{int(time.time())}.jpg"
            queue_snapshot_save(snap_path, frame.copy())
        queue_csv_row([ts, track_id, v_class, zone, vtype, details, snap_path])
        violation_type_counter[vtype] += 1
        log_event(f"{vtype} | {v_class} #{track_id} | {details}", level="CRITICAL")

    for _msg, _lvl in _startup_notices:
        log_event(_msg, level=_lvl)
    if DRIVE_MOUNTED:
        log_event(f"Google Drive mounted — calibration/logs/snapshots persist at {_PERSIST_DIR} across restarts.", level="INFO")
    else:
        log_event(f"Google Drive not mounted ({_DRIVE_MOUNT_ERROR}) — using ephemeral {_PERSIST_DIR}, which a Colab "
                  f"runtime restart wipes. Re-run this cell and approve the Drive mount prompt to make calibration permanent.", level="WARNING")
    if USING_CALIBRATION_FILE:
        log_event(f"Loaded calibrated zones from {CALIBRATION_FILE}.", level="INFO")
    else:
        log_event(f"{CALIBRATION_FILE} not found — using built-in default zones. If you calibrated before, "
                  f"a Colab runtime restart wipes /content: restore your downloaded backup (RUN_CALIBRATION=True, "
                  f"click Restore Backup) then rerun this cell.", level="WARNING")
    log_event(f"Display capped at {12:.0f} FPS with adaptive JPEG quality, and duplicate HLS frames are now "
              f"skipped before detection — both target the 'lagging hard' symptom directly.", level="INFO")
    log_event("CSV writes now run on a background thread instead of blocking the render loop; GPU cache is only "
              "cleared when reserved memory actually climbs, instead of on a fixed timer.", level="INFO")
    log_event("Parallel-reconnect trigger raised to 7.5s (was 3.0s) — logs from the last run showed this feed "
              "settling into a ~6s freeze cycle, which reads like the relay's own segment length rather than a "
              "dropped connection. The old 3.0s value was tripping on every normal cycle and doing nothing "
              "useful. If freezes still happen roughly every ~6s after this, that's very likely the source, "
              "not this script — a native rtsp:// link (if rtsp.me offers one) would be the real fix for that.",
              level="INFO")

    def build_stats_html(mode_str_emoji, congestion_alerted, roundabout_jam, fps_now, governor_note):
        cls_str = " &nbsp;|&nbsp; ".join(f"{TRACKED_CLASSES.get(k, k)}: {class_counts.get(k, 0)}" for k in TRACKED_CLASSES)
        top_violations = violation_type_counter.most_common(5)
        viol_str = " &nbsp;|&nbsp; ".join(f"{k}: {v}" for k, v in top_violations) or "None yet"
        congestion_html = "CONGESTION" if congestion_alerted else "Flowing"
        jam_html = " &nbsp;|&nbsp; ROUNDABOUT GRIDLOCK" if roundabout_jam else ""
        stride_html = f" &nbsp;|&nbsp; detection every {detection_stride} frame(s)" if detection_stride > 1 else ""
        storm_html = f" &nbsp;|&nbsp; VIOLATION LOGGING MUTED (storm)" if violation_storm_active else ""
        return f"""
        <div style='font-family:monospace;font-size:12px;padding:8px;background:#111;color:#eee;border-radius:6px;line-height:1.6'>
        <b>{mode_str_emoji}</b> &nbsp;|&nbsp; Traffic: {congestion_html}{jam_html}{storm_html}
        &nbsp;|&nbsp; Active tracks: {len(tracks)} &nbsp;|&nbsp; FPS: {fps_now:.1f} ({DETECT_IMGSZ}px{governor_note}){stride_html}<br>
        <b>By class:</b> {cls_str}<br>
        <b>Top violations:</b> {viol_str}<br>
        <b>Log status:</b> CRITICAL {severity_counter['CRITICAL']} &nbsp; WARNING {severity_counter['WARNING']} &nbsp; INFO {severity_counter['INFO']}
        &nbsp;|&nbsp; Reconnects: {reader.reconnects} &nbsp;|&nbsp; GPU: {last_gpu_mem_str}
        </div>
        """

    def evaluate_violations(frame, current_time, track_id, class_id, zone, inside_dist):
        v_class = TRACKED_CLASSES.get(class_id, "Unknown")
        t = tracks[track_id]

        is_car = class_id in VEHICLE_CLASSES
        is_scooter = class_id in SCOOTER_CLASSES
        is_human = class_id in PEDESTRIAN_CLASSES

        triggered = False
        vtype = ""
        detail = ""

        # These three inside_dist cutoffs aren't the same number on purpose.
        # inside_dist is measured in pixels from the zone edge using the
        # tracked object's ground point, and how far "in" that point sits
        # before it means something is different per class:
        #   cars (6.0) - a car's ground point is the bottom-center of a wide
        #     bbox, which clips a few px into an adjacent zone constantly on
        #     any curve or camera-angle foreshortening. 6px is roughly what
        #     it takes to separate "actually on the sidewalk" from "normal
        #     bbox noise on a vehicle that's still on the road" - this is
        #     also why there's a separate WARNING_BOUNDARY tier below instead
        #     of just raising this number and losing the near-miss signal.
        #   pedestrians (1.0) - a person's footprint is small and their
        #     ground point tracks their actual feet closely, so there's no
        #     equivalent bbox-clipping noise to buffer against; a person's
        #     ground point in ROAD basically means they're in the road.
        #   scooters (2.0) - splits the difference: narrower footprint than
        #     a car but still enough bbox jitter at speed to want some buffer.
        if is_car:
            if zone == "SIDEWALK" and inside_dist >= 6.0:
                triggered = True
                vtype = "VEHICLE_ON_SIDEWALK"
                detail = "Vehicle ground-point sits well inside the sidewalk boundary"
            elif zone == "RESTRICTED_GRASS" and inside_dist >= 6.0:
                triggered = True
                vtype = "VEHICLE_ON_GRASS"
                detail = "Vehicle ground-point sits well inside the restricted grass area"

        elif is_human:
            if zone == "ROAD" and inside_dist >= 1.0:
                triggered = True
                vtype = "PEDESTRIAN_ON_ROAD"
                detail = "Pedestrian tracked walking on live lane"
            elif zone == "RESTRICTED_GRASS" and inside_dist >= 1.0:
                triggered = True
                vtype = "PEDESTRIAN_ON_GRASS"
                detail = "Pedestrian tracked inside restricted grass zone"

        elif is_scooter:
            if zone == "RESTRICTED_GRASS" and inside_dist >= 2.0:
                triggered = True
                vtype = "SCOOTER_ON_GRASS"
                detail = "Rider tracked on restricted grass"

        if triggered:
            key = f"{vtype}_{zone}"
            last = t["last_violation_log"].get(key, 0)
            if current_time - last >= VIOLATION_COOLDOWN_SEC:
                t["last_violation_log"][key] = current_time
                log_violation(frame, track_id, v_class, zone, vtype, detail)
                return "CRITICAL"
            return "CRITICAL_COOLDOWN"

        if is_car and zone in ("SIDEWALK", "RESTRICTED_GRASS") and (0.0 <= inside_dist < 6.0):
            return "WARNING_BOUNDARY"

        return "OK"

    GPU_MEMORY_CLEANUP_INTERVAL_SEC = 300.0
    GPU_RESERVED_MB_CLEAR_THRESHOLD = 3000.0
    last_gpu_mem_str = "N/A (no GPU)" if not torch.cuda.is_available() else "collecting..."

    def gpu_memory_checkpoint():
        # NEW: empty_cache() forces CUDA to synchronize, which is a real,
        # visible stall on its own. Only pay that cost if memory is actually
        # climbing toward a problem, not on a fixed 5-minute timer regardless.
        global last_gpu_mem_str
        if not torch.cuda.is_available():
            return
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        cleared = False
        if reserved_mb > GPU_RESERVED_MB_CLEAR_THRESHOLD:
            torch.cuda.empty_cache()
            cleared = True
        last_gpu_mem_str = f"{allocated_mb:.0f}MB alloc / {reserved_mb:.0f}MB reserved"
        ram_note = ""
        if _HAS_PSUTIL:
            try:
                vm = psutil.virtual_memory()
                ram_note = f" | System RAM: {vm.percent:.0f}% used ({vm.used / (1024**3):.1f}/{vm.total / (1024**3):.1f} GB)"
            except Exception:
                pass
        log_event(f"GPU memory check: {last_gpu_mem_str}{' (cache cleared)' if cleared else ''}{ram_note}.", level="INFO")

    def enforce_snapshot_cap():
        try:
            files = sorted(
                (os.path.join(SNAPSHOT_DIR, f) for f in os.listdir(SNAPSHOT_DIR)),
                key=os.path.getmtime,
            )
        except Exception:
            return
        excess = len(files) - MAX_SNAPSHOT_FILES
        if excess <= 0:
            return
        for path in files[:excess]:
            try:
                os.remove(path)
            except Exception:
                pass
        log_event(f"Snapshot housekeeping: removed {excess} oldest snapshot(s) (cap {MAX_SNAPSHOT_FILES}).", level="INFO")

    def prune_stale_tracks(current_time):
        stale = [tid for tid, t in tracks.items() if t["history"] and current_time - t["history"][-1][2] > TRACK_STALE_SECONDS]
        for tid in stale:
            del tracks[tid]

    MM_W, MM_H = 170, 100
    _mm_scale_x = MM_W / FRAME_W
    _mm_scale_y = MM_H / FRAME_H
    _mm_zone_polys = []
    for zname, polys in ZONES.items():
        for poly in polys:
            scaled = (poly.astype(np.float32) * [_mm_scale_x, _mm_scale_y]).astype(np.int32)
            _mm_zone_polys.append((zname, scaled))

    def draw_minimap(frame_resized, minimap_points):
        # 0.75 favors the minimap panel over whatever's in that corner of the
        # live feed - at 0.5/0.5 the zone colors got hard to read against
        # moving video underneath them, and this corner rarely has anything
        # in the actual feed worth seeing through anyway.
        x0, y0 = FRAME_W - MM_W - 10, FRAME_H - MM_H - 10
        panel = np.zeros((MM_H, MM_W, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)
        for zname, poly in _mm_zone_polys:
            cv2.fillPoly(panel, [poly], ZONE_DRAW_COLOR[zname])
        for gx, gy, color in minimap_points:
            mx, my = int(gx * _mm_scale_x), int(gy * _mm_scale_y)
            cv2.circle(panel, (mx, my), 2, color, -1)
        cv2.rectangle(panel, (0, 0), (MM_W - 1, MM_H - 1), (200, 200, 200), 1)
        roi = frame_resized[y0:y0 + MM_H, x0:x0 + MM_W]
        blended = cv2.addWeighted(roi, 0.25, panel, 0.75, 0)
        frame_resized[y0:y0 + MM_H, x0:x0 + MM_W] = blended

    def blur_region(frame_resized, x1, y1, x2, y2):
        # Only the top 55% of the box, not the whole bbox - that's roughly
        # head-and-shoulders on a standing pedestrian at this camera's
        # typical distance. Blurring the full box also smeared out legs/feet,
        # which made it harder to tell pedestrians from scooters/shadows in
        # the already-blurred output, defeating half the point of watching
        # the feed at all.
        h = y2 - y1
        bx1, by1, bx2, by2 = x1, y1, x2, y1 + int(h * 0.55)
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(FRAME_W, bx2), min(FRAME_H, by2)
        if bx2 <= bx1 or by2 <= by1:
            return
        roi = frame_resized[by1:by2, bx1:bx2]
        if roi.size == 0:
            return
        frame_resized[by1:by2, bx1:bx2] = cv2.GaussianBlur(roi, (25, 25), 0)

    reader = FreshestFrameReader(STREAM_URL, FRAME_W, FRAME_H)
    tracks = {}
    total_vehicles_counted = set()
    total_pedestrians_counted = set()
    class_counts = Counter()
    frame_count = 0

    heatmap_accum = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
    congestion_start_time = None
    congestion_alerted = False
    roundabout_jam_start = None
    roundabout_jam_alerted = False

    fps_history = deque(maxlen=GOVERNOR_WINDOW)
    fps_panic_history = deque(maxlen=5)
    governor_note = ""
    last_results = None
    frames_since_detection = 0
    last_prune_time = time.time()
    last_daynight_check = time.time()
    last_snapshot_cleanup = time.time()
    last_gpu_cleanup = time.time()
    last_stale_warning = 0.0
    last_reader_watchdog = time.time()
    consecutive_errors = 0

    # --- reconnect-storm / dead-URL detection (NEW) ---
    _reconnect_log = deque()          # timestamps of observed reconnect increments
    _last_seen_reconnect_count = 0
    _reconnect_storm_warned_until = 0.0

    JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    LOW_LOAD_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 55]

    DISPLAY_TARGET_FPS = 12.0
    DISPLAY_MIN_INTERVAL = 1.0 / DISPLAY_TARGET_FPS
    last_display_push = 0.0

    last_processed_frame_ts = 0.0
    _last_processed_wallclock = time.time()
    # If the gap since the last frame we actually processed is bigger than
    # this, something discontinuous happened -- either we were frozen
    # (genuine stall) or the reader just raced through a backlog of
    # already-buffered frames to catch up (which looks like fast-forward
    # video). Either way, computing "distance moved / time elapsed" across
    # that gap gives nonsense speeds, which is exactly what was showing up
    # as unrealistically fast cars. When this happens we reset tracking so
    # speed math only ever looks at genuinely back-to-back frames.
    STALL_RESET_GAP_SEC = 1.5

    _wait_start = time.time()
    while reader.frame is None and time.time() - _wait_start < 15:
        time.sleep(0.2)

    if reader.frame is not None:
        update_day_night_status(reader.frame)
    else:
        log_event("No frame received yet from STREAM_URL after 15s — will keep retrying in the background. "
                  "If this never arrives, open STREAM_URL directly in a browser/VLC to check whether the link "
                  "itself is dead before assuming it's a script bug.", level="WARNING")

    try:
        while True:
            start_time = time.time()

            # Watchdog: if the background reader thread has died outright
            # (an uncaught exception, etc.) the old code would sit frozen
            # forever with no new frames and no error either. Detect and
            # restart it instead of silently hanging.
            if start_time - last_reader_watchdog > READER_WATCHDOG_INTERVAL_SEC:
                if not reader.is_healthy_thread():
                    log_event("Reader thread died unexpectedly — restarting it now.", level="CRITICAL")
                    try:
                        reader.stop()
                    except Exception:
                        pass
                    reader = FreshestFrameReader(STREAM_URL, FRAME_W, FRAME_H)
                last_reader_watchdog = start_time

            # Reconnect-storm detection: if the reader is reconnecting over
            # and over in a short window, that's a strong signal STREAM_URL
            # itself is dead/expired rather than a transient network blip.
            if reader.reconnects > _last_seen_reconnect_count:
                for _ in range(reader.reconnects - _last_seen_reconnect_count):
                    _reconnect_log.append(start_time)
                _last_seen_reconnect_count = reader.reconnects
            while _reconnect_log and start_time - _reconnect_log[0] > RECONNECT_STORM_WINDOW_SEC:
                _reconnect_log.popleft()
            if len(_reconnect_log) >= RECONNECT_STORM_THRESHOLD and start_time > _reconnect_storm_warned_until:
                log_event(f"{len(_reconnect_log)} reconnect attempts in the last "
                          f"{RECONNECT_STORM_WINDOW_SEC:.0f}s. This looks like a dead/expired STREAM_URL, not "
                          f"a code bug — go grab a fresh embed link from rtsp.me and swap STREAM_URL.", level="CRITICAL")
                _reconnect_storm_warned_until = start_time + RECONNECT_STORM_WINDOW_SEC

            ok, frame = reader.read()
            stale_for = reader.seconds_since_last_frame()

            if not ok or frame is None:
                if stale_for > STREAM_STALE_TIMEOUT_SEC and start_time - last_stale_warning > STALE_WARNING_REPEAT_SEC:
                    log_event(f"No new frames for {stale_for:.0f}s (reconnect attempts: {reader.reconnects}). Still retrying.", level="WARNING")
                    last_stale_warning = start_time
                status_label.value = f"Reconnecting... {stale_for:.0f}s since last frame, {reader.reconnects} attempt(s)"
                time.sleep(0.05)
                continue

            frame_ts = reader.get_last_frame_time()
            if frame_ts == last_processed_frame_ts:
                time.sleep(0.01)
                continue
            last_processed_frame_ts = frame_ts

            try:
                frame_raw = frame
                current_time = time.time()
                frame_gap = current_time - _last_processed_wallclock
                _last_processed_wallclock = current_time
                frame_count += 1
                under_load = (imgsz_level_idx > 0) or (detection_stride > 1)

                if frame_gap > STALL_RESET_GAP_SEC:
                    for _t in tracks.values():
                        _t["history"].clear()
                        _t["stationary_since"] = None
                        _t["logged_afk"] = False
                        _t["logged_speed"] = False
                    log_event(f"{frame_gap:.1f}s gap before this frame (stall or catch-up) — resetting speed "
                              f"tracking so it doesn't compute a bogus speed across the gap.", level="INFO")

                if current_time - last_daynight_check > 5.0:
                    update_day_night_status(frame_raw)
                    last_daynight_check = current_time
                if current_time - last_prune_time > TRACK_PRUNE_INTERVAL_SEC:
                    prune_stale_tracks(current_time)
                    last_prune_time = current_time
                if current_time - last_snapshot_cleanup > SNAPSHOT_CLEANUP_INTERVAL_SEC:
                    enforce_snapshot_cap()
                    last_snapshot_cleanup = current_time
                if current_time - last_gpu_cleanup > GPU_MEMORY_CLEANUP_INTERVAL_SEC:
                    gpu_memory_checkpoint()
                    last_gpu_cleanup = current_time

                detection_input = enhance_for_detection(frame_raw) if IS_NIGHT_MODE else frame_raw
                color_source_frame = frame_raw
                show_ai_vision = IS_NIGHT_MODE and ai_vision_toggle.value
                frame_resized = detection_input if show_ai_vision else frame_raw

                frames_since_detection += 1
                if last_results is None or frames_since_detection >= detection_stride:
                    results = model.track(
                        detection_input, persist=True, classes=list(TRACKED_CLASSES.keys()),
                        verbose=False, conf=0.10, imgsz=DETECT_IMGSZ, tracker="bytetrack.yaml",
                        half=USE_HALF,
                    )
                    last_results = results
                    frames_since_detection = 0
                else:
                    results = last_results

                for name, polys in ZONES.items():
                    for poly in polys:
                        cv2.polylines(frame_resized, [poly], isClosed=True, color=ZONE_DRAW_COLOR[name], thickness=2)

                cars_on_road_now = 0
                roundabout_stationary_count = 0
                minimap_points = []

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    confs = results[0].boxes.conf.cpu().numpy()

                    for box, track_id, class_id, score in zip(boxes, track_ids, class_ids, confs):
                        if score < CLASS_CONF.get(class_id, 0.25):
                            continue

                        x1, y1, x2, y2 = map(int, box)
                        gx, gy = (x1 + x2) // 2, y2
                        v_class = TRACKED_CLASSES.get(class_id, "Unknown")

                        zone, inside_dist = classify_zone_with_distance(gx, gy, class_id)

                        is_pedestrian = class_id in PEDESTRIAN_CLASSES
                        is_scooter = class_id in SCOOTER_CLASSES
                        is_car = class_id in VEHICLE_CLASSES

                        if track_id not in tracks:
                            tracks[track_id] = {
                                "history": deque(maxlen=25), "color_samples": deque(maxlen=12),
                                "color": "Analyzing...", "speed": 0.0, "direction": "IDLE",
                                "stationary_since": None, "logged_afk": False, "logged_speed": False,
                                "last_violation_log": {}, "frames_seen": 0,
                            }
                            class_counts[class_id] += 1

                        t = tracks[track_id]
                        t["history"].append((gx, gy, current_time))
                        t["frames_seen"] += 1

                        status_tier = evaluate_violations(frame_resized, current_time, track_id, class_id, zone, inside_dist)

                        if is_pedestrian:
                            total_pedestrians_counted.add(track_id)
                            minimap_points.append((gx, gy, (0, 255, 255)))
                            if blur_toggle.value:
                                blur_region(frame_resized, x1, y1, x2, y2)

                            box_color = (0, 0, 255) if status_tier == "CRITICAL" else (255, 255, 0)
                            tag = " VIOLATION" if status_tier == "CRITICAL" else ""
                            label = f"{v_class} #{track_id} | {zone} | {score*100:.0f}%{tag}"
                            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(frame_resized, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)
                            continue

                        if is_scooter:
                            minimap_points.append((gx, gy, (0, 255, 100)))
                        else:
                            total_vehicles_counted.add(track_id)
                            minimap_points.append((gx, gy, (255, 120, 0)))

                        if zone == "ROAD" and is_car:
                            cars_on_road_now += 1

                        if heatmap_toggle.value:
                            cv2.circle(heatmap_accum, (gx, gy), 14, 1.0, -1)

                        trail_color = (0, 0, 255) if status_tier == "CRITICAL" else (0, 165, 255) if status_tier == "WARNING_BOUNDARY" else (0, 255, 255)
                        if len(t["history"]) > 2:
                            pts = np.array([p[:2] for p in t["history"]], np.int32).reshape((-1, 1, 2))
                            cv2.polylines(frame_resized, [pts], isClosed=False, color=trail_color, thickness=2)

                        color_sample_stride = 10 if under_load else 5
                        if not is_scooter and t["frames_seen"] % color_sample_stride == 0:
                            sample = extract_vehicle_color_sample(color_source_frame, x1, y1, x2, y2)
                            if sample == "TOO_FAR":
                                t["color"] = "Too Far"
                            else:
                                t["color_samples"].append(sample)
                                t["color"] = resolve_track_color(t)

                        if len(t["history"]) > 5:
                            old_gx, old_gy, old_time = t["history"][0]
                            dt = current_time - old_time
                            if dt > 0:
                                dx, dy = gx - old_gx, gy - old_gy
                                dist_px = np.sqrt(dx ** 2 + dy ** 2)
                                t["direction"] = get_direction(dx, dy)

                                depth_multiplier = 1.0 + (np.power((FRAME_H - gy) / FRAME_H, 1.8) * 3.5)
                                feet_moved = dist_px * CALIBRATION_FACTOR * depth_multiplier
                                mph = (feet_moved / dt) * 0.681818
                                t["speed"] = 0.8 * t["speed"] + 0.2 * mph

                                if dist_px < 7:
                                    if t["stationary_since"] is None:
                                        t["stationary_since"] = current_time
                                    else:
                                        idle_for = current_time - t["stationary_since"]
                                        if idle_for > AFK_SECONDS and not t["logged_afk"]:
                                            t["logged_afk"] = True
                                            log_event(f"Idle: {v_class} #{track_id} in {zone}", level="WARNING")
                                        if ROUNDABOUT_VICINITY is not None and idle_for >= ROUNDABOUT_JAM_STATIONARY_SEC and \
                                                cv2.pointPolygonTest(ROUNDABOUT_VICINITY, (float(gx), float(gy)), False) >= 0:
                                            roundabout_stationary_count += 1
                                else:
                                    t["stationary_since"] = None
                                    t["logged_afk"] = False

                                if t["speed"] > SPEED_LIMIT_MPH and not t["logged_speed"]:
                                    t["logged_speed"] = True
                                    log_violation(
                                        frame_resized.copy(), track_id, v_class, zone, "SPEED_FLAG",
                                        f"Est. {t['speed']:.1f} MPH (limit {SPEED_LIMIT_MPH:.0f})"
                                    )

                        v_color, v_dir, v_speed = t["color"], t["direction"], t["speed"]

                        if is_scooter:
                            label = f"{v_class} #{track_id} | {v_speed:.1f}mph | {score*100:.0f}%"
                        else:
                            label = f"#{track_id} {v_color} {v_class} | {v_dir} | {v_speed:.1f}mph | {score*100:.0f}%"

                        if status_tier == "CRITICAL":
                            border_color = (0, 0, 255)
                            label += " | VIOLATION"
                        elif status_tier == "WARNING_BOUNDARY":
                            border_color = (0, 165, 255)
                            label += " | BOUNDARY WARNING"
                        else:
                            border_color = (0, 255, 150)

                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), border_color, 2)
                        cv2.rectangle(frame_resized, (x1, y1 - 22), (x1 + len(label) * 7 + 10, y1), border_color, -1)
                        cv2.putText(frame_resized, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

                if cars_on_road_now >= CONGESTION_CAR_THRESHOLD:
                    if congestion_start_time is None:
                        congestion_start_time = current_time
                    elif current_time - congestion_start_time >= CONGESTION_SUSTAIN_SEC and not congestion_alerted:
                        congestion_alerted = True
                        log_event("Sustained congestion", level="WARNING")
                else:
                    congestion_start_time = None
                    congestion_alerted = False

                if roundabout_stationary_count >= ROUNDABOUT_JAM_CAR_THRESHOLD:
                    if roundabout_jam_start is None:
                        roundabout_jam_start = current_time
                    elif not roundabout_jam_alerted:
                        roundabout_jam_alerted = True
                        log_violation(
                            frame_resized.copy(), -1, "Multiple", "ROUNDABOUT", "ROUNDABOUT_GRIDLOCK",
                            "Roundabout flow gridlocked"
                        )
                else:
                    roundabout_jam_start = None
                    roundabout_jam_alerted = False

                if heatmap_toggle.value:
                    heatmap_accum *= 0.98
                    hm = cv2.normalize(heatmap_accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                    frame_resized = cv2.addWeighted(frame_resized, 0.7, hm_color, 0.3, 0)

                if minimap_toggle.value:
                    draw_minimap(frame_resized, minimap_points)

                mode_str_plain = "NIGHT MODE" if IS_NIGHT_MODE else "DAY MODE"
                mode_str_emoji = ("NIGHT" if IS_NIGHT_MODE else "DAY")
                batumi_clock = datetime.now(BATUMI_TZ).strftime("%H:%M")
                fps_now = 1.0 / max(time.time() - start_time, 1e-6)
                fps_history.append(fps_now)
                fps_panic_history.append(fps_now)

                panicked = False
                if len(fps_panic_history) == fps_panic_history.maxlen:
                    panic_avg = sum(fps_panic_history) / len(fps_panic_history)
                    if panic_avg < PANIC_LOW_FPS:
                        if imgsz_level_idx < len(DETECT_IMGSZ_LEVELS) - 1:
                            imgsz_level_idx += 1
                            DETECT_IMGSZ = DETECT_IMGSZ_LEVELS[imgsz_level_idx]
                            governor_note = " PANIC-DOWN"
                            log_event(f"Performance Governor PANIC: {panic_avg:.1f} FPS over last 5 frames — dropping to {DETECT_IMGSZ}px immediately.", level="CRITICAL")
                            panicked = True
                        elif detection_stride < DETECTION_STRIDE_MAX:
                            detection_stride += 1
                            governor_note = " PANIC-DOWN"
                            log_event(f"Performance Governor PANIC: {panic_avg:.1f} FPS over last 5 frames — detection every {detection_stride} frames immediately.", level="CRITICAL")
                            panicked = True
                        if panicked:
                            fps_history.clear()
                            fps_panic_history.clear()

                if not panicked and len(fps_history) == GOVERNOR_WINDOW:
                    avg_fps = sum(fps_history) / len(fps_history)
                    if avg_fps < GOVERNOR_LOW_FPS and imgsz_level_idx < len(DETECT_IMGSZ_LEVELS) - 1:
                        imgsz_level_idx += 1
                        DETECT_IMGSZ = DETECT_IMGSZ_LEVELS[imgsz_level_idx]
                        governor_note = " DOWN"
                        log_event(f"Performance Governor: dropping to {DETECT_IMGSZ}px (avg {avg_fps:.1f} FPS)", level="WARNING")
                        fps_history.clear()
                    elif avg_fps < GOVERNOR_LOW_FPS and imgsz_level_idx == len(DETECT_IMGSZ_LEVELS) - 1 and detection_stride < DETECTION_STRIDE_MAX:
                        detection_stride += 1
                        governor_note = " DOWN"
                        log_event(f"Performance Governor: running inference every {detection_stride} frames.", level="WARNING")
                        fps_history.clear()
                    elif avg_fps > GOVERNOR_HIGH_FPS and detection_stride > 1:
                        detection_stride -= 1
                        governor_note = " UP"
                        log_event(f"Performance Governor: FPS recovered ({avg_fps:.1f}) — back to detecting every frame.", level="INFO")
                        fps_history.clear()
                    elif avg_fps > GOVERNOR_HIGH_FPS and imgsz_level_idx > 0:
                        imgsz_level_idx -= 1
                        DETECT_IMGSZ = DETECT_IMGSZ_LEVELS[imgsz_level_idx]
                        governor_note = " UP"
                        log_event(f"Performance Governor: raising to {DETECT_IMGSZ}px (avg {avg_fps:.1f} FPS)", level="INFO")
                        fps_history.clear()
                    else:
                        governor_note = ""

                cv2.putText(frame_resized,
                            f"VEHICLES: {len(total_vehicles_counted)} | PEDS: {len(total_pedestrians_counted)} | {mode_str_plain} | Batumi {batumi_clock}",
                            (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_resized, f"FPS: {fps_now:.1f} | reconnects: {reader.reconnects}", (20, FRAME_H - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                if show_ai_vision:
                    cv2.putText(frame_resized, "AI VISION MODE - enhanced low-light input", (20, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

                if frame_count % 10 == 0:
                    stats_panel.value = build_stats_html(mode_str_emoji, congestion_alerted, roundabout_jam_alerted, fps_now, governor_note)

                if current_time - last_display_push >= DISPLAY_MIN_INTERVAL:
                    jpeg_params_now = LOW_LOAD_JPEG_PARAMS if under_load else JPEG_PARAMS
                    _, encoded_img = cv2.imencode('.jpg', frame_resized, jpeg_params_now)
                    video_screen.value = encoded_img.tobytes()
                    last_display_push = current_time
                status_label.value = "Live feed. Running."
                consecutive_errors = 0

            except Exception as loop_err:
                consecutive_errors += 1
                log_event(f"Frame error ({type(loop_err).__name__}): {loop_err}", level="WARNING")
                if consecutive_errors >= 10:
                    log_event("10+ consecutive frame errors. Check STREAM_URL, GPU status, and package versions.", level="CRITICAL")
                    consecutive_errors = 0

            elapsed_time = time.time() - start_time
            time_per_frame = 1.0 / reader.stream_fps
            if elapsed_time < time_per_frame:
                time.sleep(time_per_frame - elapsed_time)

    except KeyboardInterrupt:
        status_label.value = "Stopped."
    finally:
        reader.stop()
        print(f"Session summary: {len(total_vehicles_counted)} vehicles, {len(total_pedestrians_counted)} pedestrians tracked, {sum(violation_type_counter.values())} violations logged.")
        print(f"Violation log: {LOG_CSV_PATH} | Snapshots: {SNAPSHOT_DIR}")
