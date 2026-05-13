# Multi-Frame Object Tracking with Kalman Filtering
 
A multi-object tracker built from scratch in Python, using Kalman filtering for state estimation and the Hungarian algorithm for data association, built on top of a YOLOv11 detection pipeline applied to the KITTI dataset.
 
![Tracking Demo](output/tracking0005.gif)

---
## Overview
 
Multi-object tracking (MOT) is a core component of any autonomous driving perception stack. Detection alone tells you where objects are — tracking tells you *which* object is which across time, giving the system continuity of identity and the ability to predict future positions.
 
This implementation builds tracking from first principles on top of an existing YOLOv11 detector. The tracker maintains a persistent state estimate for each detected object across frames, handling the noise inherent in bounding box detections, missed detections due to occlusion, and the data association problem of matching detections to existing tracks.
 
The pipeline introduces no external tracking libraries. The Kalman filter, cost matrix construction, and Hungarian matching are implemented directly using NumPy and SciPy — producing a clean, interpretable system that shows the full predict/update cycle at work.

---
 
## Mathematical Foundation
 
### State Representation
 
Each tracked object is represented by a 4-dimensional state vector:
 
$$
\mathbf{x} = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}
$$

where $(x, y)$ is the bounding box center in pixel space and $(v_x, v_y)$ is the estimated velocity in pixels per frame. Position is directly observable from YOLO detections; velocity is inferred by the filter from successive measurements.
 
### Constant Velocity Motion Model
 
State propagation uses a constant velocity model with timestep $\Delta t = 0.1$s (10Hz KITTI frame rate):
 
$$
\mathbf{x}_{t} = F\mathbf{x}_{t-1}
$$

$$
F = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\\\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$
 
### Kalman Predict Step
 
The predict step propagates both the state estimate and its uncertainty forward:
 
$$
\mathbf{x}_{t|t-1} = F\mathbf{x}_{t-1|t-1}
$$

$$
P_{t|t-1} = FP_{t-1|t-1}F^\top + Q
$$
 
where $P$ is the $4 \times 4$ covariance matrix representing uncertainty in the state estimate, and $Q$ is the process noise matrix — a scaled identity matrix parameterized per object class reflecting how unpredictably each object type moves.
 
### Observation Model
 
YOLO detections are 2D bounding box centers — a subset of the full 4D state. The observation matrix $H$ projects from state space to measurement space:
 
$$
H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}
$$

### Kalman Update Step
 
When a detection is associated with a track, the update step fuses the predicted state with the measurement:
 
$$
\mathbf{y} = \mathbf{z} - H\mathbf{x}_{t|t-1} \quad \text{(residual)}
$$

$$
K = P_{t|t-1}H^\top(HP_{t|t-1}H^\top + R)^{-1} \quad \text{(Kalman gain)}
$$

$$
\mathbf{x}_{t|t} = \mathbf{x}_{t|t-1} + K\mathbf{y}
$$

$$
P_{t|t} = (I - KH)P_{t|t-1}
$$

where $R$ is the $2 \times 2$ measurement noise matrix reflecting YOLO bounding box jitter, and $K$ is the Kalman gain — the dynamic weight that shifts trust between prediction and measurement based on their relative uncertainties. When prediction uncertainty is high, $K$ weights the measurement more heavily; when measurement noise is high, the prediction dominates.
 
The posterior covariance $P_{t|t}$ is always lower than either $P_{t|t-1}$ or $R$ alone — fusing two independent sources of information strictly reduces uncertainty.
 
### Data Association — Hungarian Algorithm
 
Each frame, detections from YOLO must be matched to existing tracks. This is formulated as a linear assignment problem: given $M$ tracks and $N$ detections, find the optimal one-to-one pairing that minimizes total assignment cost.
 
The cost matrix $C \in \mathbb{R}^{M \times N}$ is built using Euclidean distance between predicted track positions and detection centers:
 
$$
C_{ij} = \|\hat{\mathbf{x}}_i^{(pos)} - \mathbf{z}_j\|_2
$$

The Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`) solves this in $O(n^3)$ time. Pairs exceeding a distance threshold are rejected and treated as unmatched, preventing implausible long-range associations.

---
 
## Implementation Details
 
### Two-Layer Architecture
 
The system is split into two classes:
 
`KalmanTracker` manages the state of a single tracked object — holding the state vector, covariance matrix, motion model, and a position history deque for tail visualization. It exposes `predict()` and `update(measurement)` methods implementing the equations above.
 
`KalmanManager` manages the full list of active trackers across frames. Each call to `updateFrame(detections, labels)` runs the complete pipeline: predict all trackers, build the cost matrix, run Hungarian matching, apply threshold gating, handle births and deaths, and coast unmatched tracks.
 
### Per-Class Process Noise
 
Different object classes have fundamentally different motion characteristics. A parked bus changes position slowly and predictably; a motorcycle can accelerate, brake, and lane-change erratically. Process noise $Q$ is parameterized per class:
 
| Class | Process Noise ($\sigma^2$) | Rationale |
|---|---|---|
| Person | 70 | Slow, moderate unpredictability |
| Bicycle | 80 | Faster than pedestrian, road-constrained |
| Car | 110 | Fast, but road-constrained trajectory |
| Motorcycle | 120 | Fast and most erratic of vehicle classes |
| Bus | 100 | Large, slow to accelerate |
| Truck | 100 | Similar to bus |
 
### Track Lifecycle
 
**Birth** — a detection with no matching track spawns a new `KalmanTracker`, initialized with the detection position, zero velocity, and a high initial covariance ($Q \cdot I_4$) reflecting maximum uncertainty at birth.
 
**Update** — matched detections run the full Kalman update, resetting the coast counter.
 
**Coast** — unmatched tracks continue predicting forward without measurement. Covariance grows each predict step, reflecting increasing uncertainty. Coasting handles occlusion and missed detections without immediately destroying track identity.
 
**Death** — tracks coasting beyond the configured threshold are pruned from the active list.

---
 
## Results
 
Results were generated using KITTI sequence `2011_09_26_drive_0011`.

![Tracking Output](output/tracking0011.gif)
*Tracked objects with persistent IDs and trajectory tails overlaid on KITTI frames. Yellow dots show the position history of each active track.*

---

## Installation
 
Python 3.11 required.
 
```bash
pip install pykitti open3d opencv-python numpy ultralytics scipy torch
```
 
### Dataset Setup
 
Download the KITTI raw dataset from https://www.cvlibs.net/datasets/kitti/raw_data.php.
Download the synced+rectified data and calibration files for sequence `2011_09_26_drive_0011`.
 
Organize the data as follows:
 
```
data/
└── kitti/
    └── 2011_09_26/
        ├── calib_cam_to_cam.txt
        ├── calib_imu_to_velo.txt
        ├── calib_velo_to_cam.txt
        └── 2011_09_26_drive_0011_sync/
            ├── image_02/
            └── velodyne_points/
```

---
 
## Usage
 
```bash
python main.py
```
 
Output is written to `fullDrive.mp4` in the project root.
 
Key parameters in `main.py`:
 
| Parameter | Default | Description |
|---|---|---|
| `confThreshold` | 0.5 | YOLO detection confidence threshold |
| `distThreshold` | 100 | Max pixel distance for valid track-detection match |
| `coastThreshold` | 15 | Frames a track can coast before deletion |
 
Key parameters in `kalman_manager.py`:
 
| Parameter | Description |
|---|---|
| `measureVariance` | YOLO bounding box noise (pixels²) |
| `classVariance` | Per-class process noise dict |

---
 
## Related Projects
 
This is the fourth project in a progression of AV perception systems built on the KITTI dataset:
 
1. **[Occupancy Grid Mapping](https://github.com/gsactown30/OccupancyGridMapping)** — Bayesian probabilistic mapping with log-odds updates and Bresenham ray casting
2. **[LiDAR Camera Sensor Fusion](https://github.com/gsactown30/KITTI-lidar-camera-sensor-fusion)** — Full projection pipeline, dense depth completion, RGB-colored 3D point clouds
3. **[KITTI 2D Object Detection and 3D Localization](https://github.com/gsactown30/KITTI-2D-object-detection-and-3D-localization)** — YOLO11 detection, LiDAR depth sampling, back-projection into 3D camera coordinates
4. **[Multi-Frame Object Tracking with Kalman Filtering](https://github.com/gsactown30/multi-frame-object-tracking-with-kalman-filtering)** — Multi-object tracking from scratch using Kalman filtering and Hungarian matching on KITTI
---
 
## Future Work
 
- **IoU-based matching** — replacing Euclidean distance in the cost matrix with Intersection over Union would make matching more robust to scale variation and provide a natural threshold in normalized units.
- **Appearance features** — augmenting the cost matrix with CNN embedding distance would reduce identity swaps when tracks cross.
- **Adaptive process noise** — current per-class noise is fixed. Estimating noise online from residual statistics (innovations covariance) would allow the filter to adapt to scene-specific dynamics.
- **3D state vector** — back-projecting detections to 3D using the existing LiDAR pipeline from Project 3 would allow tracking in metric space rather than pixel space, allowing increased tracking accuracy and eliminating scale ambiguity as objects move toward or away from the camera.
---
 
## References
 
- Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Bewley, A., Ge, Z., Ott, L., Ramos, F., Upcroft, B. (2016). Simple Online and Realtime Tracking. *ICIP 2016*. (SORT)
- Geiger, A., Lenz, P., Stiller, C., Urtasun, R. (2013). Vision meets Robotics: The KITTI Dataset. *International Journal of Robotics Research*.
- Kuhn, H.W. (1955). The Hungarian Method for the Assignment Problem. *Naval Research Logistics Quarterly*, 2(1–2), 83–97.
- Ultralytics YOLOv11 Documentation. https://docs.ultralytics.com
- SciPy `linear_sum_assignment`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html