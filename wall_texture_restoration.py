import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation

# 선택: 바닥/천장 삼각분할
try:
    from mapbox_earcut import triangulate_float32
except Exception:
    triangulate_float32 = None


@dataclass
class AlignmentTransform:
    """structure_detection.py align_entire_scene()의 변환을 그대로 저장/재사용"""
    centroid: np.ndarray  # (3,)
    R: np.ndarray         # (3,3)
    rotation_angle_rad: float = 0.0

    def apply_to_point(self, p: np.ndarray) -> np.ndarray:
        return (self.R @ (p - self.centroid).reshape(3, 1)).reshape(3)

    def apply_to_pose(self, R_wc: np.ndarray, C_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R_wc_al = self.R @ R_wc
        C_al = self.R @ (C_w - self.centroid)
        return R_wc_al, C_al


@dataclass
class WallSpec:
    wall_id: int
    p1_xz: np.ndarray
    p2_xz: np.ndarray
    width_m: float
    height_m: float
    floor_y: float
    ceiling_y: float

    def frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p1 = np.array([self.p1_xz[0], self.floor_y, self.p1_xz[1]], dtype=np.float32)
        p2 = np.array([self.p2_xz[0], self.floor_y, self.p2_xz[1]], dtype=np.float32)
        t = p2 - p1
        t[1] = 0.0
        t_norm = np.linalg.norm(t) + 1e-8
        u_hat = t / t_norm
        v_hat = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        n_hat = np.cross(v_hat, u_hat)
        n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-8)
        return p1, u_hat, v_hat, n_hat


class WallTextureRestorerEnhanced:
    """
    ✨ Enhanced version with improved floor texture quality
    
    개선 사항:
    1. Multi-pass floor baking (거리 임계값 점진적 완화)
    2. View angle weighting (바닥을 잘 보는 각도에 가중치)
    3. Adaptive confidence threshold
    4. Better hole filling strategy
    """

    def __init__(
        self,
        dataset_path: str,
        detected_dir: str = "room_detected_results",
        recon_json: str = "room_detected_results/room_dimensions.json",
        out_dir: str = "room_wall_textures_out",
    ):
        self.dataset_path = Path(dataset_path)
        self.detected_dir = Path(detected_dir)
        self.recon_json_path = Path(recon_json)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._load_sensor_data()
        self._load_detection_outputs()
        self._load_wall_specs()

    def _load_sensor_data(self):
        cam_path = self.dataset_path / "camera_matrix.csv"
        odo_path = self.dataset_path / "odometry.csv"
        if not cam_path.exists():
            raise FileNotFoundError(f"camera_matrix.csv not found: {cam_path}")
        if not odo_path.exists():
            raise FileNotFoundError(f"odometry.csv not found: {odo_path}")

        self.K = np.loadtxt(str(cam_path), delimiter=",").astype(np.float32)
        self.odometry = pd.read_csv(str(odo_path))
        self.odometry.columns = self.odometry.columns.str.strip()

        self.depth_files = sorted((self.dataset_path / "depth").glob("*.png"))
        self.conf_files = sorted((self.dataset_path / "confidence").glob("*.png"))
        self.rgb_path = self.dataset_path / "rgb.mp4"

        if not self.depth_files:
            raise FileNotFoundError(f"No depth PNGs in: {self.dataset_path / 'depth'}")
        if not self.rgb_path.exists():
            raise FileNotFoundError(f"rgb.mp4 not found: {self.rgb_path}")

        depth0 = cv2.imread(str(self.depth_files[0]), cv2.IMREAD_UNCHANGED)
        self.depth_h, self.depth_w = depth0.shape[:2]

        cap = cv2.VideoCapture(str(self.rgb_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.rgb_path}")
        self.rgb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.rgb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        sx = self.depth_w / max(1, self.rgb_w)
        sy = self.depth_h / max(1, self.rgb_h)
        self.fx = float(self.K[0, 0] * sx)
        self.fy = float(self.K[1, 1] * sy)
        self.cx = float(self.K[0, 2] * sx)
        self.cy = float(self.K[1, 2] * sy)

        u = np.arange(self.depth_w, dtype=np.float32)
        v = np.arange(self.depth_h, dtype=np.float32)
        self.U, self.V = np.meshgrid(u, v)

        print(f"✅ Sensor loaded: depth={self.depth_w}x{self.depth_h}, video={self.rgb_w}x{self.rgb_h}")

    def _load_detection_outputs(self):
        pkl_path = self.detected_dir / "walls_data.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"walls_data.pkl not found: {pkl_path}")

        import pickle
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.floor_y = float(data["floor_height"])
        self.ceiling_y = float(data["ceiling_height"])
        self.room_height = float(data.get("room_height", self.ceiling_y - self.floor_y))

        alignment = data.get("alignment", None)
        if alignment is None:
            raise ValueError("walls_data.pkl에 alignment 정보가 없습니다.")

        centroid = np.array(alignment["centroid"], dtype=np.float32).reshape(3)
        R = np.array(alignment["R"], dtype=np.float32).reshape(3, 3)
        rot = float(alignment.get("rotation_angle_rad", 0.0))
        self.align = AlignmentTransform(centroid=centroid, R=R, rotation_angle_rad=rot)

        print("✅ Detection outputs loaded (with alignment).")

    def _load_wall_specs(self):
        if not self.recon_json_path.exists():
            raise FileNotFoundError(f"{self.recon_json_path} not found.")
        js = json.loads(self.recon_json_path.read_text(encoding="utf-8"))

        walls = js.get("walls", [])
        if not walls:
            raise ValueError("room_dimensions.json에 walls 정보가 비어있습니다.")

        height_m = float(js["room_summary"]["room_height_m"])
        self.wall_specs: List[WallSpec] = []
        for w in walls:
            wid = int(w["wall_id"])
            p1 = np.array(w["start_point"], dtype=np.float32)
            p2 = np.array(w["end_point"], dtype=np.float32)
            width_m = float(w["width_m"])
            self.wall_specs.append(
                WallSpec(
                    wall_id=wid,
                    p1_xz=p1,
                    p2_xz=p2,
                    width_m=width_m,
                    height_m=height_m,
                    floor_y=self.floor_y,
                    ceiling_y=self.ceiling_y,
                )
            )
        print(f"✅ Loaded {len(self.wall_specs)} wall specs from room_dimensions.json.")

    def _iter_video_frames(self):
        cap = cv2.VideoCapture(str(self.rgb_path))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1
        cap.release()

    # ==================== Wall Baking (기존 코드 유지) ====================
    def bake_wall_textures(
        self,
        ppm: int = 256,
        sample_rate: int = 3,
        max_frames: Optional[int] = None,
        confidence_threshold: int = 2,
        plane_dist_th_m: float = 0.03,
        save_intermediate: bool = True,
        default_color: List[int] = [240, 240, 245],  # 🔥 NEW: 기본 벽 색상
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """벽 텍스처 베이킹 + 🔥 검은 벽면 방지"""
        buffers = {}
        for w in self.wall_specs:
            w_px = int(max(8, round(w.width_m * ppm)))
            h_px = int(max(8, round(w.height_m * ppm)))
            acc = np.zeros((h_px * w_px, 3), dtype=np.float32)
            cnt = np.zeros((h_px * w_px,), dtype=np.float32)
            buffers[w.wall_id] = {
                "w_px": w_px,
                "h_px": h_px,
                "acc": acc,
                "cnt": cnt,
            }

        used = 0
        for idx, rgb in self._iter_video_frames():
            if idx >= len(self.depth_files) or idx >= len(self.odometry):
                break
            if idx % sample_rate != 0:
                continue
            if max_frames is not None and used >= max_frames:
                break

            depth = cv2.imread(str(self.depth_files[idx]), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            depth_m = depth.astype(np.float32) / 1000.0

            if self.conf_files:
                conf = cv2.imread(str(self.conf_files[idx]), cv2.IMREAD_GRAYSCALE)
                if conf is None:
                    valid = (depth_m > 0.1)
                else:
                    valid = (conf >= confidence_threshold) & (depth_m > 0.1)
            else:
                valid = (depth_m > 0.1)

            if valid.sum() < 300:
                continue

            rgb_small = cv2.resize(rgb, (self.depth_w, self.depth_h), interpolation=cv2.INTER_AREA)
            rgb_small = rgb_small.astype(np.float32)

            row = self.odometry.iloc[idx]
            C = np.array([row["x"], row["y"], row["z"]], dtype=np.float32)
            quat = np.array([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float32)
            R_wc = Rotation.from_quat(quat).as_matrix().astype(np.float32)

            R_wc_al, C_al = self.align.apply_to_pose(R_wc, C)

            vv, uu = np.where(valid)
            z = depth_m[vv, uu]
            x = (uu.astype(np.float32) - self.cx) * z / self.fx
            y = (vv.astype(np.float32) - self.cy) * z / self.fy
            P_cam = np.stack([x, y, z], axis=1)

            P_w = (R_wc_al @ P_cam.T).T + C_al.reshape(1, 3)
            cols = rgb_small[vv, uu]

            for w in self.wall_specs:
                origin, u_hat, v_hat, n_hat = w.frame()
                d = -float(np.dot(n_hat, origin))

                dist = (P_w @ n_hat.reshape(3, 1)).reshape(-1) + d
                dist = np.abs(dist)
                m = dist < plane_dist_th_m
                if m.sum() < 50:
                    continue

                Pw = P_w[m]
                Cw = cols[m]

                rel = Pw - origin.reshape(1, 3)
                u = rel @ u_hat.reshape(3, 1)
                v = rel @ v_hat.reshape(3, 1)
                u = u.reshape(-1)
                v = v.reshape(-1)

                inside = (u >= 0.0) & (u <= w.width_m) & (v >= 0.0) & (v <= w.height_m)
                if inside.sum() < 20:
                    continue

                u = u[inside]
                v = v[inside]
                Cw = Cw[inside]

                w_px = buffers[w.wall_id]["w_px"]
                h_px = buffers[w.wall_id]["h_px"]
                ix = np.floor(u * ppm).astype(np.int32)
                iy = np.floor((w.height_m - v) * ppm).astype(np.int32)
                ix = np.clip(ix, 0, w_px - 1)
                iy = np.clip(iy, 0, h_px - 1)
                flat = iy * w_px + ix

                np.add.at(buffers[w.wall_id]["acc"], flat, Cw)
                np.add.at(buffers[w.wall_id]["cnt"], flat, 1.0)

            used += 1
            if used % 20 == 0:
                print(f"  wall baking... used_frames={used} (last idx={idx})")

        results: Dict[int, Dict[str, np.ndarray]] = {}
        for wid, b in buffers.items():
            w_px, h_px = b["w_px"], b["h_px"]
            cnt = b["cnt"]
            acc = b["acc"]
            mask = (cnt.reshape(h_px, w_px) <= 0.0).astype(np.uint8) * 255
            
            # 🔥 개선: 기본 색상으로 초기화 (검은색 방지)
            img = np.tile(default_color, (h_px * w_px, 1)).astype(np.float32)
            nz = cnt > 0
            img[nz] = acc[nz] / cnt[nz].reshape(-1, 1)
            img = img.reshape(h_px, w_px, 3).astype(np.uint8)

            results[wid] = {"baked": img, "mask": mask, "count": cnt.reshape(h_px, w_px)}

            if save_intermediate:
                cv2.imwrite(str(self.out_dir / f"wall_{wid:02d}_baked.png"), img)
                cv2.imwrite(str(self.out_dir / f"wall_{wid:02d}_mask.png"), mask)

        print(f"✅ Wall baking done. outputs in: {self.out_dir}")
        return results

    # ==================== 🔥 Enhanced Floor Baking ====================
    @staticmethod
    def _points_in_poly(x: np.ndarray, z: np.ndarray, poly_xz: np.ndarray) -> np.ndarray:
        """Vectorized point-in-polygon test"""
        poly = np.asarray(poly_xz, dtype=np.float32)
        if len(poly) < 3:
            return np.zeros_like(x, dtype=bool)

        x = x.astype(np.float32)
        z = z.astype(np.float32)

        x0 = poly[:, 0]
        z0 = poly[:, 1]
        x1 = np.roll(x0, -1)
        z1 = np.roll(z0, -1)

        cond = ((z0 > z[:, None]) != (z1 > z[:, None]))
        x_int = x0 + (z[:, None] - z0) * (x1 - x0) / (z1 - z0 + 1e-12)
        crosses = cond & (x[:, None] < x_int)
        inside = np.sum(crosses, axis=1) % 2 == 1
        return inside

    def _floor_bounds(self) -> Tuple[float, float, float, float, np.ndarray]:
        corners_xz = np.stack([w.p1_xz for w in self.wall_specs], axis=0).astype(np.float32)
        min_x = float(corners_xz[:, 0].min())
        max_x = float(corners_xz[:, 0].max())
        min_z = float(corners_xz[:, 1].min())
        max_z = float(corners_xz[:, 1].max())
        return min_x, max_x, min_z, max_z, corners_xz

    def _compute_floor_view_weight(self, R_wc_al: np.ndarray, C_al: np.ndarray) -> float:
        """
        바닥을 얼마나 잘 보는 시점인지 가중치 계산
        - 카메라가 바닥을 향하고 있을수록 높은 가중치
        - 높이가 낮을수록 (바닥에 가까울수록) 높은 가중치
        
        Returns: 0.0 ~ 2.0
        """
        # 카메라 Z축 (광축) 방향
        cam_z_axis = R_wc_al[:, 2]  # (3,)
        
        # 바닥 법선 (up vector)
        floor_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # 카메라가 아래를 향하는 정도 (dot product)
        # -1 (완전히 아래), 0 (수평), 1 (완전히 위)
        view_dot = float(np.dot(cam_z_axis, floor_normal))
        
        # 아래를 향할수록 가중치 증가 (음수를 양수로)
        angle_weight = np.clip(-view_dot, 0.0, 1.0)
        
        # 높이 기반 가중치 (바닥에 가까울수록 좋음)
        height = C_al[1] - self.floor_y
        # 0.3m ~ 2.0m 범위를 0.5 ~ 1.5로 매핑
        height_weight = np.clip(1.5 - height / 3.0, 0.5, 1.5)
        
        # 조합 (각도 가중치를 더 중요하게)
        total_weight = angle_weight * 1.5 + height_weight * 0.5
        
        return float(np.clip(total_weight, 0.1, 2.0))

    def bake_floor_texture_enhanced(
        self,
        ppm: int = 256,
        sample_rate: int = 2,
        max_frames: Optional[int] = None,
        confidence_threshold: int = 1,
        multi_pass: bool = True,
        save_intermediate: bool = True,
        weight_threshold: float = 0.5,  # 🔥 NEW: 가중치 임계값
    ) -> Dict[str, np.ndarray]:
        """
        🔥 Enhanced floor texture baking + 밝기 개선
        
        개선사항:
        1. Multi-pass baking (거리 임계값을 점진적으로 완화)
        2. View angle weighting (바닥을 잘 보는 각도에 가중치)
        3. Weighted averaging (단순 평균 대신 가중 평균)
        4. 🔥 Weight threshold (어두운 프레임 제외)
        """
        print("\n" + "="*60)
        print("🏠 IMPROVED Floor Texture Baking")
        print("="*60)
        print(f"   🔥 Weight threshold: {weight_threshold} (어두운 프레임 필터링)")
        
        min_x, max_x, min_z, max_z, corners_xz = self._floor_bounds()
        width_m = max_x - min_x
        depth_m = max_z - min_z

        w_px = int(max(16, round(width_m * ppm)))
        h_px = int(max(16, round(depth_m * ppm)))

        # Multi-pass를 위한 거리 임계값
        if multi_pass:
            distance_thresholds = [0.02, 0.04, 0.08]  # 2cm → 4cm → 8cm
        else:
            distance_thresholds = [0.03]

        # 누적 버퍼 (가중치 포함)
        acc = np.zeros((h_px * w_px, 3), dtype=np.float32)
        weights = np.zeros((h_px * w_px,), dtype=np.float32)

        for pass_idx, dist_th in enumerate(distance_thresholds):
            print(f"\n📐 Pass {pass_idx + 1}/{len(distance_thresholds)}: distance threshold = {dist_th*100:.1f}cm")
            
            used = 0
            for idx, rgb in self._iter_video_frames():
                if idx >= len(self.depth_files) or idx >= len(self.odometry):
                    break
                if idx % sample_rate != 0:
                    continue
                if max_frames is not None and used >= max_frames:
                    break

                depth = cv2.imread(str(self.depth_files[idx]), cv2.IMREAD_UNCHANGED)
                if depth is None:
                    continue
                depth_m_img = depth.astype(np.float32) / 1000.0

                if self.conf_files:
                    conf = cv2.imread(str(self.conf_files[idx]), cv2.IMREAD_GRAYSCALE)
                    if conf is None:
                        valid = (depth_m_img > 0.1)
                    else:
                        # Pass 2+ 에서는 confidence를 더 낮춤
                        conf_th = confidence_threshold if pass_idx == 0 else max(0, confidence_threshold - 1)
                        valid = (conf >= conf_th) & (depth_m_img > 0.1)
                else:
                    valid = (depth_m_img > 0.1)

                if valid.sum() < 300:
                    continue

                rgb_small = cv2.resize(rgb, (self.depth_w, self.depth_h), interpolation=cv2.INTER_AREA)
                rgb_small = rgb_small.astype(np.float32)

                row = self.odometry.iloc[idx]
                C = np.array([row["x"], row["y"], row["z"]], dtype=np.float32)
                quat = np.array([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float32)
                R_wc = Rotation.from_quat(quat).as_matrix().astype(np.float32)

                R_wc_al, C_al = self.align.apply_to_pose(R_wc, C)

                # 🔥 시점 가중치 계산
                view_weight = self._compute_floor_view_weight(R_wc_al, C_al)
                
                # 🔥 가중치 필터링 (어두운 프레임 제외)
                if view_weight < weight_threshold:
                    continue  # 너무 낮은 가중치는 무시

                vv, uu = np.where(valid)
                z = depth_m_img[vv, uu]
                x = (uu.astype(np.float32) - self.cx) * z / self.fx
                y = (vv.astype(np.float32) - self.cy) * z / self.fy
                P_cam = np.stack([x, y, z], axis=1)
                P_w = (R_wc_al @ P_cam.T).T + C_al.reshape(1, 3)

                # 바닥 평면 근처 필터
                dy = np.abs(P_w[:, 1] - self.floor_y)
                m = dy < dist_th
                if m.sum() < 50:
                    continue

                Pw = P_w[m]
                Cw = rgb_small[vv, uu][m]

                # (x,z) -> pixel
                px = (Pw[:, 0] - min_x) * ppm
                py = (max_z - Pw[:, 2]) * ppm
                ix = np.floor(px).astype(np.int32)
                iy = np.floor(py).astype(np.int32)

                inside_aabb = (ix >= 0) & (ix < w_px) & (iy >= 0) & (iy < h_px)
                if inside_aabb.sum() < 20:
                    continue

                ix = ix[inside_aabb]
                iy = iy[inside_aabb]
                Cw = Cw[inside_aabb]

                flat = iy * w_px + ix
                
                # 🔥 가중치 적용 (거리가 가까울수록, 좋은 시점일수록 높은 가중치)
                point_weights = np.full(len(flat), view_weight, dtype=np.float32)
                
                np.add.at(acc, flat, Cw * point_weights[:, None])
                np.add.at(weights, flat, point_weights)

                used += 1
                if used % 20 == 0:
                    print(f"    frames={used} (idx={idx}), view_weight={view_weight:.2f}", end='\r')
            
            print(f"\n    ✅ Pass {pass_idx + 1} done: {used} frames used")

        # 가중 평균
        img = np.zeros((h_px * w_px, 3), dtype=np.float32)
        nz = weights > 0
        img[nz] = acc[nz] / weights[nz].reshape(-1, 1)
        img = img.reshape(h_px, w_px, 3).astype(np.uint8)

        # 폴리곤 마스크
        gx = np.linspace(min_x, max_x, w_px, dtype=np.float32)
        gz = np.linspace(max_z, min_z, h_px, dtype=np.float32)
        GX, GZ = np.meshgrid(gx, gz)
        inside_poly = self._points_in_poly(GX.reshape(-1), GZ.reshape(-1), corners_xz)
        inside_poly = inside_poly.reshape(h_px, w_px)

        mask = (weights.reshape(h_px, w_px) <= 0.0) | (~inside_poly)
        mask = mask.astype(np.uint8) * 255

        # 🔥 통계 출력 (버그 수정)
        hole_pixels = (mask > 0).sum()  # 255가 아니라 hole 개수
        total_pixels = mask.size
        coverage = 100.0 * (1.0 - hole_pixels / total_pixels)
        print(f"\n📊 Floor Coverage: {coverage:.1f}%")
        print(f"   - Total pixels: {total_pixels:,}")
        print(f"   - Observed pixels: {int((weights > 0).sum()):,}")
        print(f"   - Hole pixels: {hole_pixels:,}")

        if save_intermediate:
            cv2.imwrite(str(self.out_dir / "floor_baked.png"), img)
            cv2.imwrite(str(self.out_dir / "floor_mask.png"), mask)
            
            # 가중치 히트맵 저장 (디버깅용)
            weight_vis = np.clip(weights.reshape(h_px, w_px) / weights.max() * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(self.out_dir / "floor_weight_heatmap.png"), weight_vis)

        print(f"✅ Enhanced floor baking done. outputs in: {self.out_dir}")
        return {
            "baked": img,
            "mask": mask,
            "count": weights.reshape(h_px, w_px),
            "bounds": np.array([min_x, max_x, min_z, max_z], dtype=np.float32),
            "corners_xz": corners_xz,
        }

    # ==================== Inpainting ====================
    def inpaint_holes_opencv(
        self,
        baked: np.ndarray,
        mask: np.ndarray,
        inpaint_radius: int = 5,
        dilate_iter: int = 2,
    ) -> np.ndarray:
        """OpenCV TELEA 인페인팅"""
        m = mask.copy()
        if dilate_iter > 0:
            k = np.ones((3, 3), np.uint8)
            m = cv2.dilate(m, k, iterations=dilate_iter)
        out = cv2.inpaint(baked, m, inpaint_radius, cv2.INPAINT_TELEA)
        return out

    # ==================== Mesh Building ====================
    def build_dense_room_mesh(
        self,
        wall_grid_m: float = 0.05,
        floor_grid_m: float = 0.05,
        add_floor: bool = True,     
        add_ceiling: bool = False,   
    ) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
        """Dense room mesh 생성 + 🔥 천장 제거 옵션"""
        corners_xz = [w.p1_xz for w in self.wall_specs]
        corners_xz = np.stack(corners_xz, axis=0).astype(np.float32)

        meshes: List[o3d.geometry.TriangleMesh] = []

        # 🔥 Floor/Ceiling (선택적)
        if add_floor or add_ceiling:
            min_x, max_x, min_z, max_z, _ = self._floor_bounds()

            xs = np.arange(min_x, max_x + 1e-6, floor_grid_m, dtype=np.float32)
            zs = np.arange(min_z, max_z + 1e-6, floor_grid_m, dtype=np.float32)
            nx, nz = len(xs), len(zs)

            GX, GZ = np.meshgrid(xs, zs)
            inside = self._points_in_poly(GX.reshape(-1), GZ.reshape(-1), corners_xz)
            inside = inside.reshape(nz, nx)

            verts_floor = np.column_stack([GX.reshape(-1), np.full(GX.size, self.floor_y, dtype=np.float32), GZ.reshape(-1)]).astype(np.float64)
            verts_ceil = np.column_stack([GX.reshape(-1), np.full(GX.size, self.ceiling_y, dtype=np.float32), GZ.reshape(-1)]).astype(np.float64)

            tris = []
            for j in range(nz - 1):
                for i in range(nx - 1):
                    i0 = j * nx + i
                    i1 = j * nx + (i + 1)
                    i2 = (j + 1) * nx + i
                    i3 = (j + 1) * nx + (i + 1)
                    if inside[j, i] and inside[j, i + 1] and inside[j + 1, i] and inside[j + 1, i + 1]:
                        tris.append([i0, i1, i2])
                        tris.append([i2, i1, i3])
            tris = np.array(tris, dtype=np.int32)

            if tris.size > 0:
                used = np.unique(tris.reshape(-1))
                remap = -np.ones((verts_floor.shape[0],), dtype=np.int32)
                remap[used] = np.arange(len(used), dtype=np.int32)
                tris_c = remap[tris]

                # 🔥 Floor (선택적)
                if add_floor:
                    floor_mesh = o3d.geometry.TriangleMesh()
                    floor_mesh.vertices = o3d.utility.Vector3dVector(verts_floor[used])
                    floor_mesh.triangles = o3d.utility.Vector3iVector(tris_c)
                    floor_mesh.compute_vertex_normals()
                    floor_mesh.paint_uniform_color([0.6, 0.4, 0.2])
                    meshes.append(floor_mesh)
                    print("  ✅ Floor added")

                # 🔥 Ceiling (선택적)
                if add_ceiling:
                    ceil_mesh = o3d.geometry.TriangleMesh()
                    ceil_mesh.vertices = o3d.utility.Vector3dVector(verts_ceil[used])
                    ceil_mesh.triangles = o3d.utility.Vector3iVector(tris_c)
                    ceil_mesh.compute_vertex_normals()
                    ceil_mesh.paint_uniform_color([0.85, 0.85, 0.85])
                    meshes.append(ceil_mesh)
                    print("  ✅ Ceiling added")
                else:
                    print("  ⏭️  Ceiling skipped (open ceiling)")

        # Walls (dense grid)
        all_vertices = []
        all_colors = []
        all_tris = []
        base = 0

        for w in self.wall_specs:
            origin, u_hat, v_hat, n_hat = w.frame()
            nu = max(2, int(np.ceil(w.width_m / wall_grid_m)) + 1)
            nv = max(2, int(np.ceil(w.height_m / wall_grid_m)) + 1)

            us = np.linspace(0.0, w.width_m, nu, dtype=np.float32)
            vs = np.linspace(0.0, w.height_m, nv, dtype=np.float32)
            UU, VV = np.meshgrid(us, vs)

            P = origin.reshape(1, 1, 3) + UU[..., None] * u_hat.reshape(1, 1, 3) + VV[..., None] * v_hat.reshape(1, 1, 3)
            P = P.reshape(-1, 3)

            C = np.tile(np.array([[180, 180, 200]], dtype=np.float32), (P.shape[0], 1))

            tris = []
            for j in range(nv - 1):
                for i in range(nu - 1):
                    i0 = base + j * nu + i
                    i1 = base + j * nu + (i + 1)
                    i2 = base + (j + 1) * nu + i
                    i3 = base + (j + 1) * nu + (i + 1)
                    tris.append([i0, i1, i2])
                    tris.append([i2, i1, i3])
            tris = np.array(tris, dtype=np.int32)

            all_vertices.append(P)
            all_colors.append(C)
            all_tris.append(tris)
            base += P.shape[0]

        V = np.vstack(all_vertices).astype(np.float64)
        C = np.vstack(all_colors).astype(np.float64) / 255.0
        T = np.vstack(all_tris).astype(np.int32)

        wall_mesh = o3d.geometry.TriangleMesh()
        wall_mesh.vertices = o3d.utility.Vector3dVector(V)
        wall_mesh.triangles = o3d.utility.Vector3iVector(T)
        wall_mesh.vertex_colors = o3d.utility.Vector3dVector(C)
        wall_mesh.compute_vertex_normals()
        meshes.append(wall_mesh)

        merged = meshes[0]
        for m in meshes[1:]:
            merged = merged + m

        merged.remove_duplicated_triangles()
        merged.remove_degenerate_triangles()
        merged.remove_non_manifold_edges()

        return merged, {"corners_xz": corners_xz}

    def paint_baked_textures_to_dense_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        baked_results: Dict[int, Dict[str, np.ndarray]],
        floor_baked: Optional[Dict[str, np.ndarray]],
        ppm: int,
        wall_grid_m: float,
        floor_ppm: Optional[int] = None,
        inpaint: bool = True,
        inpaint_radius: int = 5,
    ) -> o3d.geometry.TriangleMesh:
        """Texture painting to mesh vertices"""
        V = np.asarray(mesh.vertices)
        C = np.asarray(mesh.vertex_colors)
        if C.shape[0] != V.shape[0]:
            C = np.zeros((V.shape[0], 3), dtype=np.float64)

        # Wall textures
        wall_tex = {}
        for w in self.wall_specs:
            wid = w.wall_id
            baked = baked_results[wid]["baked"]
            mask = baked_results[wid]["mask"]
            if inpaint:
                final = self.inpaint_holes_opencv(baked, mask, inpaint_radius=inpaint_radius)
                cv2.imwrite(str(self.out_dir / f"wall_{wid:02d}_final.png"), final)
            else:
                final = baked
            wall_tex[wid] = final

        if len(mesh.vertex_normals) == 0:
            mesh.compute_vertex_normals()
        N = np.asarray(mesh.vertex_normals)
        y = V[:, 1]
        wall_zone = (np.abs(N[:, 1]) < 0.5) & (y >= (self.floor_y - 0.05)) & (y <= (self.ceiling_y + 0.05))
        idxs = np.where(wall_zone)[0]

        if len(idxs) > 0:
            Pw = V[idxs].astype(np.float32)

            best_wid = np.full((Pw.shape[0],), -1, dtype=np.int32)
            best_dist = np.full((Pw.shape[0],), 1e9, dtype=np.float32)

            wall_frames = {}
            for w in self.wall_specs:
                origin, u_hat, v_hat, n_hat = w.frame()
                d = -float(np.dot(n_hat, origin))
                dist = (Pw @ n_hat.reshape(3, 1)).reshape(-1) + d
                dist = np.abs(dist)
                wall_frames[w.wall_id] = (origin, u_hat, v_hat, n_hat, d)
                m = dist < best_dist
                best_dist[m] = dist[m]
                best_wid[m] = w.wall_id

            assigned = best_dist < 0.05
            idxs2 = idxs[assigned]
            Pw2 = Pw[assigned]
            wid2 = best_wid[assigned]

            for w in self.wall_specs:
                wid = w.wall_id
                m = (wid2 == wid)
                if m.sum() == 0:
                    continue

                origin, u_hat, v_hat, n_hat, d = wall_frames[wid]
                P = Pw2[m]
                rel = P - origin.reshape(1, 3)
                u = (rel @ u_hat.reshape(3, 1)).reshape(-1)
                v = (rel @ v_hat.reshape(3, 1)).reshape(-1)

                inside = (u >= 0.0) & (u <= w.width_m) & (v >= 0.0) & (v <= w.height_m)
                if inside.sum() == 0:
                    continue

                u = u[inside]
                v = v[inside]
                vertex_ids = idxs2[m][inside]

                tex = wall_tex[wid]
                h_px, w_px = tex.shape[:2]

                ix = np.floor(u * ppm).astype(np.int32)
                iy = np.floor((w.height_m - v) * ppm).astype(np.int32)
                ix = np.clip(ix, 0, w_px - 1)
                iy = np.clip(iy, 0, h_px - 1)

                col_bgr = tex[iy, ix].astype(np.float32) / 255.0
                col_rgb = col_bgr[:, ::-1]
                C[vertex_ids] = col_rgb

        mesh.vertex_colors = o3d.utility.Vector3dVector(C)

        # Floor painting
        if floor_baked is not None:
            floor_tex = floor_baked["baked"]
            floor_mask = floor_baked["mask"]
            if inpaint:
                floor_final = self.inpaint_holes_opencv(floor_tex, floor_mask, inpaint_radius=inpaint_radius)
                cv2.imwrite(str(self.out_dir / "floor_final.png"), floor_final)
            else:
                floor_final = floor_tex

            bounds = floor_baked["bounds"].astype(np.float32)
            min_x, max_x, min_z, max_z = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
            ppm_f = ppm if floor_ppm is None else int(floor_ppm)

            V_all = np.asarray(mesh.vertices)
            N_all = np.asarray(mesh.vertex_normals)
            y_all = V_all[:, 1]
            floor_zone = (np.abs(y_all - self.floor_y) < 0.05) & (np.abs(N_all[:, 1]) > 0.8)
            fidx = np.where(floor_zone)[0]
            
            if len(fidx) > 0:
                P = V_all[fidx].astype(np.float32)

                w_px = floor_final.shape[1]
                h_px = floor_final.shape[0]
                ix = np.floor((P[:, 0] - min_x) * ppm_f).astype(np.int32)
                iy = np.floor((max_z - P[:, 2]) * ppm_f).astype(np.int32)
                inside = (ix >= 0) & (ix < w_px) & (iy >= 0) & (iy < h_px)
                
                if inside.sum() > 0:
                    ix2 = ix[inside]
                    iy2 = iy[inside]
                    ids2 = fidx[inside]
                    col_bgr = floor_final[iy2, ix2].astype(np.float32) / 255.0
                    C[ids2] = col_bgr[:, ::-1]
                    mesh.vertex_colors = o3d.utility.Vector3dVector(C)

        return mesh

    # ==================== Full Pipeline ====================
    def run(
        self,
        ppm: int = 256,
        sample_rate: int = 3,
        max_frames: Optional[int] = None,
        confidence_threshold: int = 2,
        plane_dist_th_m: float = 0.03,
        wall_grid_m: float = 0.05,
        floor_grid_m: float = 0.05,
        floor_sample_rate: int = 2,
        floor_multi_pass: bool = True,
        save_textured_mesh: bool = True,
        # 🔥 NEW: 3가지 개선 파라미터
        floor_weight_threshold: float = 0.5,
        wall_default_color: List[int] = [240, 240, 245],
        add_ceiling: bool = False,
    ):
        print("\n" + "="*60)
        print("🔥 IMPROVED TEXTURE RESTORATION")
        print("="*60)
        print(f"  1. Floor weight threshold: {floor_weight_threshold} (밝기 개선)")
        print(f"  2. Wall default color: {wall_default_color} (검은 벽 방지)")
        print(f"  3. Ceiling: {'YES' if add_ceiling else 'NO (open)'}")
        print("="*60)
        
        # Wall baking
        baked = self.bake_wall_textures(
            ppm=ppm,
            sample_rate=sample_rate,
            max_frames=max_frames,
            confidence_threshold=confidence_threshold,
            plane_dist_th_m=plane_dist_th_m,
            save_intermediate=True,
            default_color=wall_default_color, 
        )

        # 🔥 Enhanced floor baking
        floor_baked = self.bake_floor_texture_enhanced(
            ppm=ppm,
            sample_rate=floor_sample_rate,
            max_frames=max_frames,
            confidence_threshold=confidence_threshold,
            multi_pass=floor_multi_pass,
            save_intermediate=True,
            weight_threshold=floor_weight_threshold,
        )

        # Mesh building & painting
        mesh, _ = self.build_dense_room_mesh(
            wall_grid_m=wall_grid_m,
            floor_grid_m=floor_grid_m,
            add_floor=True,
            add_ceiling=add_ceiling,
        )
        
        mesh = self.paint_baked_textures_to_dense_mesh(
            mesh=mesh,
            baked_results=baked,
            floor_baked=floor_baked,
            ppm=ppm,
            wall_grid_m=wall_grid_m,
            inpaint=True,
            inpaint_radius=5,
        )

        if save_textured_mesh:
            out_mesh = self.out_dir / "textured_room_vertexcolor.ply"
            o3d.io.write_triangle_mesh(str(out_mesh), mesh, write_vertex_colors=True)
            print(f"\n✅ Saved textured mesh: {out_mesh}")

        print("\n🎉 IMPROVED texture restoration done!")
        return mesh


if __name__ == "__main__":
    # [1] 현재 파일(wall_texture_restoration.py)의 위치를 기준으로 경로 설정
    # 이렇게 하면 폴더를 통째로 옮겨도 코드가 깨지지 않습니다.
    BASE_DIR = Path(__file__).resolve().parent  # STRAYSCANNER 폴더
    
    # [2] 폴더 구조에 맞춰 경로 매핑
    # 데이터 위치: STRAYSCANNER/data/room
    DATA_PATH = BASE_DIR / "data" / "room"
    
    # 출력물 루트: STRAYSCANNER/output
    OUTPUT_ROOT = BASE_DIR / "output"
    
    # 세부 입력/출력 경로 설정
    # output/room_detected_results 안에 있는 파일들을 참조
    DETECTED_DIR = OUTPUT_ROOT / "room_detected_results"
    RECON_JSON = DETECTED_DIR / "room_dimensions.json"
    
    # 결과가 저장될 위치: output/room_wall_textures_out
    OUT_DIR = OUTPUT_ROOT / "room_wall_textures_out"

    # [3] 경로가 진짜 있는지 체크 (디버깅용)
    if not DATA_PATH.exists():
        print(f"❌ 데이터 경로를 찾을 수 없습니다: {DATA_PATH}")
        print("💡 팁: 'data' 폴더 안에 'room' 폴더가 있는지 확인하세요.")
        exit(1)
    if not DETECTED_DIR.exists():
        print(f"❌ 감지 결과 경로를 찾을 수 없습니다: {DETECTED_DIR}")
        print("💡 팁: 이전 단계(structure_detection.py 등)를 먼저 실행해서 output 폴더를 만드세요.")
        exit(1)

    print(f"📂 Project Root: {BASE_DIR}")
    print(f"📂 Data Path:    {DATA_PATH}")
    print(f"📂 Output Path:  {OUT_DIR}")

    # [4] 클래스 실행 (수정된 경로 주입)
    restorer = WallTextureRestorerEnhanced(
        dataset_path=str(DATA_PATH),      # data/room
        detected_dir=str(DETECTED_DIR),   # output/room_detected_results
        recon_json=str(RECON_JSON),       # output/room_detected_results/room_dimensions.json
        out_dir=str(OUT_DIR),             # output/room_wall_textures_out
    )

    mesh = restorer.run(
        ppm=256,
        sample_rate=3,
        floor_sample_rate=2,
        max_frames=600,
        confidence_threshold=2,
        plane_dist_th_m=0.03,
        wall_grid_m=0.05,
        floor_grid_m=0.05,
        floor_multi_pass=True,
        save_textured_mesh=True,
        floor_weight_threshold=0.5,
        wall_default_color=[240, 240, 245],
        add_ceiling=False,
    )

    o3d.visualization.draw_geometries(
        [mesh],
        window_name="IMPROVED Textured Room (No Ceiling!)",
        width=1280,
        height=720,
        mesh_show_back_face=True,
    )