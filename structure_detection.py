import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import pickle
import json
import sys

# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class WallFloorCeilingDetector:
    def __init__(self, dataset_path):
        # [수정] Path 객체로 확실하게 변환
        self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"❌ 데이터 경로가 없습니다: {self.dataset_path}")
            
        self.load_data()
        
        # ✅ alignment 정보 저장용 변수
        self.alignment_centroid = None
        self.alignment_R = None
        self.alignment_rotation_angle = None
        
    def load_data(self):
        print(f"📂 데이터 로딩 중... ({self.dataset_path.name})")
        camera_matrix_path = self.dataset_path / 'camera_matrix.csv'
        odometry_path = self.dataset_path / 'odometry.csv'
        
        if not camera_matrix_path.exists() or not odometry_path.exists():
            raise FileNotFoundError("필수 파일(camera_matrix.csv, odometry.csv)이 누락되었습니다.")

        self.intrinsics = np.loadtxt(str(camera_matrix_path), delimiter=',')
        self.odometry = pd.read_csv(str(odometry_path))
        self.odometry.columns = self.odometry.columns.str.strip()
        
        depth_dir = self.dataset_path / 'depth'
        self.depth_files = sorted(list(depth_dir.glob('*.png')))
        print(f"✅ {len(self.depth_files)}개 프레임 로드 완료")
    
    def create_pointcloud(self, confidence_threshold=2, sample_rate=3, max_depth=5.0):
        print("\n" + "="*60)
        print("🎯 1단계: 포인트 클라우드 생성 & 법선 계산")
        print("="*60)
        
        voxel_size = 0.02
        voxel_data = {}
        fx = self.intrinsics[0, 0] * (256 / 1920)
        fy = self.intrinsics[1, 1] * (192 / 1440)
        cx = self.intrinsics[0, 2] * (256 / 1920)
        cy = self.intrinsics[1, 2] * (192 / 1440)
        
        rgb_path = str(self.dataset_path / 'rgb.mp4')
        if not os.path.exists(rgb_path):
             raise FileNotFoundError(f"rgb.mp4 없음: {rgb_path}")

        # [수정] skvideo 대신 cv2 사용 (라이브러리 통일)
        cap = cv2.VideoCapture(rgb_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {rgb_path}")
        
        idx = 0
        while True:
            ret, rgb_frame_bgr = cap.read()
            if not ret:
                break
            
            # cv2는 BGR로 읽으므로 RGB로 변환
            rgb_frame = cv2.cvtColor(rgb_frame_bgr, cv2.COLOR_BGR2RGB)
            
            # 루프 제어
            if idx % sample_rate != 0: 
                idx += 1
                continue
            if idx >= len(self.odometry): 
                break
                
            print(f"  프레임 {idx}/{len(self.odometry)}", end='\r')
            
            depth = cv2.imread(str(self.depth_files[idx]), cv2.IMREAD_UNCHANGED)
            if depth is None: 
                idx += 1
                continue

            depth_m = depth.astype(np.float32) / 1000.0
            confidence_path = self.dataset_path / 'confidence' / f'{idx:06d}.png'
            
            if confidence_path.exists():
                confidence = cv2.imread(str(confidence_path), cv2.IMREAD_GRAYSCALE)
                valid_mask = (confidence >= confidence_threshold) & (depth_m > 0.1) & (depth_m < max_depth)
            else:
                valid_mask = (depth_m > 0.1) & (depth_m < max_depth)
            
            if valid_mask.sum() < 100: 
                idx += 1
                continue
            
            row = self.odometry.iloc[idx]
            T_WC = np.eye(4)
            T_WC[:3, :3] = Rotation.from_quat([row['qx'], row['qy'], row['qz'], row['qw']]).as_matrix()
            T_WC[:3, 3] = [row['x'], row['y'], row['z']]
            
            v, u = np.where(valid_mask)
            z = depth_m[v, u]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            points_world = (T_WC @ np.stack([x, y, z, np.ones_like(x)], axis=-1).T).T[:, :3]
            colors = cv2.resize(rgb_frame, (256, 192))[v, u]
            
            voxel_keys = np.floor(points_world / voxel_size).astype(int)
            for pt, col, vk in zip(points_world, colors, voxel_keys):
                key = tuple(vk)
                if key not in voxel_data:
                    voxel_data[key] = {'points': [], 'colors': [], 'count': 0}
                voxel_data[key]['points'].append(pt)
                voxel_data[key]['colors'].append(col)
                voxel_data[key]['count'] += 1
            
            idx += 1
            
        cap.release() # 자원 해제
        
        filtered_points, filtered_colors = [], []
        for vd in voxel_data.values():
            if vd['count'] >= 2:
                filtered_points.append(np.mean(vd['points'], axis=0))
                filtered_colors.append(np.mean(vd['colors'], axis=0))
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(filtered_colors) / 255.0)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        print("\n  법선(Normal) 계산 중...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        self.pointcloud = pcd
        self.points = np.asarray(pcd.points)
        self.colors = np.asarray(pcd.colors)
        self.normals = np.asarray(pcd.normals)
        print(f"✅ 포인트 클라우드 생성 완료: {len(self.points):,}개")

    def detect_floor_ceiling(self):
        print("\n" + "="*60)
        print("🏠 2단계: 바닥/천장 분리")
        print("="*60)
        
        y_axis = np.array([0, 1, 0])
        horizontal_mask = np.abs(np.dot(self.normals, y_axis)) > 0.85
        horizontal_points = self.points[horizontal_mask]
        
        hist, bin_edges = np.histogram(horizontal_points[:, 1], bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peaks, _ = find_peaks(hist, height=len(horizontal_points) * 0.01)
        
        if len(peaks) >= 2:
            h1, h2 = bin_centers[peaks[0]], bin_centers[peaks[-1]]
            self.floor_height = min(h1, h2)
            self.ceiling_height = max(h1, h2)
        else:
            self.floor_height = np.percentile(horizontal_points[:, 1], 5)
            self.ceiling_height = np.percentile(horizontal_points[:, 1], 95)
            
        self.room_height = self.ceiling_height - self.floor_height
        print(f"  바닥: {self.floor_height:.2f}m, 천장: {self.ceiling_height:.2f}m")
        
        self.floor_mask = (horizontal_mask) & (np.abs(self.points[:, 1] - self.floor_height) < 0.15)
        self.ceiling_mask = (horizontal_mask) & (np.abs(self.points[:, 1] - self.ceiling_height) < 0.15)
        self.floor_points = self.points[self.floor_mask]
        self.ceiling_points = self.points[self.ceiling_mask]

    def extract_vertical_surfaces(self):
        print("\n" + "="*60)
        print("🧱 2.5단계: 수직면 추출")
        print("="*60)
        
        vertical_mask = (np.abs(np.dot(self.normals, [0, 1, 0])) < 0.3) 
        vertical_mask = vertical_mask & (~self.floor_mask) & (~self.ceiling_mask)
        
        self.vertical_points = self.points[vertical_mask]
        self.vertical_normals = self.normals[vertical_mask]
        print(f"  ✅ 수직면 포인트 추출 완료: {len(self.vertical_points):,}개")

    def detect_walls(self):
        print("\n" + "="*60)
        print("🧱 3단계: 벽 검출 및 분리")
        print("="*60)
        
        normals_2d = self.vertical_normals[:, [0, 2]]
        normals_2d /= np.linalg.norm(normals_2d, axis=1, keepdims=True)
        angles_deg = np.degrees(np.mod(np.arctan2(normals_2d[:, 1], normals_2d[:, 0]), np.pi))
        
        hist, bin_edges = np.histogram(angles_deg, bins=90, range=(0, 180))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peaks, _ = find_peaks(hist, height=len(angles_deg)*0.015, distance=10)
        
        self.walls = []
        for p in peaks:
            peak_deg = bin_centers[p]
            diff = np.abs(angles_deg - peak_deg)
            diff = np.minimum(diff, 180 - diff)
            mask_dir = diff < 20
            points_dir = self.vertical_points[mask_dir]
            normals_dir = normals_2d[mask_dir]
            if len(points_dir) < 100: continue
            
            main_normal = normals_dir.mean(axis=0)
            main_normal /= np.linalg.norm(main_normal)
            
            proj = np.dot(points_dir[:, [0, 2]], main_normal)
            hist_pos, edges_pos = np.histogram(proj, bins=50)
            centers_pos = (edges_pos[:-1] + edges_pos[1:]) / 2
            peaks_pos, _ = find_peaks(hist_pos, height=len(proj)*0.05, distance=3)
            
            if len(peaks_pos) == 0: peaks_pos = [np.argmax(hist_pos)]
            dists = np.abs(proj[:, None] - centers_pos[peaks_pos][None, :])
            labels = np.argmin(dists, axis=1)
            
            for i, peak_idx in enumerate(peaks_pos):
                mask_pos = (labels == i) & (np.min(dists, axis=1) < 0.5)
                wall_pts = points_dir[mask_pos]
                if len(wall_pts) < 50: continue
                h = wall_pts[:, 1]
                if (h.max() - h.min()) < self.room_height * 0.5: continue
                
                self.walls.append({
                    'points': wall_pts,
                    'center': wall_pts.mean(axis=0),
                    'angle_deg': peak_deg,
                    'count': len(wall_pts)
                })
        print(f"  총 {len(self.walls)}개의 벽 검출됨.")

    def align_entire_scene(self):
        """
        ✅ 수정: alignment 정보를 인스턴스 변수에 저장
        """
        print("\n" + "="*60)
        print("📐 3.5단계: 벽면 기준 정밀 정렬 (Histogram Alignment)")
        print("="*60)
        
        all_wall_points = np.vstack([w['points'] for w in self.walls])
        centroid = np.mean(all_wall_points, axis=0)
        
        if len(self.walls) > 0:
            normals_xz = self.vertical_normals[:, [0, 2]]
            angles = np.arctan2(normals_xz[:, 1], normals_xz[:, 0])
            
            hist, bins = np.histogram(angles, bins=360, range=(-np.pi, np.pi))
            peak_idx = np.argmax(hist)
            dominant_angle = (bins[peak_idx] + bins[peak_idx+1]) / 2
            
            target_angle = round(dominant_angle / (np.pi/2)) * (np.pi/2)
            rotation_angle = target_angle - dominant_angle
        else:
            rotation_angle = 0
            
        print(f"  🔄 회전 각도 보정: {np.degrees(rotation_angle):.2f}도")

        c, s = np.cos(rotation_angle), np.sin(rotation_angle)
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

        # ✅ alignment 정보 저장
        self.alignment_centroid = centroid.copy()
        self.alignment_R = R.copy()
        self.alignment_rotation_angle = rotation_angle

        def apply_transform(points):
            if len(points) == 0: return points
            return (points - centroid) @ R.T

        self.floor_points = apply_transform(self.floor_points)
        self.ceiling_points = apply_transform(self.ceiling_points)
        self.points = apply_transform(self.points)
        self.normals = self.normals @ R.T 
        
        for w in self.walls:
            w['points'] = apply_transform(w['points'])
            w['center'] = apply_transform(w['center'][np.newaxis, :])[0]
            
        print("  ✅ 모든 데이터 정렬 완료 (벽면 기준)")

    # [수정] output_dir를 인자로 받도록 변경
    def save_and_visualize(self, output_dir):
        self.align_entire_scene()

        # Path 객체 변환
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # parents=True: 중간 폴더 없으면 생성
        
        print(f"\n💾 결과 저장 시작: {output_dir}/")
        
        full_points = []
        full_colors = []

        if len(self.floor_points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.floor_points)
            c = [0.6, 0.4, 0.2]
            pcd.paint_uniform_color(c)
            o3d.io.write_point_cloud(str(output_dir / 'floor.ply'), pcd)
            full_points.append(self.floor_points)
            full_colors.append(np.tile(c, (len(self.floor_points), 1)))
            
        if len(self.ceiling_points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.ceiling_points)
            c = [0.8, 0.8, 0.8]
            pcd.paint_uniform_color(c)
            o3d.io.write_point_cloud(str(output_dir / 'ceiling.ply'), pcd)
            full_points.append(self.ceiling_points)
            full_colors.append(np.tile(c, (len(self.ceiling_points), 1)))
            
        wall_cloud_dir = output_dir / 'wall_clouds'
        wall_cloud_dir.mkdir(exist_ok=True)
        colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1]] 
        
        for i, w in enumerate(self.walls):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(w['points'])
            c = colors[i % len(colors)]
            pcd.paint_uniform_color(c)
            o3d.io.write_point_cloud(str(wall_cloud_dir / f'wall_{i+1}.ply'), pcd)
            full_points.append(w['points'])
            full_colors.append(np.tile(c, (len(w['points']), 1)))

        if full_points:
            combined_pcd = o3d.geometry.PointCloud()
            combined_pcd.points = o3d.utility.Vector3dVector(np.vstack(full_points))
            combined_pcd.colors = o3d.utility.Vector3dVector(np.vstack(full_colors))
            o3d.io.write_point_cloud(str(output_dir / 'full_room_structure.ply'), combined_pcd)

        self._save_metadata(output_dir)
        self._visualize_plots(str(output_dir / 'detection_report.png'))

    def _save_metadata(self, output_dir):
        """
        ✅ 수정: alignment 정보 추가
        """
        pkl_data = {
            'walls': self.walls,
            'floor_height': self.floor_height,
            'ceiling_height': self.ceiling_height,
            'room_height': self.room_height,
            'num_walls': len(self.walls),
            # ✅ alignment 정보 추가
            'alignment': {
                'centroid': self.alignment_centroid.tolist(),
                'R': self.alignment_R.tolist(),
                'rotation_angle_rad': float(self.alignment_rotation_angle)
            }
        }
        pkl_path = output_dir / 'walls_data.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        print(f"  ✅ walls_data.pkl 저장 완료 (alignment 정보 포함)")

        json_walls = []
        for i, w in enumerate(self.walls):
            center = w['center'].tolist()
            min_xz = [float(np.min(w['points'][:,0])), float(np.min(w['points'][:,2]))]
            max_xz = [float(np.max(w['points'][:,0])), float(np.max(w['points'][:,2]))]
            json_walls.append({
                "id": f"wall_{i+1}",
                "point_count": int(w['count']),
                "center_xyz": [round(c, 3) for c in center],
                "bounds_xz": {"min": min_xz, "max": max_xz}
            })

        json_data = {
            "room_info": {
                "floor_height_m": round(float(self.floor_height), 3),
                "ceiling_height_m": round(float(self.ceiling_height), 3),
                "room_height_m": round(float(self.room_height), 3),
                "total_walls": len(self.walls)
            },
            "walls": json_walls
        }
        json_path = output_dir / 'room_dimensions.json' # [주의] 이름 통일 (walls_metadata -> room_dimensions)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

    def _visualize_plots(self, save_path):
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title("Top View (Aligned)")
        ax1.set_aspect('equal'); ax1.grid(alpha=0.3)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, w in enumerate(self.walls):
            ax1.scatter(w['points'][:,0], w['points'][:,2], s=1, color=colors[i%10], label=f'W{i+1}')
        
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.set_title("3D Structure (Aligned)")
        for i, w in enumerate(self.walls):
            pts = w['points'][::5]
            ax2.scatter(pts[:,0], pts[:,2], pts[:,1], s=1, color=colors[i%10])
        ax2.set_zlim(self.floor_height, self.ceiling_height)
        
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title("Wall Normals")
        if len(self.normals) > 0:
            n2d = self.normals[:, [0, 2]]
            deg = np.degrees(np.arctan2(n2d[:,1], n2d[:,0]))
            ax3.hist(deg, bins=90, color='blue', alpha=0.7)
            
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"  ✅ [Report] {save_path} 저장됨")

    # [수정] output_dir를 인자로 받아서 전달
    def run(self, output_dir):
        self.create_pointcloud()
        self.detect_floor_ceiling()
        self.extract_vertical_surfaces()
        self.detect_walls()
        self.save_and_visualize(output_dir)

if __name__ == "__main__":
    # [1] 경로 자동 설정 (Dynamic Path Config)
    BASE_DIR = Path(__file__).resolve().parent
    
    # 예시: room 데이터 실행
    TARGET_SCENE = "room"  # class, lab, room 중 선택
    
    DATA_PATH = BASE_DIR / "data" / TARGET_SCENE
    OUTPUT_DIR = BASE_DIR / "output" / f"{TARGET_SCENE}_detected_results"
    
    print(f"📂 Project Root: {BASE_DIR}")
    print(f"📂 Input Data:   {DATA_PATH}")
    print(f"📂 Output Dir:   {OUTPUT_DIR}")
    
    # [2] 데이터 경로 확인
    if not DATA_PATH.exists():
        print(f"❌ 오류: 데이터 경로를 찾을 수 없습니다: {DATA_PATH}")
        sys.exit(1)
        
    detector = WallFloorCeilingDetector(DATA_PATH)
    
    # [3] run 호출 시 output_dir 전달
    detector.run(OUTPUT_DIR)
    
    print("\n🎮 1단계 완료: detected_results 확인")
    print("   ✅ alignment 정보가 walls_data.pkl에 저장되었습니다!")