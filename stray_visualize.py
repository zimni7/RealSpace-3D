import numpy as np
import open3d as o3d
import cv2  # skvideo 제거하고 cv2로 통일
import pandas as pd
from scipy.spatial.transform import Rotation
import os
import argparse
from pathlib import Path

def load_data(dataset_path):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ 데이터 경로가 없습니다: {dataset_path}")

    # 1. 파일 경로 설정
    camera_matrix_path = dataset_path / 'camera_matrix.csv'
    odometry_path = dataset_path / 'odometry.csv'
    rgb_path = dataset_path / 'rgb.mp4'
    depth_dir = dataset_path / 'depth'

    # 2. 필수 파일 확인
    if not camera_matrix_path.exists():
        raise FileNotFoundError("camera_matrix.csv가 없습니다.")
    if not odometry_path.exists():
        raise FileNotFoundError("odometry.csv가 없습니다.")
    if not rgb_path.exists():
        raise FileNotFoundError("rgb.mp4가 없습니다.")

    # 3. 데이터 로드
    intrinsics = np.loadtxt(str(camera_matrix_path), delimiter=',')
    odometry = pd.read_csv(str(odometry_path))
    odometry.columns = odometry.columns.str.strip()
    depth_files = sorted(list(depth_dir.glob('*.png')))

    return intrinsics, odometry, depth_files, rgb_path

def main():
    parser = argparse.ArgumentParser(description="StrayScanner Data Visualizer")
    parser.add_argument('--input', type=str, required=True, help='Path to data directory (e.g., data/room)')
    args = parser.parse_args()

    print(f"📂 Loading data from: {args.input}")
    try:
        intrinsics, odometry, depth_files, rgb_path = load_data(args.input)
    except Exception as e:
        print(e)
        return

    # 카메라 파라미터 조정 (StrayScanner 해상도)
    fx = intrinsics[0, 0] * (256 / 1920)
    fy = intrinsics[1, 1] * (192 / 1440)
    cx = intrinsics[0, 2] * (256 / 1920)
    cy = intrinsics[1, 2] * (192 / 1440)

    points = []
    colors = []
    
    # [수정] skvideo 대신 OpenCV(cv2) 사용
    cap = cv2.VideoCapture(str(rgb_path))
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {rgb_path}")
        return

    sample_rate = 5 # 속도를 위해 5프레임마다 하나씩 처리
    idx = 0
    total_frames = min(len(odometry), len(depth_files))

    print("☁️ 포인트 클라우드 생성 중... (잠시만 기다려주세요)")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # 범위 체크 및 샘플링
        if idx >= total_frames:
            break
        if idx % sample_rate != 0:
            idx += 1
            continue
            
        print(f"  Processing frame {idx}/{total_frames}", end='\r')

        # BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb_frame, (256, 192))

        # Depth 로드
        depth = cv2.imread(str(depth_files[idx]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            idx += 1
            continue
        
        depth_m = depth.astype(np.float32) / 1000.0
        
        # 유효한 깊이 마스크
        mask = (depth_m > 0.1) & (depth_m < 3.0) # 3미터 이내만
        
        if mask.sum() < 100:
            idx += 1
            continue

        # 좌표 계산
        row = odometry.iloc[idx]
        q = [row['qx'], row['qy'], row['qz'], row['qw']]
        t = [row['x'], row['y'], row['z']]
        R = Rotation.from_quat(q).as_matrix()
        
        # Pixel to Camera
        v, u = np.where(mask)
        z = depth_m[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Camera to World
        cam_points = np.stack([x, y, z], axis=1)
        world_points = (R @ cam_points.T).T + t
        
        points.append(world_points)
        colors.append(rgb_resized[v, u] / 255.0)
        
        idx += 1

    cap.release()
    print("\n✅ 포인트 클라우드 병합 중...")
    
    if not points:
        print("❌ 생성된 포인트가 없습니다.")
        return

    # Open3D 시각화
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    
    # 노이즈 제거 (선택)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 좌표축 추가
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    print("🎮 3D 뷰어 실행 (창을 닫으면 종료됩니다)")
    o3d.visualization.draw_geometries([pcd, axes], window_name="StrayScanner Raw Data")

if __name__ == "__main__":
    main()