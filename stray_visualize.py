import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image
import skvideo.io

description = """
This script visualizes datasets collected using the Stray Scanner app.
"""

usage = """
Basic usage: python stray_visualize.py <path-to-dataset-folder>
"""

DEPTH_WIDTH = 256    # Depth 이미지 너비 (픽셀)
DEPTH_HEIGHT = 192   # Depth 이미지 높이 (픽셀)
MAX_DEPTH = 20.0     # 최대 depth 거리 - 20m 이상은 무시

# 커멘드 라인 인자 파싱
def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('path', type=str, help="Path to StrayScanner dataset to process.")
    # 카메라의 이동 경로를 선으로 시각화
    parser.add_argument('--trajectory', '-t', action='store_true', help="Visualize the trajectory of the camera as a line.")
    # 파일의 카메라 좌표계 프레임 시각화
    parser.add_argument('--frames', '-f', action='store_true', help="Visualize camera coordinate frames from the odometry file.")  
    # 연결된 포인트 클라우드 표시
    parser.add_argument('--point-clouds', '-p', action='store_true', help="Show concatenated point clouds.")
    # Open3D RGB-D 통합 파이프라인으로 포인트 클라우드를 통합하고 시각화
    parser.add_argument('--integrate', '-i', action='store_true', help="Integrate point clouds using the Open3D RGB-D integration pipeline, and visualize it.")
    # 포인트 클라우드 통합으로 생성된 메쉬를 저장할 파일명 
    parser.add_argument('--mesh-filename', type=str, help='Mesh generated from point cloud integration will be stored in this file. open3d.io.write_triangle_mesh will be used.', default=None)
    # 매 n번째 포인트 클라우드와 좌표 프레임만 표시
    parser.add_argument('--every', type=int, default=60, help="Show only every nth point cloud and coordinate frames. Only used for point cloud and odometry visualization.")
    # RGB-D 통합에 사용할 복셀 크기 (미터 단위)
    parser.add_argument('--voxel-size', type=float, default=0.015, help="Voxel size in meters to use in RGB-D integration.")
    # 주어진 값 이상의 신뢰도를 가진 depth만 유지. 0, 1, 2 중 선택
    parser.add_argument('--confidence', '-c', type=int, default=1,
            help="Keep only depth estimates with confidence equal or higher to the given value. There are three different levels: 0, 1 and 2. Higher is more confident.")
    return parser.parse_args()

# 카메라 행렬 리사이징
def _resize_camera_matrix(camera_matrix, scale_x, scale_y):
    """
    카메라 내부 파라미터 행렬을 스케일 조정

    iPhone의 RGB 카메라 해상도(1920x1440)와 
    Depth 카메라 해상도(256x192)가 다르기 때문에 
    intrinsic matrix를 스케일링
    
    Args:
        camera_matrix: 3x3 카메라 intrinsic 행렬
        scale_x: X축 스케일 비율
        scale_y: Y축 스케일 비율
    
    Returns:
        np.array: 스케일링된 3x3 행렬
    """
    fx = camera_matrix[0, 0]    # X축 초점 거리
    fy = camera_matrix[1, 1]    # Y축 초점 거리
    cx = camera_matrix[0, 2]    # X축 주점 (principal point)
    cy = camera_matrix[1, 2]    # Y축 주점
    # 스케일 적용된 새 행렬 반환
    return np.array([[fx * scale_x, 0.0, cx * scale_x],
        [0., fy * scale_y, cy * scale_y],
        [0., 0., 1.0]])

# 데이터 로딩
def read_data(flags):
    """
    StrayScanner 데이터셋의 모든 파일을 읽어옴
    
    Args:
        flags: 커맨드 라인 인자
    
    Returns:
        dict: {
            'poses': 각 프레임의 4x4 변환 행렬 리스트,
            'intrinsics': 3x3 카메라 intrinsic 행렬,
            'depth_frames': depth 이미지 파일 경로 리스트
        }
    """
    # 1. 카메라 intrinsic 행렬 로드 (3x3)
    intrinsics = np.loadtxt(os.path.join(flags.path, 'camera_matrix.csv'), delimiter=',')
    # 2. 오도메트리 데이터 로드 (카메라 위치/회전 정보)
    odometry = np.loadtxt(os.path.join(flags.path, 'odometry.csv'), delimiter=',', skiprows=1)
    poses = []

    for line in odometry:
        # timestamp, frame, x, y, z, qx, qy, qz, qw
        # 위치 정보 추출 (x,y,z)
        position = line[2:5]
        # 회전 정보 추출 (qx, qy, qz, qw)
        quaternion = line[5:]
        # 4x4 변환 행렬 생성 (T_WC = World에서 Camera로의 변환)
        T_WC = np.eye(4)
        # 쿼터니언을 3x3 회전 행렬로 변환
        T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        # 이동 벡터 설정
        T_WC[:3, 3] = position
        poses.append(T_WC)

    # 3. Depth 이미지 파일 경로 수집
    depth_dir = os.path.join(flags.path, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
    # .npy 또는 .png 파일만 필터링
    depth_frames = [f for f in depth_frames if '.npy' in f or '.png' in f]
    return { 'poses': poses, 'intrinsics': intrinsics, 'depth_frames': depth_frames }

# Depth 이미지 로딩
def load_depth(path, confidence=None, filter_level=0):
    # 파일 형식에 따라 로드
    if path[-4:] == '.npy':
        depth_mm = np.load(path)
    elif path[-4:] == '.png':
        depth_mm = np.array(Image.open(path))
    # 밀리미터를 미터로 변환 (1000으로 나누기)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    # 신뢰도 필터링: 낮은 신뢰도 포인트는 0으로 설정
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0
    return o3d.geometry.Image(depth_m)

# 신뢰도 맵 로딩
def load_confidence(path):
    return np.array(Image.open(path))

# Intrinsic 행렬 스케일링
def get_intrinsics(intrinsics):
    """
    Scales the intrinsics matrix to be of the appropriate scale for the depth maps.
    """
     # 스케일 비율 계산
    intrinsics_scaled = _resize_camera_matrix(intrinsics, DEPTH_WIDTH / 1920, DEPTH_HEIGHT / 1440)
    # Open3D 카메라 객체 생성
    return o3d.camera.PinholeCameraIntrinsic(width=DEPTH_WIDTH, height=DEPTH_HEIGHT, fx=intrinsics_scaled[0, 0],
        fy=intrinsics_scaled[1, 1], cx=intrinsics_scaled[0, 2], cy=intrinsics_scaled[1, 2])

# 카메라 궤적 시각화
def trajectory(flags, data):
    """
    Returns a set of lines connecting each camera poses world frame position.
    returns: [open3d.geometry.LineSet]
    """
    line_sets = []
    previous_pose = None
    for i, T_WC in enumerate(data['poses']):
        if previous_pose is not None:
            # 이전 위치와 현재 위치를 연결하는 선 생성
            points = o3d.utility.Vector3dVector([previous_pose[:3, 3], T_WC[:3, 3]])
            lines = o3d.utility.Vector2iVector([[0, 1]])
            line = o3d.geometry.LineSet(points=points, lines=lines)
            line_sets.append(line)
        previous_pose = T_WC
    return line_sets

# 카메라 좌표계 프레임 시각화
def show_frames(flags, data):
    """
    Returns a list of meshes of coordinate axes that have been transformed to represent the camera matrix
    at each --every:th frame.

    flags: Command line arguments
    data: dict with keys ['poses', 'intrinsics']
    returns: [open3d.geometry.TriangleMesh]
    """
    frames = [o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.25, np.zeros(3))]
    for i, T_WC in enumerate(data['poses']):
        if not i % flags.every == 0:
            continue
        print(f"Frame {i}", end="\r")
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, np.zeros(3))
        frames.append(mesh.transform(T_WC))
    return frames

# 포인트 클라우드 생성 및 병합
def point_clouds(flags, data):
    """
    Converts depth maps to point clouds and merges them all into one global point cloud.
    flags: command line arguments
    data: dict with keys ['intrinsics', 'poses']
    returns: [open3d.geometry.PointCloud]
    """
    pcs = []
    intrinsics = get_intrinsics(data['intrinsics'])
    pc = o3d.geometry.PointCloud()
    meshes = []
    # RGB 비디오 열기
    rgb_path = os.path.join(flags.path, 'rgb.mp4')
    # 각 프레임 처리
    video = skvideo.io.vreader(rgb_path)
    for i, (T_WC, rgb) in enumerate(zip(data['poses'], video)):
        # --every 옵션: 매 n번째 프레임만 처리
        if i % flags.every != 0:
            continue
        print(f"Point cloud {i}", end="\r")
        # 카메라 좌표계에서 월드 좌표계로의 역변환
        T_CW = np.linalg.inv(T_WC)
        # 신뢰도 맵 로드
        confidence = load_confidence(os.path.join(flags.path, 'confidence', f'{i:06}.png'))
        # Depth 이미지 로드 (신뢰도 필터링 적용)
        depth_path = data['depth_frames'][i]
        depth = load_depth(depth_path, confidence, filter_level=flags.confidence)
        # RGB 이미지를 Depth 크기로 리사이즈
        rgb = Image.fromarray(rgb)
        rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb = np.array(rgb)
        # RGB-D 이미지 생성
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), depth,
            depth_scale=1.0,        # Depth는 이미 미터 단위
            depth_trunc=MAX_DEPTH,  # 20m 이상 제거
            convert_rgb_to_intensity=False) # 컬러 유지
        # RGB-D에서 포인트 클라우드 생성 후 누적
        pc += o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, extrinsic=T_CW)
    return [pc]

# RGB-D 통합 (메쉬 생성)
def integrate(flags, data):
    """
    Integrates collected RGB-D maps using the Open3D integration pipeline.

    flags: command line arguments
    data: dict with keys ['intrinsics', 'poses']
    Returns: open3d.geometry.TriangleMesh
    """
    # TSDF (Truncated Signed Distance Function) Volume 생성
    # 3D 공간을 복셀로 나누어 표면을 표현
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=flags.voxel_size,
            sdf_trunc=0.05,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    intrinsics = get_intrinsics(data['intrinsics'])
     # RGB 비디오 열기
    rgb_path = os.path.join(flags.path, 'rgb.mp4')
    video = skvideo.io.vreader(rgb_path)
     # 모든 프레임을 TSDF Volume에 통합
    for i, (T_WC, rgb) in enumerate(zip(data['poses'], video)):
        print(f"Integrating frame {i:06}", end='\r')
        # Depth 로드
        depth_path = data['depth_frames'][i]
        depth = load_depth(depth_path)
        # RGB 리사이즈
        rgb = Image.fromarray(rgb)
        rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb = np.array(rgb)
        # RGB-D 이미지 생성
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), depth,
            depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False)
        # Volume에 현재 프레임 통합
        # np.linalg.inv(T_WC): 월드 -> 카메라 변환
        volume.integrate(rgbd, intrinsics, np.linalg.inv(T_WC))
    # Volume에서 삼각형 메쉬 추출 (Marching Cubes 알고리즘)
    mesh = volume.extract_triangle_mesh()
    # 노멀 벡터 계산 (조명/렌더링에 필요)
    mesh.compute_vertex_normals()
    return mesh

# 데이터셋 유효성 검사
def validate(flags):
    if not os.path.exists(os.path.join(flags.path, 'rgb.mp4')):
        absolute_path = os.path.abspath(flags.path)
        print(f"The directory {absolute_path} does not appear to be a directory created by the Stray Scanner app.")
        return False
    return True

# 메인 함수
def main():
     # 1. 커맨드 라인 인자 파싱
    flags = read_args()

     # 2. 데이터셋 유효성 검사
    if not validate(flags):
        return

    # 3. 옵션이 하나도 없으면 기본 시각화 활성화
    if not flags.frames and not flags.point_clouds and not flags.integrate:
        flags.frames = True
        flags.point_clouds = True
        flags.trajectory = True

    # 4. 데이터 로드
    data = read_data(flags)
    # 5. 시각화할 객체들 수집
    geometries = []
    if flags.trajectory:
        geometries += trajectory(flags, data)
    if flags.frames:
        geometries += show_frames(flags, data)
    if flags.point_clouds:
        geometries += point_clouds(flags, data)
    if flags.integrate:
        mesh = integrate(flags, data)
        # 메쉬 파일 저장 (옵션)
        if flags.mesh_filename is not None:
            o3d.io.write_triangle_mesh(flags.mesh_filename, mesh)
        geometries += [mesh]
    # 6. Open3D 뷰어로 시각화
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    main()

