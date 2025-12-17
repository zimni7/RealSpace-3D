import numpy as np
import open3d as o3d
import cv2
import pickle
import json
import sys
from pathlib import Path  # [핵심] Pathlib 사용
from mapbox_earcut import triangulate_float32
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull

class GridBasedRoomReconstructor:
    """
    완벽한 직각 방 재구성 (히스토그램 기반)
    
    기존 detection 코드로 생성된 walls_data.pkl을 읽어서
    완벽한 직각 코너를 가진 3D 메쉬 생성
    """
    
    def __init__(self, pkl_path):
        # [수정] Path 객체로 변환 및 절대 경로 확인
        self.pkl_path = Path(pkl_path).resolve()
        self.load_data()
        
    def load_data(self):
        """Detection 코드에서 생성한 pkl 파일 로드"""
        if not self.pkl_path.exists():
            print(f"❌ [Error] 파일을 찾을 수 없습니다: {self.pkl_path}")
            sys.exit(1)
            
        print(f"📂 데이터 로딩: {self.pkl_path.name}")
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        all_wall_points = []
        for w in data['walls']:
            all_wall_points.append(w['points'])
            
        if not all_wall_points:
            raise ValueError("벽 데이터가 비어있습니다.")
            
        self.wall_points = np.vstack(all_wall_points)
        self.floor_height = data['floor_height']
        self.ceiling_height = data['ceiling_height']
        
        print(f"✅ 데이터 로드 완료: {len(self.wall_points):,}개 포인트")

    def find_room_corners(self):
        """
        히스토그램 피크 검출로 완벽한 직각 코너 생성
        """
        print("\n📐 완벽한 직각 코너 계산 중...")
        
        x = self.wall_points[:, 0]
        z = self.wall_points[:, 2]
        
        # ===== X축 평행 벽 검출 =====
        x_min, x_max = x.min(), x.max()
        bin_width = 0.05  # 5cm 단위
        n_bins_x = int((x_max - x_min) / bin_width) + 1
        
        x_hist, x_edges = np.histogram(x, bins=n_bins_x, range=(x_min, x_max))
        
        # 피크 검출
        peaks_x, _ = find_peaks(
            x_hist,
            height=len(x) * 0.01,  # 전체의 1% 이상
            distance=int(0.3 / bin_width)  # 최소 30cm 간격
        )
        
        x_planes = []
        for peak_idx in peaks_x:
            x_pos = x_edges[peak_idx] + bin_width / 2
            x_planes.append(x_pos)
        
        print(f"   🔹 X축 평행 벽: {len(x_planes)}개")
        
        # ===== Z축 평행 벽 검출 =====
        z_min, z_max = z.min(), z.max()
        n_bins_z = int((z_max - z_min) / bin_width) + 1
        
        z_hist, z_edges = np.histogram(z, bins=n_bins_z, range=(z_min, z_max))
        
        peaks_z, _ = find_peaks(
            z_hist,
            height=len(z) * 0.01,
            distance=int(0.3 / bin_width)
        )
        
        z_planes = []
        for peak_idx in peaks_z:
            z_pos = z_edges[peak_idx] + bin_width / 2
            z_planes.append(z_pos)
        
        print(f"   🔹 Z축 평행 벽: {len(z_planes)}개")
        
        # ===== 평면 교차점 = 완벽한 직각 코너 =====
        all_intersections = []
        for x_pos in x_planes:
            for z_pos in z_planes:
                all_intersections.append([x_pos, z_pos])
        
        all_intersections = np.array(all_intersections)
        
        # 실제 벽 포인트 근처에 있는 교차점만 선택
        wall_points_2d = self.wall_points[:, [0, 2]]
        
        valid_corners = []
        for intersection in all_intersections:
            distances = np.linalg.norm(wall_points_2d - intersection, axis=1)
            min_distance = distances.min()
            
            # 0.8m 이내에 벽 포인트가 있으면 유효
            if min_distance < 0.8:
                valid_corners.append(intersection)
        
        if len(valid_corners) == 0:
            raise ValueError("❌ 유효한 코너를 찾을 수 없습니다!")
            
        valid_corners = np.array(valid_corners)
        
        # Convex Hull로 외곽만 선택
        if len(valid_corners) >= 3:
            hull = ConvexHull(valid_corners)
            hull_corners = valid_corners[hull.vertices]
            
            # 반시계 방향 정렬
            center = hull_corners.mean(axis=0)
            angles = np.arctan2(hull_corners[:, 1] - center[1],
                               hull_corners[:, 0] - center[0])
            sorted_idx = np.argsort(angles)
            self.corners = hull_corners[sorted_idx]
        else:
            self.corners = valid_corners
        
        print(f"   ✅ 최종 코너: {len(self.corners)}개 (완벽한 직각!)")

    def calculate_polygon_area(self, corners):
        """Shoelace formula로 폴리곤 면적 계산"""
        x = corners[:, 0]
        y = corners[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def save_dimensions(self, output_path):
        """방의 치수 정보를 JSON으로 저장"""
        # [수정] output_path가 Path 객체인지 확인
        output_path = Path(output_path)
        print(f"\n📝 치수 데이터 계산 중...")
        
        room_height = self.ceiling_height - self.floor_height
        floor_area = self.calculate_polygon_area(self.corners)
        
        walls_info = []
        n = len(self.corners)
        for i in range(n):
            p1 = self.corners[i]
            p2 = self.corners[(i+1) % n]
            width = np.linalg.norm(p1 - p2)
            
            walls_info.append({
                "wall_id": i + 1,
                "width_m": round(float(width), 3),
                "height_m": round(float(room_height), 3),
                "area_m2": round(float(width * room_height), 3),
                "start_point": [float(p1[0]), float(p1[1])],
                "end_point": [float(p2[0]), float(p2[1])]
            })
            
        data = {
            "room_summary": {
                "floor_area_m2": round(float(floor_area), 3),
                "room_height_m": round(float(room_height), 3)
            },
            "walls": walls_info
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"   ✅ 치수 저장 완료: {output_path.name}")

    def create_mesh(self):
        """3D 트라이앵글 메쉬 생성 (Floor + Ceiling + Walls)"""
        print("\n🏗️ 3D 메쉬 생성 중...")
        verts_2d = self.corners.astype(np.float32)
        rings = np.array([len(verts_2d)], dtype=np.uint32)

        try:
            tri_indices = triangulate_float32(verts_2d, rings)
            tri_indices = tri_indices.reshape(-1, 3)
        except Exception as e:
            print(f"   ❌ Earcut 에러: {e}")
            return None
        
        n_corners = len(self.corners)

        # 1. Floor (바닥)
        floor_vertices = []
        for x, z in self.corners: 
            floor_vertices.append([x, self.floor_height, z])
        
        floor_triangles = []
        for tri in tri_indices: 
            floor_triangles.append([tri[2], tri[1], tri[0]])  # 법선 반전

        floor_mesh = o3d.geometry.TriangleMesh()
        floor_mesh.vertices = o3d.utility.Vector3dVector(floor_vertices)
        floor_mesh.triangles = o3d.utility.Vector3iVector(floor_triangles)
        floor_mesh.compute_vertex_normals()
        floor_mesh.paint_uniform_color([0.6, 0.4, 0.2])  # 갈색

        # 2. Ceiling (천장)
        ceiling_vertices = []
        for x, z in self.corners: 
            ceiling_vertices.append([x, self.ceiling_height, z])
            
        ceiling_triangles = []
        for tri in tri_indices: 
            ceiling_triangles.append([tri[0], tri[1], tri[2]])

        ceiling_mesh = o3d.geometry.TriangleMesh()
        ceiling_mesh.vertices = o3d.utility.Vector3dVector(ceiling_vertices)
        ceiling_mesh.triangles = o3d.utility.Vector3iVector(ceiling_triangles)
        ceiling_mesh.compute_vertex_normals()
        ceiling_mesh.paint_uniform_color([0.85, 0.85, 0.85])  # 회색

        # 3. Walls (벽)
        wall_vertices = []
        wall_triangles = []
        for i in range(n_corners):
            curr_x, curr_z = self.corners[i]
            next_x, next_z = self.corners[(i+1) % n_corners]
            
            base_idx = len(wall_vertices)
            wall_vertices.append([curr_x, self.floor_height, curr_z])
            wall_vertices.append([next_x, self.floor_height, next_z])
            wall_vertices.append([curr_x, self.ceiling_height, curr_z])
            wall_vertices.append([next_x, self.ceiling_height, next_z])
            
            wall_triangles.append([base_idx, base_idx+1, base_idx+2])
            wall_triangles.append([base_idx+2, base_idx+1, base_idx+3])

        wall_mesh = o3d.geometry.TriangleMesh()
        wall_mesh.vertices = o3d.utility.Vector3dVector(wall_vertices)
        wall_mesh.triangles = o3d.utility.Vector3iVector(wall_triangles)
        wall_mesh.compute_vertex_normals()
        wall_mesh.paint_uniform_color([0.6, 0.8, 1.0])  # 파란색

        final_mesh = floor_mesh + ceiling_mesh + wall_mesh
        return final_mesh

    # [수정] output_dir 인자 추가
    def run(self, output_dir=None):
        """전체 파이프라인 실행"""
        print("\n" + "="*60)
        print("🚀 완벽한 직각 방 재구성")
        print("="*60)
        
        self.find_room_corners()    # 히스토그램 기반 코너 검출
        mesh = self.create_mesh()   # 3D 메쉬 생성
        
        if mesh is not None:
            # output_dir가 없으면 pkl 파일이 있는 곳을 기본값으로 사용
            if output_dir is None:
                output_dir = self.pkl_path.parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n📂 결과 저장 경로: {output_dir}")

            json_path = output_dir / "room_dimensions.json"
            self.save_dimensions(json_path)
            
            ply_path = output_dir / "final_shape_room.ply"
            o3d.io.write_triangle_mesh(str(ply_path), mesh)
            print(f"✨ 모델 파일 저장: {ply_path.name}")
            
            print("\n" + "="*60)
            print("✅ 완벽한 직각 방 재구성 완료!")
            print("="*60)
            
            return mesh
        else:
            return None

if __name__ == "__main__":
    # [1] 경로 자동 설정 (Dynamic Path Config)
    BASE_DIR = Path(__file__).resolve().parent
    
    # 예시: room 데이터 실행
    TARGET_SCENE = "room"  # class, lab, room 중 선택
    
    # 입력: output/room_detected_results/walls_data.pkl
    PKL_PATH = BASE_DIR / "output" / f"{TARGET_SCENE}_detected_results" / "walls_data.pkl"
    
    # 출력: 같은 폴더에 저장 (또는 원하는 곳으로 지정 가능)
    OUTPUT_DIR = PKL_PATH.parent
    
    print(f"📂 Project Root: {BASE_DIR}")
    print(f"📂 Input PKL:    {PKL_PATH}")
    
    # [2] pkl 파일 존재 확인
    if not PKL_PATH.exists():
        print(f"❌ 오류: 입력 파일을 찾을 수 없습니다.")
        print(f"   경로: {PKL_PATH}")
        print("   팁: structure_detection.py를 먼저 실행하세요.")
        sys.exit(1)
    
    reconstructor = GridBasedRoomReconstructor(PKL_PATH)
    
    # [3] 실행 (출력 경로 지정)
    mesh = reconstructor.run(output_dir=OUTPUT_DIR)
    
    if mesh:
        print("\n🎮 시각화 (바닥:갈색, 천장:회색, 벽:파란색)")
        print("   💡 모든 코너가 완벽한 직각입니다!")
        o3d.visualization.draw_geometries(
            [mesh], 
            window_name="Perfect Rectangular Room", 
            width=1024, 
            height=768, 
            mesh_show_back_face=True
        )