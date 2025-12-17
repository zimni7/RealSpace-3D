"""
visualize_first_person.py
- Open3D를 이용한 1인칭 시점 3D 뷰어
- WASD 키로 이동, Q/E 키로 회전
- 경로 자동 인식 및 유연한 파일 로딩

사용:
  python visualize_first_person.py --scene room
  python visualize_first_person.py --file output/my_mesh.ply
"""

import open3d as o3d
import numpy as np
import sys
import argparse
from pathlib import Path

def run_viewer(file_path):
    # [1] 경로 객체 변환 및 확인
    file_path = Path(file_path)
    
    print(f"\n📂 1인칭 뷰어 실행: {file_path.name}")
    print(f"   경로: {file_path}")

    if not file_path.exists():
        print("❌ 오류: 파일을 찾을 수 없습니다.")
        return

    # [2] 메쉬 로딩
    try:
        mesh = o3d.io.read_triangle_mesh(str(file_path))
        # 만약 삼각형 메쉬가 아니라면 포인트 클라우드로 시도
        if len(mesh.vertices) == 0:
            print("⚠️ 메쉬 데이터가 비어있습니다. 포인트 클라우드로 다시 시도합니다.")
            pcd = o3d.io.read_point_cloud(str(file_path))
            if len(pcd.points) == 0:
                print("❌ 오류: 데이터가 비어있습니다.")
                return
            # 포인트 클라우드를 시각화하기 위해 geometry 교체
            geometry = pcd
            is_mesh = False
        else:
            geometry = mesh
            is_mesh = True
            
    except Exception as e:
        print(f"❌ 파일 로딩 중 에러: {e}")
        return

    print("\n🎮 조작 방법:")
    print("   [W / S] : 앞 / 뒤 이동")
    print("   [A / D] : 좌 / 우 이동")
    print("   [Q / E] : 좌 / 우 회전")
    print("   [ESC]   : 종료")
    print("="*60)

    # [3] 시각화 창 설정
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"First Person Viewer - {file_path.name}", width=1280, height=720)
    vis.add_geometry(geometry)

    # 렌더링 옵션
    opt = vis.get_render_option()
    opt.light_on = False  # 조명 끄기 (텍스처/색상 원본 보기)
    opt.mesh_show_back_face = True  # 뒷면도 렌더링
    
    if not is_mesh:
        opt.point_size = 3.0  # 점군일 경우 점 크기 키움

    # [4] 카메라 초기 위치 설정 (바닥 기준 1.6m 높이)
    ctr = vis.get_view_control()
    bounds = geometry.get_axis_aligned_bounding_box()
    min_bound = bounds.get_min_bound()
    center = geometry.get_center()

    eye = center.copy()
    eye[1] = min_bound[1] + 1.6  # 눈 높이 (사람 키)
    eye[2] -= 2.0  # 약간 뒤에서 시작

    lookat = center.copy()
    lookat[1] = eye[1]  # 시선은 수평

    ctr.set_lookat(lookat)
    ctr.set_front(lookat - eye)
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.1)  # 1인칭 느낌을 위해 줌 조정

    # [5] 이동 로직 (사용자가 수정한 방향 유지)
    # 이동 속도 설정
    step = 0.1 
    
    def move(vis, x, y, z):
        # camera_local_translate(우, 상, 후) 기준
        # x: +우/-좌, y: +상/-하, z: +후(뒤)/-전(앞)
        vis.get_view_control().camera_local_translate(x, y, z)
        return False
        
    def rotate(vis, deg):
        # deg: 픽셀 단위 마우스 드래그 시뮬레이션
        # x축 회전은 막고(0), y축(좌우) 회전만 허용
        vis.get_view_control().rotate(deg, 0)
        return False

    # ✅ 사용자 지정 방향 매핑 (작성해주신 내용 그대로 유지)
    # W: 앞으로 (Open3D 카메라 기준 z축 +방향으로 이동 -> 줌인 효과)
    vis.register_key_callback(ord("W"), lambda v: move(v, 0, 0, step))  
    # S: 뒤로
    vis.register_key_callback(ord("S"), lambda v: move(v, 0, 0, -step)) 
    # A: 왼쪽
    vis.register_key_callback(ord("A"), lambda v: move(v, -step, 0, 0)) 
    # D: 오른쪽
    vis.register_key_callback(ord("D"), lambda v: move(v, step, 0, 0))  
    
    # Q/E 회전
    vis.register_key_callback(ord("Q"), lambda v: rotate(v, -20))
    vis.register_key_callback(ord("E"), lambda v: rotate(v, 20))

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # [1] 기본 경로 설정
    BASE_DIR = Path(__file__).resolve().parent
    DEFAULT_SCENE = "room"
    
    # [2] 인자 파싱
    parser = argparse.ArgumentParser(description="1인칭 3D 뷰어")
    parser.add_argument("--scene", type=str, default=DEFAULT_SCENE, help="대상 현장 (room, class, lab)")
    parser.add_argument("--file", type=str, default=None, help="직접 파일 경로 지정")
    args = parser.parse_args()

    # [3] 파일 찾기 로직
    if args.file:
        target_file = Path(args.file)
    else:
        # 우선순위: 1. 텍스처 입힌 결과 -> 2. 형상 재구성 결과 -> 3. 원본 점군
        scene_dir = BASE_DIR / "output"
        
        # 후보 파일 목록
        candidates = [
            scene_dir / f"{args.scene}_wall_textures_out" / "textured_room_vertexcolor.ply",  # 텍스처 결과
            scene_dir / f"{args.scene}_detected_results" / "final_shape_room.ply",            # 메쉬 결과
            scene_dir / f"{args.scene}_detected_results" / "full_room_structure.ply"          # 점군 결과
        ]
        
        target_file = None
        for cand in candidates:
            if cand.exists():
                target_file = cand
                break
        
        if target_file is None:
            print(f"❌ '{args.scene}'에 대한 결과 파일을 찾을 수 없습니다.")
            print(f"   탐색 경로: output/{args.scene}_wall_textures_out/ 등")
            sys.exit(1)

    # [4] 뷰어 실행
    run_viewer(target_file)