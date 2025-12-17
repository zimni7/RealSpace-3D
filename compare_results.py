import open3d as o3d
import sys
from pathlib import Path

def check_alignment(target_scene="room"):
    # [1] 경로 자동 설정 (현재 파일 위치 기준)
    BASE_DIR = Path(__file__).resolve().parent
    
    # 예: output/room_detected_results 폴더 조준
    TARGET_DIR = BASE_DIR / "output" / f"{target_scene}_detected_results"
    
    print(f"📂 타겟 디렉토리: {TARGET_DIR}")

    # 1. Detector가 만든 원본 점군 (Raw Point Cloud)
    pcd_path = TARGET_DIR / "full_room_structure.ply"
    # 2. Reconstructor가 만든 최종 메쉬 (Reconstructed Mesh)
    mesh_path = TARGET_DIR / "final_shape_room.ply"

    # [2] 파일 존재 여부 확인 (Path 객체 사용)
    if not pcd_path.exists():
        print(f"❌ 점군 파일을 찾을 수 없습니다: {pcd_path.name}")
        return
    if not mesh_path.exists():
        print(f"❌ 메쉬 파일을 찾을 수 없습니다: {mesh_path.name}")
        return

    print("📂 파일 로딩 중...")
    try:
        # Open3D는 경로를 문자열(str)로 주어야 안전함
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
            
    except Exception as e:
        print(f"❌ 파일 읽기 중 에러 발생: {e}")
        return

    # [3] 시각적 디버깅을 위한 스타일 설정
    # 점군(Raw Data) = 빨간색 (잘 보임)
    pcd.paint_uniform_color([1, 0, 0])       
    
    # 메쉬(Result) = 회색 (반투명 느낌)
    mesh.paint_uniform_color([0.8, 0.8, 0.8]) 
    
    print("\n" + "="*60)
    print(f"🎮 [{target_scene.upper()}] 정렬 확인 시작")
    print("="*60)
    print("   🔴 빨간 점 (원본 데이터)")
    print("   ⚪ 회색 면 (재구성된 방)")
    print("   👉 빨간 점들이 회색 벽/바닥 표면에 딱 붙어 있어야 성공입니다.")
    print("="*60)
    
    # 시각화 실행
    o3d.visualization.draw_geometries(
        [pcd, mesh], 
        window_name=f"Alignment Check - {target_scene.upper()}", 
        width=1280, 
        height=720,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    # 보고 싶은 현장 이름을 여기서 바꾸세요 ("room", "class", "lab")
    SCENE_NAME = "room"
    
    check_alignment(SCENE_NAME)