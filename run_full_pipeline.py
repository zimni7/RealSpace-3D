"""
run_full_pipeline.py
- CLASS, ROOM, LAB 세 버전을 한 번에 실행 (데이터가 있는 경우만)
- 경로 자동 감지 (Pathlib 사용)
- 3가지 개선사항 적용 (바닥 밝기, 검은 벽 방지, 천장 제거)

실행:
    python run_full_pipeline.py
"""

from pathlib import Path
import sys
import time

# ==================== 1. 경로 자동 설정 (Dynamic Path Config) ====================
# 현재 파일(run_full_pipeline.py)이 있는 위치를 기준으로 경로를 잡습니다.
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
OUTPUT_ROOT = BASE_DIR / "output"

# 디버깅: 현재 인식된 루트 경로 출력
print(f"📂 Project Root: {BASE_DIR}")

# 설정 딕셔너리
VERSIONS = {
    "class": {
        "dataset": DATA_ROOT / "class",
        "detected_dir": OUTPUT_ROOT / "class_detected_results",
        "dimensions_json": "room_dimensions.json", 
        "output_dir": OUTPUT_ROOT / "class_wall_textures_out",
    },
    "room": {
        "dataset": DATA_ROOT / "room",
        "detected_dir": OUTPUT_ROOT / "room_detected_results",
        "dimensions_json": "room_dimensions.json",
        "output_dir": OUTPUT_ROOT / "room_wall_textures_out",
    },
    "lab": {
        "dataset": DATA_ROOT / "lab",
        "detected_dir": OUTPUT_ROOT / "lab_detected_results",
        "dimensions_json": "room_dimensions.json", 
        "output_dir": OUTPUT_ROOT / "lab_wall_textures_out",
    },
}

def check_requirements(version_name, config):
    """필수 파일 확인"""
    dataset_path = config["dataset"]
    detected_dir = config["detected_dir"]
    
    # 1차 확인: 데이터셋 폴더 자체가 있는지 확인
    if not dataset_path.exists():
        # 폴더가 아예 없으면 조용히 False 반환 (메인 함수에서 처리)
        return "MISSING_DIR"

    # 2차 확인: 폴더는 있는데 필수 파일이 없는지 확인
    required_files = {
        'walls_data.pkl': detected_dir / 'walls_data.pkl',
        'dimensions.json': detected_dir / config["dimensions_json"],
        'camera_matrix.csv': dataset_path / 'camera_matrix.csv',
        'odometry.csv': dataset_path / 'odometry.csv',
        'rgb.mp4': dataset_path / 'rgb.mp4',
    }
    
    missing = []
    for name, path in required_files.items():
        if not path.exists():
            missing.append(f"   ❌ {name}: {path}")
    
    if missing:
        print(f"\n🔴 [{version_name.upper()}] 필수 파일 누락:")
        for m in missing:
            print(m)
        return "MISSING_FILES"
    
    return "OK"


def verify_alignment_data(config):
    """Alignment 정보 확인"""
    import pickle
    pkl_path = config["detected_dir"] / 'walls_data.pkl'
    
    if not pkl_path.exists():
        return False
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'alignment' not in data:
            return False
        
        alignment = data['alignment']
        required_keys = ['centroid', 'R', 'rotation_angle_rad']
        
        for key in required_keys:
            if key not in alignment:
                return False
        return True
    except Exception as e:
        print(f"⚠️ pkl 파일 읽기 오류: {e}")
        return False


def run_texture_restoration(version_name, config):
    """텍스처 복원 실행"""
    # 같은 폴더에 있는 wall_texture_restoration.py 불러오기
    try:
        from wall_texture_restoration import WallTextureRestorerEnhanced
    except ImportError:
        print("❌ 오류: 'wall_texture_restoration.py' 파일을 찾을 수 없습니다.")
        print(f"   현재 위치: {BASE_DIR}")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"🎨 [{version_name.upper()}] Texture Restoration")
    print("="*60)
    
    # Path 객체를 문자열(str)로 변환하여 전달 (라이브러리 호환성 위해)
    restorer = WallTextureRestorerEnhanced(
        dataset_path=str(config["dataset"]),
        detected_dir=str(config["detected_dir"]),
        recon_json=str(config["detected_dir"] / config["dimensions_json"]),
        out_dir=str(config["output_dir"]),
    )
    
    print(f"\n📋 [{version_name.upper()}] 실행 설정:")
    print("  🔥 개선사항 적용:")
    print("    - 바닥 밝기 개선 (weight threshold)")
    print("    - 검은 벽면 방지 (default color)")
    print("    - 천장 제거 (open ceiling)")
    print()
    
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
    
    print(f"\n✅ [{version_name.upper()}] 텍스처링 완료!")
    print(f"📁 출력: {config['output_dir']}")
    
    return mesh


def run_quality_report(version_name, config):
    """품질 리포트 생성"""
    print(f"\n📊 [{version_name.upper()}] Quality Report")
    
    try:
        import texture_hole_report as report_module
        
        class Args:
            dir = str(config["output_dir"])  # Path -> str 변환
            out = str(config["output_dir"])
            json_name = "texture_hole_report.json"
            csv_name = "texture_hole_report.csv"
        
        import argparse
        original_parse = argparse.ArgumentParser.parse_args
        
        def mock_parse(self, args=None, namespace=None):
            return Args()
        
        argparse.ArgumentParser.parse_args = mock_parse
        
        try:
            report_module.main()
        finally:
            argparse.ArgumentParser.parse_args = original_parse
            
        print(f"✅ [{version_name.upper()}] 리포트 생성 완료!")
        
    except ImportError:
        print(f"⚠️  [{version_name.upper()}] 'texture_hole_report.py'가 없어서 리포트를 건너뜁니다.")
    except Exception as e:
        print(f"⚠️  [{version_name.upper()}] 리포트 생성 실패: {e}")


def visualize_mesh(version_name, mesh):
    """메시 시각화"""
    try:
        import open3d as o3d
        print(f"\n🎮 [{version_name.upper()}] 3D 뷰어 실행 (창을 닫으면 종료됩니다)")
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=f"{version_name.upper()} - Final Result",
            width=1280,
            height=720,
            mesh_show_back_face=True,
        )
    except Exception as e:
        print(f"⚠️  [{version_name.upper()}] 시각화 실패: {e}")


def main():
    print("="*60)
    print("🏠 RealSpace-3D: Full Pipeline Runner")
    print("   (경로에 데이터가 있는 버전만 자동으로 실행합니다)")
    print("="*60)
    
    results = {}
    meshes = {}
    
    # 각 버전 처리
    for version_name, config in VERSIONS.items():
        print(f"\n\n{'='*60}")
        print(f"🚀 Processing: {version_name.upper()}")
        print("="*60)
        
        # 1. 파일 및 데이터 확인
        status = check_requirements(version_name, config)
        
        if status == "MISSING_DIR":
            print(f"⏭️  [{version_name.upper()}] 데이터 폴더가 없습니다. 건너뜁니다.")
            results[version_name] = "SKIPPED (No Data)"
            continue
        elif status == "MISSING_FILES":
            print(f"⏭️  [{version_name.upper()}] 필수 파일 부족으로 건너뜁니다.")
            results[version_name] = "SKIPPED (Missing Files)"
            continue
        
        # 2. Alignment 확인
        if not verify_alignment_data(config):
            results[version_name] = "SKIPPED (No Alignment)"
            print(f"⚠️  [{version_name.upper()}] Alignment 정보가 없습니다 (walls_data.pkl 확인 필요).")
            print(f"   -> 먼저 structure_detection.py를 실행하세요.")
            continue
        
        # 3. 텍스처 복원 실행
        start_time = time.time()
        try:
            mesh = run_texture_restoration(version_name, config)
            meshes[version_name] = mesh
            
            # 4. 품질 리포트
            run_quality_report(version_name, config)
            
            elapsed = time.time() - start_time
            results[version_name] = f"SUCCESS ({elapsed:.1f}s)"
            
        except Exception as e:
            print(f"\n❌ [{version_name.upper()}] 처리 중 에러 발생: {e}")
            import traceback
            traceback.print_exc()
            results[version_name] = "FAILED"
    
    # 최종 결과 요약
    print("\n\n" + "="*60)
    print("🎉 모든 작업 완료!")
    print("="*60)
    
    print("\n📊 실행 결과 요약:")
    for version, result in results.items():
        if "SUCCESS" in result:
            icon = "✅"
        elif "SKIPPED" in result:
            icon = "⏭️ "
        else:
            icon = "❌"
        print(f"  {icon} {version.upper():<5}: {result}")
    
    print("\n📁 결과물 저장 경로:")
    for version, config in VERSIONS.items():
        if version in results and "SUCCESS" in results[version]:
            print(f"  - {config['output_dir']}")
    
    # 결과 시각화 (선택)
    if meshes:
        print("\n🎮 결과 확인 (시각화할 버전을 선택하세요):")
        keys = list(meshes.keys())
        for idx, version in enumerate(keys, 1):
            print(f"  {idx}. {version.upper()}")
        print(f"  0. 종료")
        
        try:
            choice = input(f"\n선택 (0-{len(keys)}): ")
            choice_idx = int(choice)
            if choice_idx > 0 and choice_idx <= len(keys):
                selected = keys[choice_idx - 1]
                visualize_mesh(selected, meshes[selected])
            else:
                print("종료합니다.")
        except:
            print("종료합니다.")
    else:
        print("\n⚠️  생성된 결과물이 없습니다. 데이터 경로를 확인해주세요.")

if __name__ == "__main__":
    main()