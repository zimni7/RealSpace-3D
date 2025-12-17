# 🏗️ RealSpace-3D
> **Turning Real-World Spaces into Digital 3D Models with Perfect Rectilinear Structures**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Open3D](https://img.shields.io/badge/Library-Open3D-green)](http://www.open3d.org/)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

**RealSpace-3D**는 **StrayScanner** 앱으로 스캔한 RGB-D 데이터를 활용하여 실내 공간을 3D로 시각화하고, 구조(벽, 바닥, 천장)를 자동으로 인식하여 **완벽한 직각(Rectilinear) 메쉬**로 재구성하는 파이썬 프로젝트다.

---

## 🔄 프로젝트 워크플로우 (Project Workflow)

본 프로젝트는 다음과 같은 순차적인 파이프라인으로 구성된다.

```mermaid
graph LR
    A[StrayScanner Data] --> B(stray_visualize.py<br/>원본 데이터 확인)
    B --> C(structure_detection.py<br/>구조 감지 및 저장)
    C --> D(structure_reconstruction.py<br/>3D 메쉬 재구성)
    D --> E(compare_results.py<br/>결과 비교 검증)
    E --> F{run_full_pipeline.py<br/>최종 통합 실행}

1. 데이터 취득: StrayScanner 앱을 통해 공간을 스캔하고 PC로 가져온다.
2. 전처리 및 시각화: 원본 포인트 클라우드를 3D로 시각화하여 데이터 상태를 확인한다.
3. 구조 감지 (Detection): 벽, 바닥, 천장 평면을 감지하고 정보를 추출한다.
4. 재구성 (Reconstruction): 추출된 정보를 바탕으로 노이즈 없는 직각 3D 모델을 생성한다.
5. 텍스처 복원 (Texturing): 생성된 모델에 실제 영상의 텍스처를 입히고 빈 곳을 복원한다.

📂 파일 구성 및 역할 (File Description)
1. 시각화 및 검증 도구 (Visualization tools)
stray_visualize.py,[1단계] StrayScanner로 찍은 원본 데이터(Point Cloud)를 3D로 시각화하여 확인한다.
visualize_first_person.py,데이터나 결과물을 **1인칭 시점(WASD 키)**으로 이동하며 구조를 면밀히 살펴볼 수 있는 뷰어다.
compare_results.py,원본 데이터와 재구성된 결과물을 나란히 놓고 비교 분석하는 검증 코드다.

2. 핵심 처리 모듈 (Core Processing)
structure_detection.py,"[2단계] 공간의 뼈대(바닥/천장 높이, 벽면 위치)를 분석하여 데이터를 저장한다."
structure_reconstruction.py,[3단계] 감지된 정보를 바탕으로 3D 메쉬(Mesh)를 생성하고 저장한다.

3. 최종 통합 파이프라인 (Final Pipeline)
run_full_pipeline.py,[최종] 위 모든 과정(감지 → 재구성 → 텍스처링)을 한 번에 수행하는 마스터 코드다.
wall_texture_restoration.py,(Internal) 최종 파이프라인 내부에서 텍스처 매핑 및 조명 보정을 수행한다.
texture_hole_report.py,(Internal) 텍스처 품질을 분석하고 빈 영역(Hole) 리포트를 생성한다.

💻 실행 방법 (How to Run)

0. 환경 설정
pip install -r requirements.txt

1. 원본 데이터 확인 (Data Visualization)
python stray_visualize.py --input data/room

2. 단계별 실행 (Step-by-Step Execution)
Step A: 구조 감지 (Detection) 벽과 바닥 정보를 분석하여 저장한다.
python structure_detection.py
Step B: 3D 재구성 (Reconstruction) 저장된 정보를 바탕으로 3D 모델을 생성한다.
python structure_reconstruction.py
Step C: 결과 비교 (Comparison) 재구성 전후를 비교하여 알고리즘 성능을 검증한다.
python compare_results.py
3. 최종 통합 실행 (Full Pipeline)
python run_full_pipeline.py

📊 시연용 데이터셋 (Demo Dataset)
Input: data/room/ (StrayScanner Raw Data)
Output: output/room_*/ (Detected & Reconstructed Results








