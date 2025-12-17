"""
texture_hole_report.py
- 바닥 텍스처 품질을 더 정확하게 분석
- 벽/바닥 별도 임계값 적용
- 시각화 리포트 생성 (히트맵, 그래프)
- [수정] 경로 자동 인식 및 한글 폰트 지원 추가

사용:
  python texture_hole_report.py --dir output/room_wall_textures_out
  또는 그냥 실행 (기본 경로 room 사용)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import sys
import platform

# [1] 한글 폰트 설정 (그래프 깨짐 방지)
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')  # GUI 없이 저장만

try:
    import pandas as pd
except Exception:
    pd = None


@dataclass
class HoleStats:
    name: str
    surface_type: str  # "wall" or "floor"
    mask_path: str
    baked_path: str | None
    final_path: str | None
    hole_ratio_pct: float
    hole_pixels: int
    total_pixels: int
    components: int
    largest_comp_ratio_pct: float
    mean_comp_area_px: float
    edge_density_pct: float | None
    laplacian_var: float | None
    dominant_line_strength: float | None
    recommendation: str
    confidence: str
    quality_score: float  # 0-100
    notes: str


def _connected_components_stats(hole_mask_u8: np.ndarray) -> tuple[int, float, float]:
    """Hole components analysis"""
    bin_mask = (hole_mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    comp = max(0, num - 1)
    if comp == 0:
        return 0, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    total = float(bin_mask.size)
    largest_ratio = float(areas.max() / total * 100.0)
    mean_area = float(areas.mean())
    return comp, largest_ratio, mean_area


def _edge_metrics(img_rgb_u8: np.ndarray) -> tuple[float, float, float]:
    """Texture complexity metrics"""
    gray = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2GRAY)

    # Edge density
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean() * 100.0)

    # Laplacian variance (texture complexity)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    # Hough lines
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180.0, threshold=120,
        minLineLength=max(30, min(gray.shape)//20), maxLineGap=10
    )
    if lines is None or len(lines) == 0:
        line_strength = 0.0
    else:
        diag = (gray.shape[0]**2 + gray.shape[1]**2) ** 0.5
        lens = []
        for x1, y1, x2, y2 in lines[:, 0, :]:
            lens.append(((x2-x1)**2 + (y2-y1)**2) ** 0.5)
        line_strength = float(np.clip(np.sum(lens) / (diag * 10.0), 0.0, 1.0))
    return edge_density, lap_var, line_strength


def _classify_and_recommend(
    surface_type: str,
    hole_ratio_pct: float,
    largest_comp_ratio_pct: float,
    edge_density_pct: float | None,
    line_strength: float | None
) -> tuple[str, str, float, str]:
    """
    Returns:
      recommendation: text
      confidence: "high/medium/low"
      quality_score: 0-100
      notes: extra caution notes
    """
    notes = []

    # 🔥 Surface-specific thresholds
    if surface_type == "floor":
        # 바닥은 더 관대한 기준 (넓은 면적, occlusion 많음)
        quality_thresholds = [15.0, 35.0]  # high: <15%, medium: 15-35%, low: >35%
    else:  # wall
        quality_thresholds = [5.0, 20.0]

    # Quality score (0-100)
    if hole_ratio_pct <= quality_thresholds[0]:
        quality_score = 100.0 - hole_ratio_pct * 2.0
        confidence = "high"
    elif hole_ratio_pct <= quality_thresholds[1]:
        quality_score = 80.0 - (hole_ratio_pct - quality_thresholds[0]) * 3.0
        confidence = "medium"
    else:
        quality_score = max(0.0, 40.0 - (hole_ratio_pct - quality_thresholds[1]) * 1.0)
        confidence = "low"

    quality_score = float(np.clip(quality_score, 0.0, 100.0))

    # Structured texture detection
    structured = False
    if edge_density_pct is not None and edge_density_pct >= 6.0:
        structured = True
        notes.append("⚠️ 엣지 밀도가 높아 직선 구조(줄눈/몰딩) 보존 필요")
    if line_strength is not None and line_strength >= 0.25:
        structured = True
        notes.append("⚠️ 직선 패턴이 강해 inpainting 대신 추가 촬영 권장")

    # Recommendations
    if surface_type == "floor":
        if hole_ratio_pct <= 15.0:
            rec = "✅ 바닥 관측률 양호. 현재 베이킹 + 인페인팅으로 충분."
        elif hole_ratio_pct <= 35.0:
            rec = "⚠️ 일부 비가시 영역 존재. 카메라를 낮게 들고 추가 촬영 권장."
            if structured:
                rec += " 타일/패턴이 있다면 LaMa/MAT 인페인팅 권장."
        else:
            rec = "❌ 바닥 hole 비율 높음. 낮은 시점(0.5m)에서 추가 촬영 필수."
            if structured:
                rec += " 단순 인페인팅은 패턴이 어긋날 가능성이 큼."
    else:  # wall
        if hole_ratio_pct <= 5.0:
            rec = "✅ 벽면 대부분 관측됨. 현재 베이킹으로 자연스러운 결과 예상."
        elif hole_ratio_pct <= 20.0:
            rec = "⚠️ 일부 미관측 영역. 프레임 수를 늘리거나 각도 다양화 권장."
            if structured:
                rec += " 구조 보존이 중요하면 LaMa/MAT 인페인팅 권장."
        else:
            rec = "❌ 비가시 영역이 큼. 추가 촬영으로 관측을 늘리는 것이 가장 효과적."
            if structured:
                rec += " 단순 inpaint는 티가 날 가능성이 큼."

    # Largest component warning
    if largest_comp_ratio_pct >= 8.0:
        notes.append("⚠️ 가장 큰 hole 덩어리가 커서 인페인팅 티가 날 확률 높음")
        quality_score *= 0.9  # 10% 페널티

    return rec, confidence, quality_score, " ".join(notes)


def analyze_one(mask_path: Path, baked_path: Path | None, final_path: Path | None) -> HoleStats:
    """Analyze single texture"""
    # Detect surface type
    name = mask_path.stem.replace("_mask", "")
    surface_type = "floor" if "floor" in name.lower() else "wall"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read mask: {mask_path}")

    total_pixels = int(mask.size)
    hole_pixels = int((mask > 0).sum())
    hole_ratio_pct = float(hole_pixels / max(1, total_pixels) * 100.0)

    comps, largest_ratio, mean_area = _connected_components_stats(mask)

    edge_density = None
    lap_var = None
    line_strength = None

    img_for_metrics = None
    if baked_path is not None and baked_path.exists():
        bgr = cv2.imread(str(baked_path), cv2.IMREAD_COLOR)
        if bgr is not None:
            img_for_metrics = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if img_for_metrics is None and final_path is not None and final_path.exists():
        bgr = cv2.imread(str(final_path), cv2.IMREAD_COLOR)
        if bgr is not None:
            img_for_metrics = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if img_for_metrics is not None:
        edge_density, lap_var, line_strength = _edge_metrics(img_for_metrics)

    rec, conf, quality, notes = _classify_and_recommend(
        surface_type, hole_ratio_pct, largest_ratio, edge_density, line_strength
    )

    return HoleStats(
        name=name,
        surface_type=surface_type,
        mask_path=str(mask_path),
        baked_path=str(baked_path) if baked_path is not None else None,
        final_path=str(final_path) if final_path is not None else None,
        hole_ratio_pct=hole_ratio_pct,
        hole_pixels=hole_pixels,
        total_pixels=total_pixels,
        components=comps,
        largest_comp_ratio_pct=largest_ratio,
        mean_comp_area_px=float(mean_area),
        edge_density_pct=edge_density,
        laplacian_var=lap_var,
        dominant_line_strength=line_strength,
        recommendation=rec,
        confidence=conf,
        quality_score=quality,
        notes=notes
    )


def _find_pairs(root: Path):
    """Find mask/baked/final file triplets"""
    masks = sorted(root.glob("*_mask.png"))
    for m in masks:
        base = m.name.replace("_mask.png", "")
        baked = root / f"{base}_baked.png"
        final = root / f"{base}_final.png"
        yield m, (baked if baked.exists() else None), (final if final.exists() else None)


def _generate_visualization(results: list[HoleStats], out_dir: Path):
    """Generate visual report"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Texture Quality Report', fontsize=16, fontweight='bold')

    # 1. Hole ratio by surface
    ax = axes[0, 0]
    
    x_pos = np.arange(len(results))
    colors = ['#3498db' if r.surface_type == "wall" else '#e74c3c' for r in results]
    bars = ax.bar(x_pos, [r.hole_ratio_pct for r in results], color=colors, alpha=0.7)
    ax.set_xlabel('Surface')
    ax.set_ylabel('Hole Ratio (%)')
    ax.set_title('Hole Coverage by Surface')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.name for r in results], rotation=45, ha='right', fontsize=8)
    ax.axhline(y=5, color='g', linestyle='--', alpha=0.5, label='Good (<5%)')
    ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Medium (5-20%)')
    ax.legend()

    # 2. Quality scores
    ax = axes[0, 1]
    scores = [r.quality_score for r in results]
    bars = ax.barh(x_pos, scores, color=colors, alpha=0.7)
    ax.set_xlabel('Quality Score (0-100)')
    ax.set_ylabel('Surface')
    ax.set_title('Overall Quality Scores')
    ax.set_yticks(x_pos)
    ax.set_yticklabels([r.name for r in results], fontsize=8)
    ax.axvline(x=80, color='g', linestyle='--', alpha=0.5)
    ax.axvline(x=60, color='orange', linestyle='--', alpha=0.5)

    # 3. Component analysis
    ax = axes[1, 0]
    ax.scatter([r.components for r in results],
               [r.hole_ratio_pct for r in results],
               c=['#3498db' if r.surface_type == "wall" else '#e74c3c' for r in results],
               s=100, alpha=0.6)
    ax.set_xlabel('Number of Hole Components')
    ax.set_ylabel('Hole Ratio (%)')
    ax.set_title('Hole Fragmentation Analysis')
    for r in results:
        if r.hole_ratio_pct > 20 or r.components > 50:
            ax.annotate(r.name, (r.components, r.hole_ratio_pct),
                        fontsize=7, alpha=0.7)

    # 4. Texture complexity
    ax = axes[1, 1]
    edge_dens = [r.edge_density_pct if r.edge_density_pct is not None else 0 for r in results]
    line_str = [r.dominant_line_strength if r.dominant_line_strength is not None else 0 for r in results]
    
    ax.scatter(edge_dens, line_str,
               c=['#3498db' if r.surface_type == "wall" else '#e74c3c' for r in results],
               s=100, alpha=0.6)
    ax.set_xlabel('Edge Density (%)')
    ax.set_ylabel('Line Strength (0-1)')
    ax.set_title('Texture Structure Complexity')
    ax.axvline(x=6, color='orange', linestyle='--', alpha=0.5, label='High edge density')
    ax.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Strong lines')
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "quality_report.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Visual report saved: {out_dir / 'quality_report.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None, help="output dir containing *_mask.png")
    ap.add_argument("--out", type=str, default=None, help="output dir for reports")
    ap.add_argument("--json_name", type=str, default="texture_hole_report.json")
    ap.add_argument("--csv_name", type=str, default="texture_hole_report.csv")
    args = ap.parse_args()

    # [2] 경로 자동 설정 로직 추가
    if args.dir:
        root = Path(args.dir)
    else:
        # 인자가 없으면 프로젝트 내 기본 경로 탐색 (room 기준)
        base_dir = Path(__file__).resolve().parent
        root = base_dir / "output" / "room_wall_textures_out"
        print(f"⚠️ 인자가 없어 기본 경로를 사용합니다: {root}")

    if not root.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {root}")
        print("💡 팁: 'python texture_hole_report.py --dir <경로>' 형태로 실행하거나,")
        print("   먼저 wall_texture_restoration.py를 실행해서 결과물을 만드세요.")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else root
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for mask, baked, final in _find_pairs(root):
        try:
            results.append(analyze_one(mask, baked, final))
        except Exception as e:
            print(f"[WARN] failed {mask.name}: {e}")

    # Sort: floor first, then walls by hole ratio
    results.sort(key=lambda r: (r.surface_type == "wall", r.hole_ratio_pct), reverse=False)

    # Console summary
    print("\n" + "="*70)
    print(f"🎨 Texture Quality Report: {root}")
    print(f"📊 Total surfaces: {len(results)}")
    walls = [r for r in results if r.surface_type == "wall"]
    floors = [r for r in results if r.surface_type == "floor"]
    print(f"   - Walls: {len(walls)}")
    print(f"   - Floors: {len(floors)}")
    print("="*70)

    if floors:
        print("\n🏠 FLOOR:")
        for r in floors:
            print(f"  {r.name}")
            print(f"    - Hole: {r.hole_ratio_pct:.1f}%  |  Quality: {r.quality_score:.1f}/100  |  Confidence: {r.confidence}")
            if r.notes:
                print(f"    - Note: {r.notes}")
            print(f"    - {r.recommendation}")

    if walls:
        print("\n🧱 WALLS:")
        for r in walls:
            print(f"  {r.name}")
            print(f"    - Hole: {r.hole_ratio_pct:.1f}%  |  Quality: {r.quality_score:.1f}/100  |  Confidence: {r.confidence}")
            if r.notes:
                print(f"    - Note: {r.notes}")
            print(f"    - {r.recommendation}")

    print("="*70)

    # Summary stats
    avg_wall_hole = np.mean([r.hole_ratio_pct for r in walls]) if walls else 0
    avg_floor_hole = np.mean([r.hole_ratio_pct for r in floors]) if floors else 0
    avg_quality = np.mean([r.quality_score for r in results]) if results else 0

    print(f"\n📈 STATISTICS:")
    print(f"   Average wall hole ratio: {avg_wall_hole:.1f}%")
    print(f"   Average floor hole ratio: {avg_floor_hole:.1f}%")
    print(f"   Overall quality score: {avg_quality:.1f}/100")

    # Save JSON
    payload = [asdict(r) for r in results]
    (out_dir / args.json_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 JSON report: {out_dir / args.json_name}")

    # Save CSV
    if pd is not None:
        df = pd.DataFrame(payload)
        df.to_csv(out_dir / args.csv_name, index=False, encoding="utf-8-sig")
        print(f"💾 CSV report: {out_dir / args.csv_name}")

    # Generate visualization
    try:
        _generate_visualization(results, out_dir)
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")

    print("="*70)
    print("✅ Report generation complete!")


if __name__ == "__main__":
    main()