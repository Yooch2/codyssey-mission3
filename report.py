"""모드 2의 판정과 성능 표를 JSON/Markdown 파일로 저장한다.

실행: python3 report.py --json result.json --markdown result.md
기존 README와 data.json은 수정하지 않는다.
"""

import argparse
import contextlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import main


def markdown_report(result):
    """실제 실행 결과로 판정·실패·성능·복잡도 리포트를 만든다."""
    def cell(value):
        # 잘못된 JSON 키/라벨도 표의 열이나 HTML을 깨뜨리지 않게 한다.
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(
            ">", "&gt;").replace("|", "&#124;").replace("\n", " ").replace("\r", " ")

    summary = result["summary"]
    lines = ["# 실행 결과", "", "- 측정 시각(UTC): " + result["measured_at"],
             "- 환경: " + result["environment"],
             "- 데이터: " + cell(result["data_file"]),
             "- EPSILON: %s" % result["epsilon"],
             "- 총 %d개 / 통과 %d개 / 실패 %d개" % (
                 summary["total"], summary["passed"], summary["failed"]),
             "- 실패 분류: 동점 %d / 라벨 불일치 %d / 스키마 %d" % (
                 summary["undecided"], summary["mismatch"], summary["schema_errors"]),
             "", "## 패턴별 판정", "",
             "| 케이스 | 판정 | expected | 차이 | 결과·사유 |",
             "|---|---|---|---|---|"]
    for case in result["cases"]:
        difference = ("%.3e" % case["difference"]) if "difference" in case else "—"
        lines.append("| %s | %s | %s | %s | %s |" % (
            cell(case["key"]), cell(case["verdict"] or "—"),
            cell(case["expected"] or "—"), difference,
            "PASS" if case["passed"] else "FAIL: " + cell(case["reason"])))
    lines += ["", "## 성능", "", "| 크기 | 평균 시간(ms) | 연산 횟수 |",
              "|---|---|---|"]
    for row in result["performance"]:
        lines.append("| %dx%d | %.4f | %d |" % (
            row["size"], row["size"], row["average_ms"], row["mac_count"]))
    lines += ["", "- 워밍업 1회 후 10회 평균. 입출력과 파일 읽기는 측정 밖이다.",
              "- MAC은 N×N칸을 순회하므로 시간 O(N²), 추가 공간 O(1)이다.",
              "- 동점은 expected와 일치하지 않아 FAIL로 집계한다.",
              "- 수치는 이번 실행 기록이다. 해석과 과거 측정은 README를 참고한다."]
    return "\n".join(lines) + "\n"


def main_entry(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="JSON 출력 파일")
    parser.add_argument("--markdown", type=Path, help="Markdown 출력 파일")
    args = parser.parse_args(argv)
    if not args.json and not args.markdown:
        parser.error("--json 또는 --markdown 출력 경로를 지정하세요.")
    targets = [path.resolve() for path in (args.json, args.markdown) if path]
    source_dir = Path(main.BASE_DIR)
    protected = {path.resolve() for path in source_dir.rglob("*.py")}
    protected.add(Path(main.DATA_FILE).resolve())
    protected.update((source_dir / name).resolve()
                     for name in ("README.md", "data.json", ".gitignore"))
    if len(set(targets)) != len(targets) or any(path in protected for path in targets):
        parser.error("출력에는 서로 다른 새 파일명을 사용하세요. 저장소 파일은 덮어쓰지 않습니다.")
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        result = main.run_json_mode()
    if result is None:
        parser.exit(1, captured.getvalue())
    result["measured_at"] = datetime.now(timezone.utc).isoformat()
    result["environment"] = "%s %s / Python %s" % (
        platform.system(), platform.machine(), platform.python_version())
    # 생성부터 완료한 뒤 저장한다. 실패하면 원본 README/데이터는 그대로 남는다.
    outputs = []
    if args.json:
        outputs.append((args.json, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"))
    if args.markdown:
        outputs.append((args.markdown, markdown_report(result)))
    for path, content in outputs:
        path.write_text(content, encoding="utf-8")
        print("저장: %s" % path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main_entry())
    except (OSError, ValueError, MemoryError) as error:
        raise SystemExit("리포트를 저장하지 못했습니다: %s" % error)
    except KeyboardInterrupt:
        raise SystemExit("리포트 생성을 중단했습니다.")
