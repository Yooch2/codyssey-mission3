"""파일 리포트와 반복 측정 통계를 검증한다."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import benchmark
import main
import report


class TestReport(unittest.TestCase):
    def test_같은_실행의_JSON과_Markdown을_저장한다(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            args = ["--json", str(target / "result.json"),
                    "--markdown", str(target / "result.md")]
            with contextlib.redirect_stdout(io.StringIO()):
                report.main_entry(args)
            result = json.loads((target / "result.json").read_text())
            self.assertEqual(result["summary"], {
                "total": 6, "passed": 3, "failed": 3, "undecided": 3,
                "mismatch": 0, "schema_errors": 0})
            text = (target / "result.md").read_text()
            self.assertIn("총 6개 / 통과 3개 / 실패 3개", text)
            self.assertIn("1.110e-16", text)
            self.assertIn("평균 시간(ms)", text)
            self.assertGreaterEqual(len(text.splitlines()), 10)
            # 실행 결과 파일은 같은 명령으로 다시 만들 수 있어야 한다.
            with contextlib.redirect_stdout(io.StringIO()):
                report.main_entry(args)

    def test_원본파일을_덮어쓰지_않는다(self):
        before = Path(main.DATA_FILE).read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            report.main_entry(["--json", main.DATA_FILE])
        self.assertEqual(Path(main.DATA_FILE).read_bytes(), before)

    def test_데이터를_못읽으면_성공리포트를_쓰지_않는다(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.json"
            with mock.patch("main.DATA_FILE", str(Path(folder) / "missing.json")):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    report.main_entry(["--json", str(path)])
            self.assertEqual(caught.exception.code, 1)
            self.assertFalse(path.exists())

    def test_동일한_두_출력경로는_거부한다(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "result.json")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                report.main_entry(["--json", path, "--markdown", path])
            self.assertFalse(Path(path).exists())

    def test_스키마오류와_라벨불일치도_별도로_저장한다(self):
        data = {"filters": {"size_3": {"cross": main.make_cross(3), "x": main.make_x(3)}},
                "patterns": {"size_3_1": {"input": main.make_cross(3), "expected": "x"},
                             "size_3_2": {"input": "invalid", "expected": "x"}}}
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "data.json", Path(folder) / "result.json"
            source.write_text(json.dumps(data))
            with mock.patch("main.DATA_FILE", str(source)):
                with contextlib.redirect_stdout(io.StringIO()):
                    report.main_entry(["--json", str(target)])
            result = json.loads(target.read_text())
        self.assertEqual(result["summary"]["mismatch"], 1)
        self.assertEqual(result["summary"]["schema_errors"], 1)
        self.assertEqual(result["cases"][0]["reason"], "라벨 불일치")
        self.assertIsNone(result["cases"][1]["verdict"])

    def test_Markdown은_외부데이터의_열구분과_HTML을_이스케이프한다(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = main.run_json_mode()
        result.update(measured_at="2026-11-09", environment="test")
        result["cases"][0]["key"] = "<b>|bad\nkey"
        text = report.markdown_report(result)
        self.assertIn("&lt;b&gt;&#124;bad key", text)

    def test_표준편차는_10회평균_표본_사이에서_구한다(self):
        with mock.patch("main.PERF_SIZES", [3]), mock.patch(
                "main.measure_ms", side_effect=[1.0, 2.0, 3.0, 4.0, 5.0]):
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                benchmark.bench_variation()
        self.assertIn("중앙값(ms)", captured.getvalue())
        self.assertIn("3.000000", captured.getvalue())
        self.assertIn("1.581139", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
