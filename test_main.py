"""main.py의 핵심 로직 점검용 테스트.

표준 라이브러리 unittest만 사용한다.
실행: python3 -m unittest -v
"""

import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from unittest import mock

import main


class TestMac(unittest.TestCase):
    """MAC 연산 자체를 확인한다."""

    def test_위치별로_곱해서_더한다(self):
        pattern = [[1, 2], [3, 4]]
        matrix_filter = [[1, 0], [0, 1]]
        # 1*1 + 2*0 + 3*0 + 4*1 = 5
        self.assertEqual(main.mac(pattern, matrix_filter), 5.0)

    def test_결과는_실수형이다(self):
        self.assertIsInstance(main.mac([[1]], [[1]]), float)

    def test_소수도_계산한다(self):
        # 0.5*2 + 0.5*4 + 0.5*0 + 0.5*0 = 3.0
        self.assertAlmostEqual(main.mac([[0.5, 0.5], [0.5, 0.5]], [[2, 4], [0, 0]]), 3.0)

    def test_1차원_버전도_같은_값을_낸다(self):
        pattern = main.make_x(5)
        matrix_filter = main.make_cross(5)
        two_dim = main.mac(pattern, matrix_filter)
        one_dim = main.mac_flat(main.flatten(pattern), main.flatten(matrix_filter))
        self.assertEqual(two_dim, one_dim)

    def test_flatten은_길이가_N제곱이다(self):
        self.assertEqual(len(main.flatten(main.make_x(13))), 169)


class TestNormalizeLabel(unittest.TestCase):
    """라벨 정규화를 확인한다."""

    def test_더하기표는_Cross가_된다(self):
        self.assertEqual(main.normalize_label("+"), main.CROSS)

    def test_cross문자열도_Cross가_된다(self):
        self.assertEqual(main.normalize_label("cross"), main.CROSS)

    def test_대문자와_공백을_무시한다(self):
        self.assertEqual(main.normalize_label(" Cross "), main.CROSS)
        self.assertEqual(main.normalize_label("X"), main.X_MARK)

    def test_소문자x는_X가_된다(self):
        self.assertEqual(main.normalize_label("x"), main.X_MARK)

    def test_모르는_표기는_None이다(self):
        self.assertIsNone(main.normalize_label("별표"))
        self.assertIsNone(main.normalize_label(None))
        self.assertIsNone(main.normalize_label(3))


class TestDecide(unittest.TestCase):
    """허용오차 기반 판정을 확인한다."""

    def test_큰_쪽을_고른다(self):
        self.assertEqual(main.decide(5.0, 1.0), main.CROSS)
        self.assertEqual(main.decide(1.0, 5.0), main.X_MARK)

    def test_완전히_같으면_동점이다(self):
        self.assertEqual(main.decide(2.0, 2.0), main.UNDECIDED)

    def test_부동소수점_오차는_동점으로_본다(self):
        # 0.3을 세 번 더하면 0.8999999999999999가 되어 0.9와 미세하게 다르다
        self.assertNotEqual(0.3 + 0.3 + 0.3, 0.9)
        self.assertEqual(main.decide(0.9, 0.3 + 0.3 + 0.3), main.UNDECIDED)

    def test_허용오차보다_큰_차이는_구분한다(self):
        self.assertEqual(main.decide(1.0, 1.0 + 1e-6), main.X_MARK)

    def test_허용오차_경계에서_갈린다(self):
        # EPSILON보다 작으면 동점, 크거나 같으면 구분한다.
        half = main.EPSILON / 2
        self.assertEqual(main.decide(1.0, 1.0 + half), main.UNDECIDED)
        self.assertEqual(main.decide(1.0, 1.0 + main.EPSILON * 2), main.X_MARK)
        self.assertEqual(main.decide(1.0 + main.EPSILON * 2, 1.0), main.CROSS)


class TestGenerator(unittest.TestCase):
    """패턴 생성기를 확인한다."""

    def test_십자가는_가운데_행과_열이_1이다(self):
        pattern = main.make_cross(5)
        self.assertEqual(pattern[2], [1.0] * 5)
        self.assertEqual([row[2] for row in pattern], [1.0] * 5)
        self.assertEqual(pattern[0][0], 0.0)

    def test_X자는_두_대각선이_1이다(self):
        pattern = main.make_x(5)
        self.assertEqual([pattern[i][i] for i in range(5)], [1.0] * 5)
        self.assertEqual([pattern[i][4 - i] for i in range(5)], [1.0] * 5)
        self.assertEqual(pattern[0][1], 0.0)

    def test_생성한_패턴은_자기_필터에서_점수가_더_높다(self):
        for size in (3, 5, 13, 25):
            cross = main.make_cross(size)
            x_mark = main.make_x(size)
            self.assertEqual(main.decide(main.mac(cross, cross), main.mac(cross, x_mark)),
                             main.CROSS)
            self.assertEqual(main.decide(main.mac(x_mark, cross), main.mac(x_mark, x_mark)),
                             main.X_MARK)


class TestSchema(unittest.TestCase):
    """스키마 검증과 키 해석을 확인한다."""

    def test_크기를_키에서_뽑는다(self):
        self.assertEqual(main.parse_size("size_13"), 13)
        self.assertEqual(main.parse_size("size_25_4"), 25)

    def test_이상한_키는_None이다(self):
        self.assertIsNone(main.parse_size("filter_5"))
        self.assertIsNone(main.parse_size("size_abc"))
        self.assertIsNone(main.parse_size("size"))

    def test_0_이하_크기는_거부한다(self):
        # size_0을 통과시키면 빈 배열이 0x0 정사각 행렬로 인정되어
        # 점수 0.0 대 0.0의 가짜 동점 판정이 나온다.
        self.assertIsNone(main.parse_size("size_0"))
        self.assertIsNone(main.parse_size("size_-3"))
        self.assertIsNone(main.parse_size("size_0_1"))
        self.assertTrue(main.is_square([], 0))  # 0을 막지 않으면 통과한다는 근거

    def test_필터가_없으면_FAIL_사유를_돌려준다(self):
        parsed, reason = main.check_case("size_7_1",
                                         {"input": main.make_x(7), "expected": "x"}, {})
        self.assertIsNone(parsed)
        self.assertIn("필터가 data.json에 없음", reason)

    def test_필터에_한쪽_라벨만_있으면_FAIL이다(self):
        filters = {3: {main.CROSS: main.make_cross(3)}}
        parsed, reason = main.check_case("size_3_1",
                                         {"input": main.make_x(3), "expected": "x"}, filters)
        self.assertIsNone(parsed)
        self.assertIn("Cross/X 가 모두 있지 않음", reason)

    def test_패턴_항목이_객체가_아니면_FAIL이다(self):
        parsed, reason = main.check_case("size_3_1", "객체 아님", {})
        self.assertIsNone(parsed)
        self.assertIn("객체 형식이 아님", reason)

    def test_정렬은_크기와_번호_순이다(self):
        keys = ["size_13_2", "size_5_10", "size_5_2", "size_25_1"]
        self.assertEqual(sorted(keys, key=main.sort_key),
                         ["size_5_2", "size_5_10", "size_13_2", "size_25_1"])

    def test_정사각_배열만_통과한다(self):
        self.assertTrue(main.is_square([[1, 2], [3, 4]], 2))
        self.assertFalse(main.is_square([[1, 2], [3]], 2))
        self.assertFalse(main.is_square([[1, 2]], 2))
        self.assertFalse(main.is_square("배열 아님", 2))
        self.assertFalse(main.is_square([[True, 1], [1, 1]], 2))

    def test_크기가_안_맞으면_FAIL_사유를_돌려준다(self):
        filters = {3: {main.CROSS: main.make_cross(3), main.X_MARK: main.make_x(3)}}
        case = {"input": main.make_x(3), "expected": "x"}
        parsed, reason = main.check_case("size_5_1", case, filters)
        self.assertIsNone(parsed)
        self.assertIn("5x5", reason)

    def test_모르는_expected는_FAIL_사유를_돌려준다(self):
        filters = {3: {main.CROSS: main.make_cross(3), main.X_MARK: main.make_x(3)}}
        case = {"input": main.make_x(3), "expected": "세모"}
        parsed, reason = main.check_case("size_3_1", case, filters)
        self.assertIsNone(parsed)
        self.assertIn("expected", reason)

    def test_정상_케이스는_정규화된_값을_돌려준다(self):
        filters = {3: {main.CROSS: main.make_cross(3), main.X_MARK: main.make_x(3)}}
        case = {"input": main.make_cross(3), "expected": "+"}
        parsed, reason = main.check_case("size_3_1", case, filters)
        self.assertIsNone(reason)
        self.assertEqual(parsed[1], main.CROSS)


class TestDataFile(unittest.TestCase):
    """함께 제출하는 data.json이 스키마를 지키는지 확인한다."""

    def test_모든_패턴이_판정_가능하다(self):
        with open(main.DATA_FILE, "r", encoding="utf-8") as data_file:
            data = json.load(data_file)
        filters = {}
        for key, value in data["filters"].items():
            size = main.parse_size(key)
            filters[size] = {main.normalize_label(label): matrix
                             for label, matrix in value.items()}
        for key, case in data["patterns"].items():
            parsed, reason = main.check_case(key, case, filters)
            self.assertIsNone(reason, "%s: %s" % (key, reason))

    def test_모드2_전체_결과가_일치한다(self):
        """미션 재현성 요구: 각 케이스의 PASS/FAIL과 총합이 일치해야 한다.

        load_filters부터 결과 요약까지 실제 경로를 그대로 돌려서 확인한다.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main.run_json_mode()
        output = buffer.getvalue()

        expected_lines = [
            "--- size_5_1 ---",
            "판정: UNDECIDED | expected: X | FAIL (동점 규칙)",
            "--- size_5_2 ---",
            "판정: Cross | expected: Cross | PASS",
            "--- size_13_1 ---",
            "판정: X | expected: X | PASS",
            "--- size_13_2 ---",
            "판정: UNDECIDED | expected: Cross | FAIL (동점 규칙)",
            "--- size_25_1 ---",
            "판정: UNDECIDED | expected: X | FAIL (동점 규칙)",
            "--- size_25_2 ---",
            "판정: Cross | expected: Cross | PASS",
            "총 테스트: 6개",
            "통과: 3개",
            "실패: 3개",
            "  - 동점(UNDECIDED): 3개",
            "  - 라벨 불일치: 0개",
            "  - 데이터/스키마 오류: 0개",
        ]
        for line in expected_lines:
            self.assertIn(line, output)

    def test_모드2_점수가_일치한다(self):
        """동점 케이스의 점수가 바뀌면 분석 내용이 통째로 흔들리므로 고정한다."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main.run_json_mode()
        output = buffer.getvalue()
        for line in ["Cross 점수: 0.9",
                     "X 점수: 0.8999999999999999",
                     "Cross 점수: 7.499999999999997",
                     "X 점수: 7.5",
                     "Cross 점수: 4.9",
                     "X 점수: 4.899999999999999"]:
            self.assertIn(line, output)

    def test_통과와_실패의_합이_전체와_같다(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main.run_json_mode()
        numbers = {}
        for line in buffer.getvalue().splitlines():
            for label in ("총 테스트", "통과", "실패"):
                if line.startswith(label + ":"):
                    numbers[label] = int(line.split(":")[1].strip().rstrip("개"))
        self.assertEqual(numbers["통과"] + numbers["실패"], numbers["총 테스트"])

    def test_읽은_파일_경로를_알려준다(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main.run_json_mode()
        self.assertIn(main.DATA_FILE, buffer.getvalue())

    def test_없는_파일을_읽어도_죽지_않는다(self):
        original = main.DATA_FILE
        with tempfile.TemporaryDirectory() as folder:
            main.DATA_FILE = os.path.join(folder, "없는파일.json")
            try:
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    main.run_json_mode()  # 예외 없이 안내만 출력하면 통과
                self.assertIn("찾을 수 없습니다", captured.getvalue())
            finally:
                main.DATA_FILE = original


class TestReadRow(unittest.TestCase):
    """모드 1의 한 줄 입력 처리를 확인한다."""

    def read(self, text, size=3):
        """입력 한 줄을 흉내 내고 (결과, 출력된 안내) 를 돌려준다."""
        buffer = io.StringIO()
        with mock.patch("builtins.input", return_value=text):
            with contextlib.redirect_stdout(buffer):
                result = main.read_row(size)
        return result, buffer.getvalue()

    def test_공백으로_구분된_세_값을_읽는다(self):
        self.assertEqual(self.read("0 1 0")[0], [0.0, 1.0, 0.0])

    def test_음수와_지수표기도_읽는다(self):
        self.assertEqual(self.read("-1 2e-3 3")[0], [-1.0, 0.002, 3.0])

    def test_붙여쓴_숫자도_읽는다(self):
        self.assertEqual(self.read("010")[0], [0.0, 1.0, 0.0])

    def test_값이_모자라면_개수를_알려준다(self):
        # '0.5'는 길이가 3이지만 숫자 하나다. 글자로 쪼개면 '.'을 숫자가
        # 아니라고 지적하게 되어 개수가 모자란다는 사실이 가려진다.
        result, message = self.read("0.5")
        self.assertIsNone(result)
        self.assertIn("3개의 숫자", message)

    def test_음수_하나만_넣어도_개수를_알려준다(self):
        result, message = self.read("-11")
        self.assertIsNone(result)
        self.assertIn("3개의 숫자", message)

    def test_숫자가_아니면_거부한다(self):
        result, message = self.read("x y z")
        self.assertIsNone(result)
        self.assertIn("숫자가 아닙니다", message)

    def test_nan과_inf는_거부한다(self):
        for text in ("nan 0 1", "inf 0 1", "-inf 0 1", "1e400 0 1"):
            result, message = self.read(text)
            self.assertIsNone(result, text)
            self.assertIn("유한한 숫자", message)

    def test_입력이_끝나면_EOFError가_올라온다(self):
        with mock.patch("builtins.input", side_effect=EOFError):
            with self.assertRaises(EOFError):
                main.read_row(3)


class TestFormatNumber(unittest.TestCase):
    """행렬 확인 출력에 쓰는 숫자 표기를 확인한다."""

    def test_정수면_소수점을_생략한다(self):
        self.assertEqual(main.format_number(1.0), "1")
        self.assertEqual(main.format_number(-3.0), "-3")

    def test_소수는_그대로_보여준다(self):
        self.assertEqual(main.format_number(0.5), "0.5")

    def test_무한과_NaN에서_죽지_않는다(self):
        # int(inf)와 int(nan)은 예외를 낸다. 그대로 두면 출력 단계에서
        # 추적 메시지를 내며 프로그램이 멈춘다.
        self.assertEqual(main.format_number(float("inf")), "inf")
        self.assertEqual(main.format_number(float("nan")), "nan")

    def test_행렬_전체를_찍어도_죽지_않는다(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main.print_matrix([[float("nan"), 1.0], [0.0, float("inf")]])
        self.assertIn("nan", buffer.getvalue())


class TestTable(unittest.TestCase):
    """표 출력의 열 정렬을 확인한다."""

    def test_한글은_두_칸으로_센다(self):
        self.assertEqual(main.display_width("크기"), 4)
        self.assertEqual(main.display_width("3x3"), 3)
        # 평균(4) + 공백(1) + 시간(4) + "(ms)"(4) = 13
        self.assertEqual(main.display_width("평균 시간(ms)"), 13)

    def test_값이_길어져도_열이_맞는다(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main.print_table(["크기", "값"],
                             [["1000x1000", "12345.6789"], ["3x3", "0.0003"]])
        lines = buffer.getvalue().splitlines()
        # 머리글과 각 행에서 두 번째 열이 시작하는 칸 위치가 모두 같아야 한다.
        # 한글 머리글이 두 칸을 차지하므로 글자 수가 아니라 표시 폭으로 잰다.
        starts = []
        for line, second in ((lines[0], "값"),
                             (lines[2], "12345.6789"),
                             (lines[3], "0.0003")):
            starts.append(main.display_width(line[:line.rindex(second)]))
        self.assertEqual(len(set(starts)), 1, (starts, lines))


class TestLoadFilters(unittest.TestCase):
    """필터 로드와 경고를 확인한다."""

    def load(self, raw):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = main.load_filters(raw)
        return result, buffer.getvalue()

    def test_라벨을_표준_라벨로_바꿔_담는다(self):
        filters, _ = self.load({"size_3": {"cross": main.make_cross(3),
                                           "x": main.make_x(3)}})
        self.assertEqual(sorted(filters[3].keys()), [main.CROSS, main.X_MARK])

    def test_크기가_안_맞는_필터는_경고하고_뺀다(self):
        filters, message = self.load({"size_5": {"cross": main.make_cross(3),
                                                 "x": main.make_x(5)}})
        self.assertNotIn(main.CROSS, filters[5])
        self.assertIn(main.X_MARK, filters[5])
        self.assertIn("5x5 숫자 배열이 아니라", message)

    def test_같은_크기가_두_번_나오면_경고한다(self):
        # 'size_3_백업'도 크기 3으로 해석되어 앞의 필터를 덮어쓴다.
        # 경고가 없으면 어느 쪽이 쓰였는지 알 수 없다.
        _, message = self.load({
            "size_3": {"cross": main.make_cross(3), "x": main.make_x(3)},
            "size_3_백업": {"cross": main.make_cross(3), "x": main.make_x(3)},
        })
        self.assertIn("이미 있습니다", message)

    def test_필터가_객체가_아니면_건너뛴다(self):
        filters, message = self.load({"size_3": "행렬 아님"})
        self.assertEqual(filters, {})
        self.assertIn("형식을 알 수 없어", message)


class TestMeasure(unittest.TestCase):
    """시간 측정 함수의 호출 횟수를 확인한다."""

    def test_워밍업은_측정에_포함되지_않는다(self):
        calls = []
        main.measure_ms(lambda a, b: calls.append(1), None, None,
                        repeat=4, warmup=3)
        self.assertEqual(len(calls), 7)  # 워밍업 3 + 측정 4

    def test_반환값은_양수인_ms이다(self):
        value = main.measure_ms(main.mac, main.make_x(5), main.make_cross(5))
        self.assertGreater(value, 0)
        self.assertTrue(math.isfinite(value))


class TestInteractiveFlow(unittest.TestCase):
    """입력 루프와 메뉴 흐름을 확인한다."""

    def run_with(self, inputs, function):
        """입력을 차례로 흘려보내고 (반환값, 출력) 을 돌려준다."""
        buffer = io.StringIO()
        with mock.patch("builtins.input", side_effect=inputs):
            with contextlib.redirect_stdout(buffer):
                result = function()
        return result, buffer.getvalue()

    def test_틀린_줄만_다시_받는다(self):
        # 1번째 줄이 두 개짜리라 거부되고, 다시 받은 줄부터 채워진다.
        matrix, message = self.run_with(
            ["0 1", "0 1 0", "1 1 1", "0 1 0"],
            lambda: main.read_matrix("필터 A", 3))
        self.assertEqual(matrix, [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        self.assertIn("1번째 줄을 다시 입력하세요", message)

    def test_중간_줄이_틀려도_앞은_유지된다(self):
        matrix, message = self.run_with(
            ["0 1 0", "x y z", "1 1 1", "0 1 0"],
            lambda: main.read_matrix("필터 A", 3))
        self.assertEqual(matrix[0], [0.0, 1.0, 0.0])
        self.assertIn("2번째 줄을 다시 입력하세요", message)

    def test_입력이_끊기면_모드1이_안내하고_끝난다(self):
        # 이 검사가 없으면 EOF를 형식 오류와 똑같이 처리해
        # '다시 입력하세요'가 끝없이 찍히는 무한 반복으로 되돌아가도 아무도 모른다.
        # side_effect를 길이 1짜리 목록으로 둔다. 첫 호출만 EOFError를 내고
        # 그 다음 호출은 StopIteration이 되므로, 버그가 되살아나도 테스트가
        # 멈추지 않고 즉시 실패한다.
        _, message = self.run_with([EOFError], main.run_user_mode)
        self.assertIn("입력이 끝나 모드 1을 중단합니다", message)

    def test_모드1은_예시대로_넣으면_B로_판정한다(self):
        answers = ["1",                           # 필터 채우기: 직접 입력
                   "0 1 0", "1 1 1", "0 1 0",     # 필터 A (십자가)
                   "1 0 1", "0 1 0", "1 0 1",     # 필터 B (X자)
                   "1 0 1", "0 1 0", "1 0 1"]     # 패턴 (X자)
        _, message = self.run_with(answers, main.run_user_mode)
        self.assertIn("A 점수: 1.0", message)
        self.assertIn("B 점수: 5.0", message)
        self.assertIn("판정: B", message)

    def test_모드1은_동점이면_판정_불가를_알린다(self):
        answers = ["1",                           # 필터 채우기: 직접 입력
                   "1 0 0", "0 0 0", "0 0 0",     # 필터 A
                   "0 0 0", "0 0 0", "0 0 1",     # 필터 B
                   "1 0 0", "0 0 0", "0 0 1"]     # 패턴 (양쪽 모두 1.0)
        _, message = self.run_with(answers, main.run_user_mode)
        self.assertIn("판정: 판정 불가", message)

    def test_모드1에서_생성기로_필터를_채울_수_있다(self):
        # 패턴 생성기(보너스)가 성능 분석뿐 아니라 모드 1 입력으로도
        # 쓰이는지 확인한다. 이 검사가 없으면 생성기와 모드 1의 연결이
        # 끊어져도 아무도 모른다.
        answers = ["2",                           # 필터 채우기: 생성기
                   "1 0 1", "0 1 0", "1 0 1"]     # 패턴 (X자)
        _, message = self.run_with(answers, main.run_user_mode)
        self.assertIn("패턴 생성기로 3x3 필터를 만들었습니다", message)
        self.assertIn("A 점수: 1.0", message)     # 십자가 필터와 한 칸 겹침
        self.assertIn("B 점수: 5.0", message)     # X자 필터와 다섯 칸 겹침
        self.assertIn("판정: B", message)

    def test_필터_채우기_선택은_잘못된_번호를_다시_묻는다(self):
        answers = ["9", "2", "1 0 1", "0 1 0", "1 0 1"]
        _, message = self.run_with(answers, main.run_user_mode)
        self.assertIn("1 또는 2 중에서 선택하세요", message)

    def test_메뉴는_잘못된_번호를_다시_묻는다(self):
        choice, message = self.run_with(["9", "엔터", "2"], main.choose_mode)
        self.assertEqual(choice, "2")
        self.assertIn("1, 2, 3, 4 중에서 선택하세요", message)

    def test_메뉴에서_4나_입력끝이면_종료한다(self):
        self.assertIsNone(self.run_with(["4"], main.choose_mode)[0])
        self.assertIsNone(self.run_with([EOFError], main.choose_mode)[0])

    def test_모드3은_크기를_받아_생성하고_판정까지_간다(self):
        # 보너스 요구: "크기 N 입력 시 N x N 십자가와 X 패턴을 자동 생성"하고
        # "생성된 패턴을 모드 1/성능 분석에 재활용"한다. 생성부터 성능 분석까지
        # 한 흐름으로 이어지는지 확인한다.
        _, message = self.run_with(["5"], main.run_generator_mode)
        self.assertIn("생성 결과 (5x5)", message)
        self.assertIn("Cross 패턴 | Cross 점수 9.0 | X 점수 1.0", message)
        self.assertIn("X 패턴 | Cross 점수 1.0 | X 점수 9.0", message)
        self.assertIn("일치", message)
        self.assertNotIn("불일치", message)
        self.assertIn("연산 횟수", message)        # [3] 성능 분석
        self.assertIn("1차원(ms)", message)        # [4] 최적화 비교

    def test_모드3은_범위_밖_크기를_다시_받는다(self):
        _, message = self.run_with(["abc", "2", "5"], main.run_generator_mode)
        self.assertIn("'abc' 는 정수가 아닙니다", message)
        self.assertIn("3 이상 1000 이하로 입력하세요", message)
        self.assertIn("생성 결과 (5x5)", message)

    def test_모드3은_큰_행렬을_찍지_않고_요약한다(self):
        # 한계보다 큰 크기를 그대로 찍으면 수백 줄이 나와 다른 출력을 밀어낸다.
        size = main.GENERATOR_PRINT_LIMIT + 1
        _, message = self.run_with([str(size)], main.run_generator_mode)
        self.assertIn("라 생략", message)
        self.assertIn("값이 1인 칸", message)

    def test_모드3은_입력이_끊기면_안내하고_끝난다(self):
        _, message = self.run_with([EOFError], main.run_generator_mode)
        self.assertIn("입력이 끝나 모드 3을 중단합니다", message)


class TestPerformanceOutput(unittest.TestCase):
    """성능 표 출력의 형태를 확인한다."""

    def capture(self, function, sizes):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = function(sizes)
        return result, buffer.getvalue()

    def test_크기마다_한_줄씩_연산_횟수와_함께_찍는다(self):
        measured, output = self.capture(main.print_performance, [3, 5])
        self.assertEqual([size for size, _ in measured], [3, 5])
        self.assertTrue(all(time > 0 for _, time in measured))
        self.assertIn("3x3", output)
        self.assertIn("9", output)     # 3의 제곱
        self.assertIn("25", output)    # 5의 제곱
        self.assertIn("연산 횟수", output)

    def test_최적화_비교는_두_방식을_나란히_찍는다(self):
        _, output = self.capture(main.print_optimization, [5])
        self.assertIn("2차원(ms)", output)
        self.assertIn("1차원(ms)", output)
        self.assertRegex(output, r"[+-]\d+\.\d%")

    def test_미션이_요구한_네_크기를_모두_다룬다(self):
        self.assertEqual(main.PERF_SIZES, [3, 5, 13, 25])
        _, output = self.capture(main.print_performance, main.PERF_SIZES)
        for size in main.PERF_SIZES:
            self.assertIn("%dx%d" % (size, size), output)
            self.assertIn(str(size * size), output)


if __name__ == "__main__":
    unittest.main()
