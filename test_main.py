"""main.py의 핵심 로직 점검용 테스트.

표준 라이브러리 unittest만 사용한다.
실행: python3 -m unittest -v
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
