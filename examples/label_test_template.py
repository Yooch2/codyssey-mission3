"""라벨 확장 시 test_main.py에 옮겨 쓸 테스트 예시.

ALIASES와 GENERATORS에 새 라벨을 추가하고, 새 판정 API에 맞춰
TestDecide의 최대값·동점 테스트도 함께 수정한다.
현재 두 라벨 예시는 저장소 루트에서 아래처럼 실행할 수 있다.
python3 -m unittest discover -s examples -p '*_template.py' -v
"""

import unittest

import main

ALIASES = {
    main.CROSS: ("+", "cross", "plus"),
    main.X_MARK: ("x",),
    # O 구현 후: main.O_MARK: ("o", "circle"),
}
GENERATORS = {
    main.CROSS: main.make_cross,
    main.X_MARK: main.make_x,
    # O 구현 후: main.O_MARK: main.make_circle,
}


class LabelExtensionTemplate(unittest.TestCase):
    def test_aliases(self):
        for expected, aliases in ALIASES.items():
            for alias in aliases:
                with self.subTest(alias=alias):
                    self.assertEqual(main.normalize_label(" " + alias.upper() + " "), expected)

    def test_generated_patterns(self):
        for size in (3, 5, 13, 25):
            filters = {label: build(size) for label, build in GENERATORS.items()}
            for expected, pattern in filters.items():
                with self.subTest(size=size, label=expected):
                    own = main.mac(pattern, filters[expected])
                    for other, matrix in filters.items():
                        if other != expected:
                            self.assertGreater(own - main.mac(pattern, matrix), main.EPSILON)


if __name__ == "__main__":
    unittest.main()
