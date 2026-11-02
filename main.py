"""Mini NPU Simulator

MAC(Multiply-Accumulate) 연산으로 입력 패턴이 십자가(Cross)에 가까운지
X자에 가까운지 판별하는 콘솔 애플리케이션이다.
표준 라이브러리(json, os, time)만 사용한다.
"""

import json
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")

# 점수 비교에 쓰는 허용오차. 두 점수의 차이가 이 값보다 작으면 동점으로 본다.
EPSILON = 1e-9

# 성능 측정 반복 횟수(미션 최소 기준 10회)
REPEAT = 10

# 프로그램 내부에서 쓰는 표준 라벨
CROSS = "Cross"
X_MARK = "X"
UNDECIDED = "UNDECIDED"

# 성능 분석에 사용할 크기 목록
PERF_SIZES = [3, 5, 13, 25]

SEPARATOR = "#" + "-" * 38


def print_header(title):
    """출력 구역을 구분하는 머리글을 찍는다."""
    print()
    print(SEPARATOR)
    print("# " + title)
    print(SEPARATOR)


# ---------------------------------------------------------------
# MAC 연산
# ---------------------------------------------------------------

def mac(pattern, matrix_filter):
    """2차원 패턴과 2차원 필터를 위치별로 곱해서 모두 더한다.

    외부 라이브러리 없이 반복문으로 직접 구현한다.
    N x N 입력이면 곱셈이 N^2번 일어나므로 시간 복잡도는 O(N^2)이다.
    """
    size = len(pattern)
    total = 0.0
    for row in range(size):
        for col in range(size):
            total += pattern[row][col] * matrix_filter[row][col]
    return total


def flatten(matrix):
    """2차원 배열을 길이 N^2의 1차원 배열로 펴준다."""
    flat = []
    for row in matrix:
        for value in row:
            flat.append(value)
    return flat


def mac_flat(flat_pattern, flat_filter):
    """1차원으로 펴둔 패턴과 필터로 MAC 연산을 한다.

    보너스 과제: 2차원 인덱싱(matrix[row][col])은 바깥 리스트를 먼저 찾고
    안쪽 리스트를 다시 찾는 2단계 접근이다. 미리 1차원으로 펴두면
    접근이 1단계로 줄어든다. 곱셈 횟수 자체는 N^2로 같다.
    """
    total = 0.0
    for index in range(len(flat_pattern)):
        total += flat_pattern[index] * flat_filter[index]
    return total


def measure_ms(function, left, right, repeat=REPEAT):
    """연산 함수를 repeat번 호출한 평균 시간을 ms로 돌려준다.

    입출력이나 파일 읽기는 측정 구간에 넣지 않고, 연산 함수 호출만 감싼다.
    """
    function(left, right)  # 첫 호출은 캐시가 비어 있어 느리므로 측정에서 뺀다
    start = time.perf_counter()
    for _ in range(repeat):
        function(left, right)
    elapsed = time.perf_counter() - start
    return (elapsed / repeat) * 1000


# ---------------------------------------------------------------
# 라벨 정규화
# ---------------------------------------------------------------

def normalize_label(raw):
    """'+', 'cross', 'Cross' 는 Cross 로, 'x', 'X' 는 X 로 바꾼다.

    data.json의 expected 값과 filter 키가 서로 다른 표기를 쓰기 때문에
    프로그램 안에서는 표준 라벨 두 가지(Cross, X)로 통일해서 비교한다.
    알 수 없는 표기는 None을 돌려줘서 호출한 쪽에서 FAIL 처리하게 한다.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip().lower()
    if text in ("+", "cross", "plus"):
        return CROSS
    if text == "x":
        return X_MARK
    return None


def decide(score_cross, score_x):
    """두 점수를 허용오차 기준으로 비교해 판정 라벨을 돌려준다."""
    if abs(score_cross - score_x) < EPSILON:
        return UNDECIDED
    if score_cross > score_x:
        return CROSS
    return X_MARK


# ---------------------------------------------------------------
# 패턴 생성기 (보너스)
# ---------------------------------------------------------------

def make_cross(size):
    """N x N 십자가 패턴을 만든다. 가운데 행과 가운데 열이 1이다."""
    middle = size // 2
    pattern = []
    for row in range(size):
        line = []
        for col in range(size):
            if row == middle or col == middle:
                line.append(1.0)
            else:
                line.append(0.0)
        pattern.append(line)
    return pattern


def make_x(size):
    """N x N X자 패턴을 만든다. 두 대각선이 1이다."""
    pattern = []
    for row in range(size):
        line = []
        for col in range(size):
            if row == col or row + col == size - 1:
                line.append(1.0)
            else:
                line.append(0.0)
        pattern.append(line)
    return pattern


# ---------------------------------------------------------------
# 입력 처리 (모드 1)
# ---------------------------------------------------------------

def read_row(size):
    """한 줄을 받아 size개의 실수 리스트로 바꾼다. 형식이 틀리면 None."""
    try:
        raw = input()
    except EOFError:
        return None
    parts = raw.split()
    # "0 1 0" 이 기본이지만 "010" 처럼 붙여 쓴 경우도 받아준다.
    if len(parts) == 1 and len(parts[0]) == size:
        parts = list(parts[0])
    if len(parts) != size:
        print("입력 형식 오류: 각 줄에 %d개의 숫자를 공백으로 구분해 입력하세요." % size)
        return None
    values = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            print("입력 형식 오류: '%s' 는 숫자가 아닙니다. 숫자만 입력하세요." % part)
            return None
    return values


def read_matrix(title, size):
    """size x size 행렬을 한 줄씩 입력받는다. 틀린 줄은 다시 받는다."""
    print("%s (%d줄 입력, 공백 구분)" % (title, size))
    matrix = []
    while len(matrix) < size:
        row = read_row(size)
        if row is None:
            print("%d번째 줄을 다시 입력하세요." % (len(matrix) + 1))
            continue
        matrix.append(row)
    return matrix


def print_matrix(matrix):
    """저장된 행렬을 확인용으로 찍는다."""
    for row in matrix:
        print("  " + " ".join(format_number(value) for value in row))


def format_number(value):
    """1.0 은 '1', 0.5 는 '0.5' 처럼 짧게 보여준다."""
    if value == int(value):
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------
# 성능 분석
# ---------------------------------------------------------------

def print_performance(sizes):
    """크기별 MAC 연산의 평균 시간과 연산 횟수를 표로 찍는다."""
    print("크기       평균 시간(ms)    연산 횟수")
    print("-" * 38)
    rows = []
    for size in sizes:
        pattern = make_x(size)
        matrix_filter = make_cross(size)
        average = measure_ms(mac, pattern, matrix_filter)
        label = "%dx%d" % (size, size)
        print("%-10s %-16.4f %d" % (label, average, size * size))
        rows.append((size, average))
    return rows


def print_optimization(sizes):
    """보너스: 2차원 접근과 1차원 접근의 평균 시간을 같은 조건으로 비교한다."""
    print("크기       2차원(ms)        1차원(ms)        변화율")
    print("-" * 56)
    for size in sizes:
        pattern = make_x(size)
        matrix_filter = make_cross(size)
        before = measure_ms(mac, pattern, matrix_filter)
        after = measure_ms(mac_flat, flatten(pattern), flatten(matrix_filter))
        if before > 0:
            change = (after - before) / before * 100
        else:
            change = 0.0
        label = "%dx%d" % (size, size)
        print("%-10s %-16.4f %-16.4f %+.1f%%" % (label, before, after, change))


# ---------------------------------------------------------------
# 모드 1: 사용자 입력 (3x3)
# ---------------------------------------------------------------

def run_user_mode():
    """3x3 필터 두 개와 패턴을 입력받아 MAC 점수를 비교한다."""
    size = 3

    print_header("[1] 필터 입력")
    filter_a = read_matrix("필터 A", size)
    print()
    filter_b = read_matrix("필터 B", size)

    print_header("[2] 저장 확인")
    print("필터 A")
    print_matrix(filter_a)
    print("필터 B")
    print_matrix(filter_b)

    print_header("[3] 패턴 입력")
    pattern = read_matrix("패턴", size)
    print("패턴")
    print_matrix(pattern)

    print_header("[4] MAC 결과")
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    average = measure_ms(mac, pattern, filter_a)

    print("A 점수: %s" % score_a)
    print("B 점수: %s" % score_b)
    print("연산 시간(평균/%d회): %.4f ms" % (REPEAT, average))

    if abs(score_a - score_b) < EPSILON:
        print("판정: 판정 불가 (|A-B| < %s)" % EPSILON)
    elif score_a > score_b:
        print("판정: A")
    else:
        print("판정: B")

    print_header("[5] 성능 분석 (평균/%d회)" % REPEAT)
    print_performance([size])


# ---------------------------------------------------------------
# 모드 2: data.json 분석
# ---------------------------------------------------------------

def is_square(matrix, size):
    """matrix가 size x size 2차원 숫자 배열인지 확인한다."""
    if not isinstance(matrix, list) or len(matrix) != size:
        return False
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            return False
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
    return True


def load_filters(raw_filters):
    """filters를 {크기: {표준라벨: 행렬}} 형태로 정규화한다."""
    filters = {}
    if not isinstance(raw_filters, dict):
        return filters
    for key, value in raw_filters.items():
        size = parse_size(key)
        if size is None or not isinstance(value, dict):
            print("경고: 필터 '%s' 는 형식을 알 수 없어 건너뜁니다." % key)
            continue
        table = {}
        for label_key, matrix in value.items():
            label = normalize_label(label_key)
            if label is None:
                print("경고: 필터 '%s' 의 라벨 '%s' 를 알 수 없어 건너뜁니다." % (key, label_key))
                continue
            if not is_square(matrix, size):
                print("경고: 필터 '%s' 의 '%s' 가 %dx%d 숫자 배열이 아니라 건너뜁니다."
                      % (key, label_key, size, size))
                continue
            table[label] = matrix
        filters[size] = table
        if table:
            loaded = ", ".join(sorted(table.keys()))
            print("  size_%d 필터 로드 완료 (%s)" % (size, loaded))
        else:
            print("  size_%d 필터를 하나도 읽지 못했습니다." % size)
    return filters


def parse_size(key):
    """'size_13' 또는 'size_13_1' 에서 13을 뽑아낸다."""
    parts = str(key).split("_")
    if len(parts) < 2 or parts[0] != "size":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def sort_key(key):
    """size_5_2 같은 키를 (크기, 번호) 순서로 정렬하기 위한 보조 함수."""
    parts = str(key).split("_")
    numbers = []
    for part in parts[1:]:
        try:
            numbers.append(int(part))
        except ValueError:
            numbers.append(0)
    return (numbers + [0, 0])[:2]


def check_case(key, case, filters):
    """케이스 하나를 검사한다. 문제가 있으면 (None, 사유)를 돌려준다."""
    size = parse_size(key)
    if size is None:
        return None, "패턴 키 '%s' 에서 크기를 읽을 수 없음" % key
    if not isinstance(case, dict):
        return None, "패턴 항목이 객체 형식이 아님"

    pattern = case.get("input")
    if not is_square(pattern, size):
        return None, "input이 %dx%d 숫자 배열이 아님" % (size, size)

    expected = normalize_label(case.get("expected"))
    if expected is None:
        return None, "expected 값 '%s' 를 표준 라벨로 바꿀 수 없음" % case.get("expected")

    table = filters.get(size)
    if table is None:
        return None, "size_%d 필터가 data.json에 없음" % size
    if CROSS not in table or X_MARK not in table:
        return None, "size_%d 필터에 Cross/X 가 모두 있지 않음" % size
    if not is_square(table[CROSS], size) or not is_square(table[X_MARK], size):
        return None, "size_%d 필터 크기가 패턴(%dx%d)과 맞지 않음" % (size, size, size)

    return (pattern, expected, table), None


def run_json_mode():
    """data.json을 읽어 모든 케이스를 일괄 판정한다."""
    print_header("[1] 필터 로드")
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as data_file:
            data = json.load(data_file)
    except FileNotFoundError:
        print("data.json을 찾을 수 없습니다: %s" % DATA_FILE)
        return
    except json.JSONDecodeError as error:
        print("data.json을 읽을 수 없습니다 (JSON 형식 오류): %s" % error)
        return
    except OSError as error:
        print("data.json을 읽을 수 없습니다: %s" % error)
        return

    if not isinstance(data, dict):
        print("data.json의 최상위가 객체 형식이 아닙니다.")
        return

    filters = load_filters(data.get("filters"))
    raw_patterns = data.get("patterns")
    if not isinstance(raw_patterns, dict):
        print("data.json에 patterns가 없거나 객체 형식이 아닙니다.")
        return

    print_header("[2] 패턴 분석 (라벨 정규화 적용)")
    passed = 0
    failures = []
    keys = sorted(raw_patterns.keys(), key=sort_key)

    for key in keys:
        print("--- %s ---" % key)
        parsed, reason = check_case(key, raw_patterns[key], filters)
        if parsed is None:
            print("판정: FAIL (%s)" % reason)
            failures.append((key, reason))
            continue

        pattern, expected, table = parsed
        score_cross = mac(pattern, table[CROSS])
        score_x = mac(pattern, table[X_MARK])
        verdict = decide(score_cross, score_x)

        print("Cross 점수: %s" % score_cross)
        print("X 점수: %s" % score_x)

        if verdict == expected:
            print("판정: %s | expected: %s | PASS" % (verdict, expected))
            passed += 1
        elif verdict == UNDECIDED:
            print("판정: %s | expected: %s | FAIL (동점 규칙)" % (verdict, expected))
            failures.append((key, "동점(UNDECIDED) 처리 규칙에 따라 FAIL"))
        else:
            print("판정: %s | expected: %s | FAIL" % (verdict, expected))
            failures.append((key, "판정 %s 와 expected %s 가 다름" % (verdict, expected)))

    print_header("[3] 성능 분석 (평균/%d회)" % REPEAT)
    print_performance(PERF_SIZES)

    print_header("[4] 최적화 비교 (2차원 배열 vs 1차원 배열)")
    print_optimization(PERF_SIZES)

    print_header("[5] 결과 요약")
    print("총 테스트: %d개" % len(keys))
    print("통과: %d개" % passed)
    print("실패: %d개" % len(failures))
    if failures:
        print()
        print("실패 케이스:")
        for key, reason in failures:
            print("- %s: %s" % (key, reason))


# ---------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------

def choose_mode():
    """모드 번호를 입력받는다. 종료를 원하면 None을 돌려준다."""
    while True:
        print()
        print("[모드 선택]")
        print()
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("3. 종료")
        try:
            choice = input("선택: ").strip()
        except EOFError:
            return None
        if choice in ("1", "2"):
            return choice
        if choice == "3":
            return None
        print("1, 2, 3 중에서 선택하세요.")


def main():
    print("=== Mini NPU Simulator ===")
    while True:
        choice = choose_mode()
        if choice is None:
            print("프로그램을 종료합니다.")
            return
        if choice == "1":
            run_user_mode()
        else:
            run_json_mode()


if __name__ == "__main__":
    main()
