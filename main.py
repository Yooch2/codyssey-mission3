"""Mini NPU Simulator

MAC(Multiply-Accumulate) 연산으로 입력 패턴이 십자가(Cross)에 가까운지
X자에 가까운지 판별하는 콘솔 애플리케이션이다.
표준 라이브러리(json, math, os, time, unicodedata)만 사용한다.
"""

import json
import math
import os
import time
import unicodedata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")

# 점수 비교에 쓰는 허용오차. 두 점수의 차이가 이 값보다 작으면 동점으로 본다.
EPSILON = 1e-9

# 성능 측정 반복 횟수(미션 최소 기준 10회)
REPEAT = 10

# 측정 전에 버리는 호출 횟수. 첫 호출은 데이터가 CPU 캐시에 없어 느릴 수 있다.
# 25x25에서 워밍업 횟수를 한 라운드 안에 번갈아 재고 라운드마다 순서를
# 섞어 봤더니, 0회에서 10회까지의 차이가 같은 조건을 되풀이했을 때의
# 흔들림보다 작았다(README 3-5절). 즉 이 크기에서는 효과가 확인되지 않는다.
# 비용이 호출 한 번뿐이라 더 큰 입력을 위한 안전장치로 1을 남겼다.
# warmup 인자로 바꿀 수 있다.
WARMUP = 1

# 프로그램 내부에서 쓰는 표준 라벨
CROSS = "Cross"
X_MARK = "X"
UNDECIDED = "UNDECIDED"

# 성능 분석에 사용할 크기 목록
PERF_SIZES = [3, 5, 13, 25]

# 보너스 패턴 생성기가 받는 크기 범위. 홀수만 받는다.
# 하한 3은 미션이 정한 최소 크기다. N이 1이면 십자가와 X자가 똑같은
# 행렬([[1.0]])이 되어 비교 자체가 성립하지 않는다.
# 상한 999는 README 5절에서 측정해본 가장 큰 크기(1000) 아래의 홀수다.
# 더 큰 값도 돌아가지만 재본 적이 없어 받지 않는다.
#
# 짝수를 막는 이유는 짝수 N에서 두 패턴이 공평하지 않기 때문이다(README 3-6절).
# 십자가는 가운데 행과 열이 하나여야 대칭이 되는데, 짝수면 size//2가
# 한쪽으로 치우친 행을 고른다(N=10이면 위 5줄 아래 4줄).
# 게다가 X자는 두 대각선이 중앙에서 겹치지 않아 1인 칸이 2N개가 되고,
# 십자가는 2N-1개에 머문다. 필터 한쪽이 1칸 무거우면 중립적인 입력도
# X로 기운다. 실제로 N=10에서 전부 1인 패턴이 X로 판정됐다.
GENERATOR_MIN_SIZE = 3
GENERATOR_MAX_SIZE = 999

# 생성한 행렬을 화면에 그대로 찍는 크기의 한계.
# 이보다 크면 줄 수가 너무 많아 다른 출력을 밀어내므로 요약만 보여준다.
GENERATOR_PRINT_LIMIT = 25

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


def measure_ms(function, left, right, repeat=REPEAT, warmup=WARMUP):
    """연산 함수를 repeat번 호출한 평균 시간을 ms로 돌려준다.

    입출력이나 파일 읽기는 측정 구간에 넣지 않고, 연산 함수 호출만 감싼다.
    warmup번의 호출은 측정 구간 밖에서 미리 돌려 캐시를 데운 뒤 버린다.
    """
    for _ in range(warmup):
        function(left, right)
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
    """한 줄을 받아 size개의 실수 리스트로 바꾼다. 형식이 틀리면 None.

    입력이 끝나면(Ctrl+D, 파이프 종료) EOFError가 그대로 위로 올라간다.
    형식 오류는 다시 받으면 되지만 입력이 끊긴 상황은 재입력이 불가능해서,
    둘을 구분하지 않으면 안내 문구만 끝없이 찍히게 된다.
    """
    raw = input()
    parts = raw.split()
    # "0 1 0" 이 기본이지만 "010" 처럼 붙여 쓴 경우도 받아준다.
    # 전부 숫자일 때만 쪼갠다. "0.5"나 "-11"까지 쪼개면 '.'이나 '-'를
    # 숫자가 아니라고 지적하게 되어, 정작 개수가 모자란다는 사실이 가려진다.
    if len(parts) == 1 and len(parts[0]) == size and parts[0].isdigit():
        parts = list(parts[0])
    if len(parts) != size:
        print("입력 형식 오류: 각 줄에 %d개의 숫자를 공백으로 구분해 입력하세요." % size)
        return None
    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError:
            print("입력 형식 오류: '%s' 는 숫자가 아닙니다. 숫자만 입력하세요." % part)
            return None
        # float()은 'inf'와 'nan'도 받아들인다. 그대로 두면 점수가 nan이 되어
        # 모든 비교가 거짓이 되고 판정이 뜻을 잃는다. 입력 단계에서 막는다.
        if not math.isfinite(value):
            print("입력 형식 오류: '%s' 는 사용할 수 없습니다. 유한한 숫자만 입력하세요." % part)
            return None
        values.append(value)
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
    if not math.isfinite(value):
        # int(inf)와 int(nan)은 예외를 낸다. 입력 단계에서 걸러지지만
        # 다른 경로로 들어와도 출력이 죽지 않도록 그대로 찍는다.
        return str(value)
    if value == int(value):
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------
# 성능 분석
# ---------------------------------------------------------------

def display_width(text):
    """터미널에 찍히는 칸 수를 센다.

    한글과 전각 기호는 글자 하나가 두 칸을 차지한다. len()은 이를 1로 세기
    때문에 그대로 쓰면 한글 머리글이 들어간 표의 열이 어긋난다.
    """
    width = 0
    for character in text:
        if unicodedata.east_asian_width(character) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def print_table(headers, rows):
    """열 너비를 내용에 맞춰 잡아 표를 찍는다.

    폭을 숫자로 박아두면 값이 길어졌을 때 칸이 밀려 읽기 어려워진다.
    머리글과 모든 칸의 폭을 먼저 재서 가장 넓은 것에 맞춘다.
    """
    widths = []
    for index, header in enumerate(headers):
        widest = display_width(header)
        for row in rows:
            widest = max(widest, display_width(row[index]))
        widths.append(widest)

    def format_row(cells):
        padded = []
        for index, cell in enumerate(cells):
            padding = widths[index] - display_width(cell)
            padded.append(cell + " " * padding)
        return "  ".join(padded).rstrip()

    print(format_row(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(format_row(row))


def print_performance(sizes):
    """크기별 MAC 연산의 평균 시간과 연산 횟수를 표로 찍는다."""
    rows = []
    measured = []
    for size in sizes:
        pattern = make_x(size)
        matrix_filter = make_cross(size)
        average = measure_ms(mac, pattern, matrix_filter)
        rows.append(["%dx%d" % (size, size), "%.4f" % average, str(size * size)])
        measured.append((size, average))
    print_table(["크기", "평균 시간(ms)", "연산 횟수"], rows)
    return measured


def print_optimization(sizes):
    """보너스: 2차원 접근과 1차원 접근의 평균 시간을 같은 조건으로 비교한다."""
    rows = []
    for size in sizes:
        pattern = make_x(size)
        matrix_filter = make_cross(size)
        before = measure_ms(mac, pattern, matrix_filter)
        after = measure_ms(mac_flat, flatten(pattern), flatten(matrix_filter))
        if before > 0:
            change = "%+.1f%%" % ((after - before) / before * 100)
        else:
            change = "측정 불가"
        rows.append(["%dx%d" % (size, size), "%.4f" % before, "%.4f" % after, change])
    print_table(["크기", "2차원(ms)", "1차원(ms)", "변화율"], rows)


# ---------------------------------------------------------------
# 모드 1: 사용자 입력 (3x3)
# ---------------------------------------------------------------

def run_user_mode():
    """3x3 필터 두 개와 패턴을 입력받아 MAC 점수를 비교한다.

    입력 도중 Ctrl+D로 입력이 끝나면 안내를 남기고 메뉴로 돌아간다.
    """
    try:
        run_user_mode_steps()
    except EOFError:
        print()
        print("입력이 끝나 모드 1을 중단합니다. 메뉴로 돌아갑니다.")


def ask_filter_source():
    """필터를 직접 입력할지 패턴 생성기로 만들지 고른다.

    생성기(make_cross/make_x)는 원래 성능 분석 입력을 만들 때만 썼다.
    같은 함수를 모드 1에도 연결해, 9칸을 두 번 입력하지 않고도
    표준 필터로 바로 돌려볼 수 있게 했다.
    """
    while True:
        print("필터를 어떻게 채울까요?")
        print("1. 직접 입력")
        print("2. 패턴 생성기로 생성 (A=십자가, B=X자)")
        choice = input("선택: ").strip()
        if choice == "1":
            return False
        if choice == "2":
            return True
        print("1 또는 2 중에서 선택하세요.")


def run_user_mode_steps():
    """모드 1의 실제 입력-연산-출력 흐름."""
    size = 3

    print_header("[1] 필터 입력")
    if ask_filter_source():
        filter_a = make_cross(size)
        filter_b = make_x(size)
        print("패턴 생성기로 %dx%d 필터를 만들었습니다. (A=십자가, B=X자)"
              % (size, size))
    else:
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
        if size in filters:
            print("경고: size_%d 필터가 이미 있습니다. '%s' 의 내용으로 덮어씁니다."
                  % (size, key))
        filters[size] = table
        if table:
            loaded = ", ".join(sorted(table.keys()))
            print("  size_%d 필터 로드 완료 (%s)" % (size, loaded))
        else:
            print("  size_%d 필터를 하나도 읽지 못했습니다." % size)
    return filters


def parse_size(key):
    """'size_13' 또는 'size_13_1' 에서 13을 뽑아낸다.

    형식이 맞지 않거나 크기가 1보다 작으면 None을 돌려준다.
    0이나 음수 크기는 행렬로 성립하지 않는데, 거르지 않으면
    빈 배열이 '0x0 정사각 행렬'로 통과해 점수 0.0 대 0.0의
    가짜 동점 판정이 나온다.
    """
    parts = str(key).split("_")
    if len(parts) < 2 or parts[0] != "size":
        return None
    try:
        size = int(parts[1])
    except ValueError:
        return None
    if size < 1:
        return None
    return size


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
    # 어느 파일로 채점했는지 남긴다. 다른 디렉터리에서 실행해도
    # main.py 옆의 data.json을 읽으므로, 경로를 보여줘야 확인할 수 있다.
    print("데이터 파일: %s" % DATA_FILE)
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
    failures = []          # (케이스, 사유, 점수 상세)
    undecided_count = 0    # 동점으로 FAIL 난 개수
    mismatch_count = 0     # 판정은 났지만 expected와 다른 개수
    schema_count = 0       # 데이터/스키마 문제로 판정 자체를 못 한 개수
    keys = sorted(raw_patterns.keys(), key=sort_key)

    for key in keys:
        print("--- %s ---" % key)
        parsed, reason = check_case(key, raw_patterns[key], filters)
        if parsed is None:
            print("판정: FAIL (%s)" % reason)
            failures.append((key, reason, ""))
            schema_count += 1
            continue

        pattern, expected, table = parsed
        score_cross = mac(pattern, table[CROSS])
        score_x = mac(pattern, table[X_MARK])
        verdict = decide(score_cross, score_x)
        detail = "Cross %s / X %s, 차이 %.3e" % (
            score_cross, score_x, abs(score_cross - score_x))

        print("Cross 점수: %s" % score_cross)
        print("X 점수: %s" % score_x)

        if verdict == expected:
            print("판정: %s | expected: %s | PASS" % (verdict, expected))
            passed += 1
        elif verdict == UNDECIDED:
            print("판정: %s | expected: %s | FAIL (동점 규칙)" % (verdict, expected))
            failures.append((key, "동점(UNDECIDED) 처리 규칙에 따라 FAIL", detail))
            undecided_count += 1
        else:
            print("판정: %s | expected: %s | FAIL" % (verdict, expected))
            failures.append((key, "판정 %s 와 expected %s 가 다름" % (verdict, expected), detail))
            mismatch_count += 1

    print_header("[3] 성능 분석 (평균/%d회)" % REPEAT)
    print_performance(PERF_SIZES)

    print_header("[4] 최적화 비교 (2차원 배열 vs 1차원 배열)")
    print_optimization(PERF_SIZES)

    print_header("[5] 결과 요약")
    print("총 테스트: %d개" % len(keys))
    print("통과: %d개" % passed)
    print("실패: %d개" % len(failures))
    print("  - 동점(UNDECIDED): %d개" % undecided_count)
    print("  - 라벨 불일치: %d개" % mismatch_count)
    print("  - 데이터/스키마 오류: %d개" % schema_count)
    if failures:
        print()
        print("실패 케이스:")
        for key, reason, detail in failures:
            print("- %s: %s" % (key, reason))
            if detail:
                print("    %s" % detail)


# ---------------------------------------------------------------
# 모드 3: 패턴 생성기 (보너스)
# ---------------------------------------------------------------

def ask_size():
    """생성할 패턴의 크기 N을 입력받는다. 조건에 맞지 않으면 다시 받는다."""
    prompt = "크기 N (%d~%d 홀수): " % (GENERATOR_MIN_SIZE, GENERATOR_MAX_SIZE)
    while True:
        raw = input(prompt).strip()
        try:
            size = int(raw)
        except ValueError:
            print("'%s' 는 정수가 아닙니다. 정수를 입력하세요." % raw)
            continue
        if size < GENERATOR_MIN_SIZE or size > GENERATOR_MAX_SIZE:
            print("%d 이상 %d 이하로 입력하세요."
                  % (GENERATOR_MIN_SIZE, GENERATOR_MAX_SIZE))
            continue
        if size % 2 == 0:
            print("%d 은 짝수입니다. 짝수에서는 십자가의 가운데 행이 한쪽으로"
                  " 치우치고, X자의 1인 칸이 십자가보다 하나 많아져" % size)
            print("두 필터가 공평하지 않습니다. 홀수를 입력하세요 (%d 또는 %d)."
                  % (size - 1, size + 1))
            continue
        return size


def count_ones(matrix):
    """행렬에서 0이 아닌 칸의 개수를 센다."""
    total = 0
    for row in matrix:
        for value in row:
            if value:
                total += 1
    return total


def show_generated(label, matrix, size):
    """생성한 패턴을 보여준다. 큰 행렬은 찍지 않고 요약만 남긴다."""
    print(label)
    if size <= GENERATOR_PRINT_LIMIT:
        print_matrix(matrix)
    else:
        print("  (%dx%d 라 생략. 값이 1인 칸 %d개 / 전체 %d칸)"
              % (size, size, count_ones(matrix), size * size))


def run_generator_mode():
    """보너스: 크기 N을 받아 십자가와 X자를 만들고 그대로 분석까지 돌린다.

    생성한 패턴을 바로 MAC 판정과 성능 분석에 넘겨서,
    생성기가 분석 경로에 실제로 연결돼 있다는 것을 한 화면에서 보여준다.
    """
    try:
        size = ask_size()
    except EOFError:
        print()
        print("입력이 끝나 모드 3을 중단합니다. 메뉴로 돌아갑니다.")
        return

    cross = make_cross(size)
    x_mark = make_x(size)

    print_header("[1] 생성 결과 (%dx%d)" % (size, size))
    show_generated("Cross", cross, size)
    show_generated("X", x_mark, size)

    print_header("[2] 생성한 패턴으로 MAC 판정")
    # 생성한 패턴을 두 필터와 비교하면 자기 쪽 필터가 이겨야 한다.
    # 그렇지 않으면 생성기나 MAC 연산 중 하나가 잘못된 것이다.
    for label, pattern in ((CROSS, cross), (X_MARK, x_mark)):
        score_cross = mac(pattern, cross)
        score_x = mac(pattern, x_mark)
        verdict = decide(score_cross, score_x)
        print("%s 패턴 | Cross 점수 %s | X 점수 %s | 판정 %s | %s"
              % (label, score_cross, score_x, verdict,
                 "일치" if verdict == label else "불일치"))

    print_header("[3] 성능 분석 (평균/%d회)" % REPEAT)
    print_performance([size])

    print_header("[4] 최적화 비교 (2차원 배열 vs 1차원 배열)")
    print_optimization([size])


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
        print("3. 패턴 생성기 (크기 N)")
        print("4. 종료")
        try:
            choice = input("선택: ").strip()
        except EOFError:
            return None
        if choice in ("1", "2", "3"):
            return choice
        if choice == "4":
            return None
        print("1, 2, 3, 4 중에서 선택하세요.")


def main():
    print("=== Mini NPU Simulator ===")
    while True:
        choice = choose_mode()
        if choice is None:
            print("프로그램을 종료합니다.")
            return
        if choice == "1":
            run_user_mode()
        elif choice == "2":
            run_json_mode()
        else:
            run_generator_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("사용자가 중단했습니다.")
