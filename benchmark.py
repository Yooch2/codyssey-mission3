"""README의 측정 수치를 다시 만들어내는 스크립트.

main.py의 함수를 그대로 불러 쓰며, 측정 방법도 본체와 같다.
결과를 README에 적힌 표 모양으로 찍으므로 수치를 대조할 수 있다.

실행: python3 benchmark.py
     python3 benchmark.py 성능        (일부만 실행)

측정값은 실행할 때마다 흔들린다. 같은 기기에서도 다른 작업이 돌고 있으면
25x25 측정값이 1.5배까지 벌어지는 것을 확인했다. 표의 각 칸은 여러 번
실행한 중앙값이고, 절대 시간보다 크기별 비율을 보는 것이 맞다.
"""

import array
import json
import platform
import statistics
import sys
import tracemalloc
from fractions import Fraction

import main

# 대규모 측정에 쓰는 크기. 1000x1000은 한 번에 20ms 남짓 걸린다.
BIG_SIZES = [100, 250, 500, 1000]


def median_ms(function, left, right, repeat=main.REPEAT, runs=5):
    """같은 측정을 runs번 되풀이한 중앙값을 돌려준다."""
    return statistics.median(
        main.measure_ms(function, left, right, repeat=repeat) for _ in range(runs))


def fit_quadratic(sizes, times):
    """시간 = 고정비용 + 곱셈당비용 x N^2 을 최소제곱으로 맞춘다.

    작은 크기에서는 함수 호출 같은 고정 비용이 전체를 가려서,
    실측 시간 배율만 보면 N^2보다 느리게 늘어나는 것처럼 보인다.
    고정 비용을 분리해야 곱셈 하나의 비용이 크기와 무관하게
    일정하다는 것을 확인할 수 있다.
    """
    squares = [size * size for size in sizes]
    count = len(squares)
    sum_x = sum(squares)
    sum_y = sum(times)
    sum_xx = sum(x * x for x in squares)
    sum_xy = sum(x * y for x, y in zip(squares, times))
    slope = (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / count
    return intercept, slope


def peak_mb(build):
    """build()가 만드는 동안의 최대 메모리 사용량을 MB로 돌려준다."""
    tracemalloc.start()
    obj = build()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    del obj
    return peak / 1048576


def section(title):
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def bench_performance():
    """README 4-3: 미션 크기의 성능 표, 배율, O(N^2) 모델."""
    section("[4-3] 성능 표와 시간 복잡도")
    sizes = main.PERF_SIZES
    two_dim = []
    one_dim = []
    for size in sizes:
        pattern = main.make_x(size)
        matrix_filter = main.make_cross(size)
        two_dim.append(median_ms(main.mac, pattern, matrix_filter))
        one_dim.append(median_ms(main.mac_flat,
                                 main.flatten(pattern), main.flatten(matrix_filter)))

    rows = []
    for size, before, after in zip(sizes, two_dim, one_dim):
        rows.append(["%dx%d" % (size, size), "%.4f" % before, "%.4f" % after,
                     "%+.1f%%" % ((after - before) / before * 100), str(size * size)])
    main.print_table(["크기", "평균 시간(ms)", "1차원(ms)", "변화율", "연산 횟수"], rows)

    print()
    print("3x3을 1배로 둔 배율")
    rows = []
    for size, time_ms in zip(sizes, two_dim):
        rows.append(["%dx%d" % (size, size),
                     "x%.1f" % ((size * size) / (sizes[0] * sizes[0])),
                     "x%.1f" % (time_ms / two_dim[0])])
    main.print_table(["크기", "연산 횟수 배율", "실측 시간 배율"], rows)

    intercept, slope = fit_quadratic(sizes, two_dim)
    print()
    print("모델: 시간(ms) = %.6f + %.4e x N^2" % (intercept, slope))
    print("      고정 비용 %.4f ms / 곱셈 1회당 %.1f ns" % (intercept, slope * 1e6))
    rows = []
    for size, time_ms in zip(sizes, two_dim):
        predicted = intercept + slope * size * size
        rows.append(["%dx%d" % (size, size), str(size * size), "%.4f" % time_ms,
                     "%.4f" % predicted, "%+.1f%%" % ((predicted - time_ms) / time_ms * 100)])
    main.print_table(["크기", "N^2", "실측(ms)", "모델(ms)", "오차"], rows)
    return intercept, slope


def bench_large(intercept, slope):
    """README 5-1, 5-2: 대규모 입력의 시간, 모델 외삽, 처리량, 메모리."""
    section("[5-1] 대규모 입력")
    rows = []
    for size in BIG_SIZES:
        pattern = main.make_x(size)
        matrix_filter = main.make_cross(size)
        repeat = 10 if size <= 250 else 3
        before = median_ms(main.mac, pattern, matrix_filter, repeat=repeat, runs=3)
        after = median_ms(main.mac_flat, main.flatten(pattern),
                          main.flatten(matrix_filter), repeat=repeat, runs=3)
        predicted = intercept + slope * size * size
        rows.append(["%dx%d" % (size, size), "%d" % (size * size),
                     "%.3f" % before, "%.3f" % after,
                     "%+.1f%%" % ((after - before) / before * 100),
                     "%.3f" % predicted,
                     "%+.0f%%" % ((predicted - before) / before * 100),
                     "%.0f M MAC/s" % ((size * size) / (before / 1000) / 1e6)])
    main.print_table(["크기", "N^2", "2차원(ms)", "1차원(ms)", "변화율",
                      "모델(ms)", "모델 오차", "처리량"], rows)

    section("[5-2] 메모리")
    size = BIG_SIZES[-1]
    shared = peak_mb(lambda: main.make_x(size))
    distinct = peak_mb(lambda: [[row * size + col + 0.5 for col in range(size)]
                                for row in range(size)])
    main.print_table(["%dx%d 행렬 하나" % (size, size), "메모리"],
                     [["0과 1만 쓰는 2차원 리스트 (make_x)", "%.2f MB" % shared],
                      ["값이 전부 다른 2차원 리스트", "%.2f MB" % distinct]])
    print()
    print("같은 크기인데 %.1f배 차이가 난다." % (distinct / shared))
    print("make_x가 쓰는 0.0과 1.0은 함수 안에 적힌 상수라 객체 하나를 공유한다:")
    print("  make_x(5)[0][0] is make_x(5)[1][1] ->",
          main.make_x(5)[0][0] is main.make_x(5)[1][1])


def bench_optimization():
    """README 5-3: 1000x1000에서 최적화 후보 네 가지 비교."""
    section("[5-3] 우선 검토할 최적화 (1000x1000)")
    size = 1000
    pattern = main.make_x(size)
    matrix_filter = main.make_cross(size)
    flat_pattern = main.flatten(pattern)
    flat_filter = main.flatten(matrix_filter)
    array_pattern = array.array("d", flat_pattern)
    array_filter = array.array("d", flat_filter)

    def mac_sparse(left, right):
        """0인 칸은 곱해도 0이므로 건너뛴다."""
        total = 0.0
        for index, value in enumerate(left):
            if value:
                total += value * right[index]
        return total

    ways = [("2차원 리스트 (현재 mac)", main.mac, pattern, matrix_filter),
            ("1차원 리스트 (현재 mac_flat)", main.mac_flat, flat_pattern, flat_filter),
            ("array('d') 1차원", main.mac_flat, array_pattern, array_filter),
            ("0인 칸 건너뛰기", mac_sparse, flat_pattern, flat_filter)]

    results = []
    rows = []
    base = None
    for name, function, left, right in ways:
        elapsed = median_ms(function, left, right, repeat=3, runs=3)
        results.append(function(left, right))
        if base is None:
            base = elapsed
            change = "기준"
        else:
            change = "%+.0f%%" % ((elapsed - base) / base * 100)
        rows.append([name, "%.2f ms" % elapsed, change])
    main.print_table(["방식", "시간", "기준 대비"], rows)

    nonzero = sum(1 for value in flat_pattern if value)
    print()
    print("패턴의 0이 아닌 칸: %d / %d (%.2f%%)" % (nonzero, size * size,
                                                nonzero / (size * size) * 100))
    print("네 방식의 결과가 모두 같은가:", len(set(results)) == 1, results[0])


def bench_epsilon():
    """README 3-3: EPSILON 값을 바꿨을 때 PASS/FAIL이 어떻게 달라지는가."""
    section("[3-3] EPSILON 민감도")
    with open(main.DATA_FILE, "r", encoding="utf-8") as data_file:
        data = json.load(data_file)
    filters = {}
    for key, value in data["filters"].items():
        filters[main.parse_size(key)] = {
            main.normalize_label(label): matrix for label, matrix in value.items()}

    cases = []
    for key in sorted(data["patterns"], key=main.sort_key):
        case = data["patterns"][key]
        table = filters[main.parse_size(key)]
        cases.append((key,
                      main.mac(case["input"], table[main.CROSS]),
                      main.mac(case["input"], table[main.X_MARK]),
                      main.normalize_label(case["expected"])))

    rows = []
    for epsilon in (0.0, 1e-18, 1e-15, 1e-12, 1e-9, 1e-6, 1.0, 10.0, 100.0):
        passed = failed = tied = 0
        for _, score_cross, score_x, expected in cases:
            if abs(score_cross - score_x) < epsilon:
                verdict = main.UNDECIDED
                tied += 1
            elif score_cross > score_x:
                verdict = main.CROSS
            else:
                verdict = main.X_MARK
            if verdict == expected:
                passed += 1
            else:
                failed += 1
        label = str(epsilon) + (" (현재)" if epsilon == main.EPSILON else "")
        rows.append([label, str(passed), str(failed), str(tied)])
    main.print_table(["EPSILON", "PASS", "FAIL", "동점 판정 수"], rows)

    print()
    print("케이스별 두 점수의 차이")
    rows = []
    for key, score_cross, score_x, _ in sorted(cases, key=lambda c: abs(c[1] - c[2])):
        rows.append([key, "%.3e" % abs(score_cross - score_x)])
    main.print_table(["케이스", "차이"], rows)

    print()
    print("동점 케이스를 유리수로 다시 계산 (오차 없이 계산하면 정말 같은가)")
    rows = []
    for key, score_cross, score_x, _ in cases:
        if abs(score_cross - score_x) >= main.EPSILON:
            continue
        case = data["patterns"][key]
        table = filters[main.parse_size(key)]

        def exact(pattern, matrix_filter):
            total = Fraction(0)
            for row in range(len(pattern)):
                for col in range(len(pattern)):
                    total += (Fraction(str(pattern[row][col]))
                              * Fraction(str(matrix_filter[row][col])))
            return total

        exact_cross = exact(case["input"], table[main.CROSS])
        exact_x = exact(case["input"], table[main.X_MARK])
        rows.append([key, str(score_cross), str(score_x),
                     str(exact_cross), str(exact_x),
                     "같음" if exact_cross == exact_x else "다름"])
    main.print_table(["케이스", "float Cross", "float X",
                      "유리수 Cross", "유리수 X", "판정"], rows)


def bench_warmup():
    """README 3-5: 워밍업 횟수를 바꾸면 측정값이 어떻게 달라지는가."""
    section("[3-5] 워밍업 횟수별 측정값 (25x25, 각 9회 실행)")
    rows = []
    for warmup in (0, 1, 2, 5, 10):
        values = []
        for _ in range(9):
            pattern = main.make_x(25)
            matrix_filter = main.make_cross(25)
            values.append(main.measure_ms(main.mac, pattern, matrix_filter,
                                          warmup=warmup))
        rows.append(["%d회" % warmup, "%.4f" % statistics.median(values),
                     "%.4f" % min(values), "%.4f" % max(values)])
    main.print_table(["워밍업", "평균 시간(ms)", "최소", "최대"], rows)

    # 결론을 적어두지 않고 이번 실행의 값으로 계산해 보고한다.
    # 실행에 따라 워밍업 효과가 보이기도 하고 전혀 안 보이기도 한다.
    medians = [float(row[1]) for row in rows]
    change = (medians[-1] - medians[0]) / medians[0] * 100
    spread = max(medians) - min(medians)
    print()
    print("이번 실행에서 0회 대비 10회의 변화: %+.1f%%" % change)
    print("워밍업 횟수별 중앙값의 최대-최소 폭: %.4f ms" % spread)
    if abs(change) < 2:
        print("→ 이번 실행에서는 워밍업 효과가 측정 잡음에 묻혀 보이지 않는다.")
    else:
        print("→ 이번 실행에서는 워밍업 횟수가 늘수록 값이 낮아지는 경향이 보인다.")
    print("두 결과 모두 관측된 적이 있다. 자세한 내용은 README 3-5절에 적었다.")


def main_entry():
    print("Mini NPU Simulator - 측정 재현 스크립트")
    print("환경: %s %s / Python %s" % (platform.system(), platform.machine(),
                                     platform.python_version()))
    print("측정: 각 크기 %d회 반복의 평균, 그 측정을 여러 번 되풀이한 중앙값" % main.REPEAT)
    print("데이터: %s" % main.DATA_FILE)

    wanted = sys.argv[1:]
    runs = {
        "성능": lambda: bench_performance(),
        "대규모": None,          # 성능 결과(모델)가 필요해서 아래에서 따로 처리
        "최적화": bench_optimization,
        "엡실론": bench_epsilon,
        "워밍업": bench_warmup,
    }
    if wanted:
        unknown = [name for name in wanted if name not in runs]
        if unknown:
            print("\n알 수 없는 이름: %s" % ", ".join(unknown))
            print("쓸 수 있는 이름: %s" % ", ".join(runs))
            return

    model = None
    if not wanted or "성능" in wanted or "대규모" in wanted:
        model = bench_performance()
    if not wanted or "대규모" in wanted:
        bench_large(*model)
    for name in ("최적화", "엡실론", "워밍업"):
        if not wanted or name in wanted:
            runs[name]()


if __name__ == "__main__":
    main_entry()
