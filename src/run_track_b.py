"""Track B 통계분석 파이프라인 실행."""

import track_b_analysis


def main():
    print("=== Track B 통계분석 파이프라인 시작 ===")
    print("Drive에서 받은 원본 데이터는 수정하지 않고 분석용 결과를 생성합니다.")

    track_b_analysis.main()

    print("=== Track B 통계분석 파이프라인 완료 ===")


if __name__ == "__main__":
    main()