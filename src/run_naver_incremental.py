"""네이버 주주토론방의 신규 데이터만 누적 수집한다.

매 실행은 최신 페이지부터 시작해 기존 CSV의 ID만 있는 페이지를 만나면 즉시 종료한다.
삼성전자·SK하이닉스 게시판은 글이 빠르게 쌓이므로 2~3시간 간격 실행을 권장한다.
"""
import crawl_board


def main() -> None:
    print("=== 네이버 주주토론방 증분 수집 ===")
    crawl_board.main()


if __name__ == "__main__":
    main()
