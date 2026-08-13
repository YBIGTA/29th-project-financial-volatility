"""전체 파이프라인 실행: 시세 -> 뉴스 -> 게시판 -> 전처리.

매일 한 번씩 돌리면(예: 스케줄러 등록) nid/id 기준 중복 제거로 데이터가 계속 누적됩니다.
"""
import crawl_board
import crawl_news
import crawl_price
import preprocess


def main():
    print("=== 1. 시세(정형) 데이터 수집 ===")
    crawl_price.main()

    print("\n=== 2. 뉴스(비정형) 데이터 수집 ===")
    crawl_news.main()

    print("\n=== 3. 주주토론방(비정형) 데이터 수집 ===")
    crawl_board.main()

    print("\n=== 4. 텍스트 전처리 ===")
    preprocess.main()


if __name__ == "__main__":
    main()
