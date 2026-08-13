"""전체 파이프라인 실행: 시세 -> 뉴스/게시판(네이버, 팍스넷, 이데일리) -> 전처리.

네이버 뉴스/게시판은 매일 한 번씩 돌리면(예: 스케줄러 등록) id/nid 기준 중복 제거로
데이터가 계속 누적됩니다 (README "데이터 수집 범위의 현실적 한계" 참고).
팍스넷·이데일리는 페이지 상한이 없어 실행할 때마다 설정된 3개월 range를 통째로 다시 훑습니다.
"""
import crawl_board
import crawl_edaily_news
import crawl_news
import crawl_paxnet_board
import crawl_price
import crawl_toss_community
import preprocess


def main():
    print("=== 1. 시세(정형) 데이터 수집 ===")
    crawl_price.main()

    print("\n=== 2. 네이버 뉴스(비정형) 데이터 수집 ===")
    crawl_news.main()

    print("\n=== 3. 네이버 주주토론방(비정형) 데이터 수집 ===")
    crawl_board.main()

    print("\n=== 4. 팍스넷 종목토론방(비정형) 데이터 수집 ===")
    crawl_paxnet_board.main()

    print("\n=== 5. 이데일리 증권뉴스(비정형) 데이터 수집 ===")
    crawl_edaily_news.main()

    print("\n=== 6. 토스증권 커뮤니티(비정형) 데이터 수집 ===")
    crawl_toss_community.main()

    print("\n=== 7. 텍스트 전처리 ===")
    preprocess.main()


if __name__ == "__main__":
    main()
