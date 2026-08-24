"""전체 파이프라인 실행: 시세 -> 게시판(네이버, 팍스넷)·이데일리·토스 -> 전처리.

네이버 게시판은 주기적으로 돌리면 nid 기준 중복 제거로 데이터가 계속 누적됩니다.
팍스넷·이데일리는 페이지 상한이 없어 실행할 때마다 설정된 3개월 range를 통째로 다시 훑습니다.
"""
import crawl_board
import crawl_edaily_news
import crawl_paxnet_board
import crawl_price
import crawl_toss_community
import preprocess


def main():
    print("=== 1. 시세(정형) 데이터 수집 ===")
    crawl_price.main()

    print("\n=== 2. 네이버 주주토론방(비정형) 데이터 수집 ===")
    crawl_board.main()

    print("\n=== 3. 팍스넷 종목토론방(비정형) 데이터 수집 ===")
    crawl_paxnet_board.main()

    print("\n=== 4. 이데일리 증권뉴스(비정형) 데이터 수집 ===")
    crawl_edaily_news.main()

    print("\n=== 5. 토스증권 커뮤니티(비정형) 데이터 수집 ===")
    crawl_toss_community.main()

    print("\n=== 6. 텍스트 전처리 ===")
    preprocess.main()


if __name__ == "__main__":
    main()
