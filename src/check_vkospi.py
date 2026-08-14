from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd


START_DATE = "2026-05-16"
END_DATE = "2026-08-14"

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_symbol(symbol: str) -> pd.DataFrame | None:
    print("=" * 60)
    print(f"테스트 심볼: {symbol}")

    try:
        df = fdr.DataReader(
            symbol,
            START_DATE,
            END_DATE
        )

        print("수집 성공")
        print("데이터 크기:", df.shape)
        print("컬럼:", df.columns.tolist())
        print("시작일:", df.index.min())
        print("종료일:", df.index.max())
        print("\n앞부분")
        print(df.head())
        print("\n뒷부분")
        print(df.tail())
        print("\n결측치")
        print(df.isna().sum())

        return df

    except Exception as error:
        print("수집 실패")
        print("오류 종류:", type(error).__name__)
        print("오류 내용:", error)

        return None


vix = check_symbol("VIX")
vkospi = check_symbol("VKOSPI")


if vix is not None and not vix.empty:
    vix = vix.reset_index()
    vix.columns = [
        str(column).strip().lower()
        for column in vix.columns
    ]

    vix.to_csv(
        OUTPUT_DIR / "price_daily_VIX_test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nVIX 테스트 파일 저장 완료")


if vkospi is not None and not vkospi.empty:
    vkospi = vkospi.reset_index()
    vkospi.columns = [
        str(column).strip().lower()
        for column in vkospi.columns
    ]

    vkospi.to_csv(
        OUTPUT_DIR / "price_daily_VKOSPI.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nVKOSPI 파일 저장 완료")