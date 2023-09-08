import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def extract_hour(datetime: str) -> int:
    """datetime 컬럼에서 시간만 추출합니다.

    현재 datetime 정보는 'YYYYMMDD hh:mm:ss'로 되어 있습니다.
    여기에서 hh만 추출하여 정수로 반환합니다.

    Args:
        datetime (str): 시간 정보
    """

    _datetime = pd.to_datetime(datetime)
    hour = _datetime.hour

    return hour


def time_extractor(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """`extract_hour()` 함수를 `FunctionTransformer`에 사용하기 위한
    Wrapping function입니다.

    Args:
        df (pd.DataFrame): 데이터프레임
        col (str): `extract_hour()`를 적용할 컬럼명
            `datetime`만 사용해야 함

    Returns:
        pd.DataFrame: 컬럼 처리 후 데이터
    """
    df[col] = df[col].apply(lambda x: extract_hour(x))
    return df


preprocess_pipeline = ColumnTransformer(
    transformers=[
        (
            "time_extractor",
            FunctionTransformer(time_extractor, kw_args={"col": "datetime"}),
            ["datetime"],
        ),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocess_pipeline.set_output(transform="pandas")
