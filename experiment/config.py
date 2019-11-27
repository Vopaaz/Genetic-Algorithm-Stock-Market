import pandas as pd
import logging

_kwargs = {"hour": 0, "minute": 0, "second": 0, "microsecond": 0, "nanosecond": 0}

TRAIN_START = pd.Timestamp(year=2016, month=3, day=1, **_kwargs)
TRAIN_END = pd.Timestamp(year=2018, month=12, day=31, **_kwargs)

TEST_START = pd.Timestamp(year=2018, month=1, day=1, **_kwargs)
TEST_END = pd.Timestamp(year=2019, month=12, day=30, **_kwargs)

logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
