import logging

import pandas as pd

_kwargs = {"hour": 0, "minute": 0, "second": 0, "microsecond": 0, "nanosecond": 0}

TRAIN_START = pd.Timestamp(year=2016, month=3, day=1, **_kwargs)
TRAIN_END = pd.Timestamp(year=2017, month=12, day=29, **_kwargs)

TEST_START = pd.Timestamp(year=2018, month=1, day=2, **_kwargs)
TEST_END = pd.Timestamp(year=2018, month=12, day=28, **_kwargs)

logging.basicConfig(format="%(asctime)s %(levelname)s | %(message)s", datefmt="%m/%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CORES = 4
