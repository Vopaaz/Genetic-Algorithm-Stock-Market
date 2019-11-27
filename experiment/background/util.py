import pandas as pd


class KnowsFullTdf(object):
    def _next_day(self, day):
        day += pd.DateOffset(days=1)
        while day not in self.full_tdf.index:
            day += pd.DateOffset(days=1)
        return day
