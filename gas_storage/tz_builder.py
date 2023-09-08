import pytz
import datetime
import pandas as pd
import itertools

class CurveScalar:

    def __init__(self, start, end, timezone='Europe/London'):
        self.timezone = pytz.timezone(timezone)

        self.start, self.end = start, end
        self.start_dt, self.end_dt = self.to_dt(start), self.to_dt(end)

        self.date_range = self.create_date_range()

    @staticmethod
    def to_dt(date_str):
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")

    def utc_to_local(self, utc_dt):
        return utc_dt.replace(tzinfo=pytz.utc).astimezone(self.timezone)

    def create_date_range(self):
        dr = pd.date_range(self.start_dt, self.end_dt, freq='h')
        return [self.utc_to_local(i) for i in dr]


if __name__ == '__main__':
    # Example usage with hourly frequency:
    start_date = "2023-03-25"
    end_date = "2023-10-29"

    power_curve = CurveScalar(start_date, end_date)
    power_curve.create_date_range()  # Hourly
