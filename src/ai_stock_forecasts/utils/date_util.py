from datetime import datetime, timedelta
from typing import Optional
import pytz

import holidays

h = holidays.financial_holidays('NYSE')

def get_next_market_open_day(d: Optional[datetime] = None):
    if not d:
        d = datetime.now()

    while d in h or d.weekday() >= 5:
        d = d + timedelta(days=1)

    return d

def get_prev_market_open_day(d: Optional[datetime] = None):
    if not d:
        d = datetime.now()

    tz = pytz.timezone('America/New_York')

    d = tz.localize(d)

    if d.hour <= 16:
        d = d - timedelta(days=1)

    while d in h or d.weekday() >= 5:
        d = d - timedelta(days=1)

    d.replace(hour=20)

    return d



if __name__ == '__main__':
    x = get_next_market_open_day()

    print(x)









