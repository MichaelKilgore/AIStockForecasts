
from abc import abstractmethod

from pandas import DataFrame
import numpy as np
from scipy.stats import norm


class BaseTradingModule:
    starting_money = 25000

    @abstractmethod
    def simulate(self, predictions: DataFrame) -> tuple:
        pass

    @abstractmethod
    def generate_buy_list(self, predictions: DataFrame) -> list[str]:
        pass

    """ assumes we can generate 5% returns annually risk free.
        sharpe_annual > 1 is considered good, but greater than 2 or even 3 is ideal.
        p_two_sided of 0.11 for example means there is an 11% chance that these results could've been generated at random.
    """
    def _calculate_sharpe_ratio_and_p_value(self, period_returns):
        rf_period = (1 + 0.05) ** (1 / len(period_returns)) - 1
        r = np.array(period_returns)
        excess = r - rf_period

        sharpe_daily = excess.mean() / excess.std(ddof=1)
        sharpe_annual = sharpe_daily * np.sqrt(len(period_returns))

        z = sharpe_daily * np.sqrt(len(period_returns))
        p_two_sided = 2 * norm.sf(abs(z))

        return sharpe_annual, p_two_sided


