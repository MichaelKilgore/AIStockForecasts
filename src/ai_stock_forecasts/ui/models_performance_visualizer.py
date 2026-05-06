import re
from typing import Any, Dict, List

_GREEN = '\033[32m'
_RED = '\033[31m'
_RESET = '\033[0m'

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def print_models_ranking(rows: List[Dict[str, Any]]):
    if not rows:
        print('no models with completed buy/sell cycles yet')
        return

    headers = ('rank', 'model_id', 'avg_weekly_performance', 'weeks_traded')
    table = [
        (
            str(i + 1),
            str(row['model_id']),
            _color_pct(row['avg_weekly_performance']),
            str(row['weeks_traded']),
        )
        for i, row in enumerate(rows)
    ]

    _print_table(headers, table)


def print_model_weekly_performance(model_id: str, rows: List[Dict[str, Any]]):
    print(f'weekly performance for model: {model_id}')

    if not rows:
        print('  no completed buy/sell cycles yet')
        return

    headers = ('week', 'avg_weekly_pct', 'num_symbols')
    table = [
        (
            row['week'],
            _color_pct(row['avg_weekly_pct']),
            str(row['num_symbols']),
        )
        for row in rows
    ]

    _print_table(headers, table)

    overall = sum(row['avg_weekly_pct'] for row in rows) / len(rows)
    print(f'\noverall avg of weekly avgs: {_color_pct(overall)} across {len(rows)} weeks')


def print_model_last_week_symbol_performance(model_id: str, rows: List[Dict[str, Any]]):
    print(f'last week per-symbol performance for model: {model_id}')

    if not rows:
        print('  no completed buy/sell cycles yet')
        return

    headers = ('symbol', 'buy_price', 'sell_price', 'pct_change')
    table = [
        (
            row['symbol'],
            f"${row['buy_price']:.2f}",
            f"${row['sell_price']:.2f}",
            _color_pct(row['pct_change']),
        )
        for row in rows
    ]

    _print_table(headers, table)

    overall = sum(row['pct_change'] for row in rows) / len(rows)
    print(f'\navg across {len(rows)} symbols: {_color_pct(overall)}')


def _color_pct(value: float) -> str:
    text = f'{value:+.2f}%'
    color = _GREEN if value >= 0 else _RED
    return f'{color}{text}{_RESET}'


def _visible_len(s: str) -> int:
    return len(_ANSI_RE.sub('', s))


def _print_table(headers, rows):
    widths = [_visible_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _visible_len(cell))

    def fmt_row(row):
        return '  '.join(cell + ' ' * (widths[i] - _visible_len(cell)) for i, cell in enumerate(row))

    print(fmt_row(headers))
    print('  '.join('-' * w for w in widths))
    for row in rows:
        print(fmt_row(row))


if __name__ == '__main__':
    from ai_stock_forecasts.utils.postgres_util import PostgresUtil

    pg = PostgresUtil()

    ranking = pg.get_models_ranked_by_avg_weekly_performance()

    print('=== print_models_ranking ===')
    print_models_ranking(ranking)

    if ranking:
        top_model_id = ranking[0]['model_id']
        print(f'\n=== print_model_weekly_performance ({top_model_id}) ===')
        print_model_weekly_performance(top_model_id, pg.get_model_weekly_performance(top_model_id))

    target_model_id = 'ubuntu-with-even-more-recent-training'
    print(f'\n=== print_model_last_week_symbol_performance ({target_model_id}) ===')
    print_model_last_week_symbol_performance(
        target_model_id,
        pg.get_model_last_week_symbol_performance(target_model_id),
    )

