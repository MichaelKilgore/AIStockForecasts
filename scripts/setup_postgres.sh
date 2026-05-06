#!/usr/bin/env bash
#
# Set up the local Postgres database and `transactions` table used to record
# buy/sell activity.
#
# Reads POSTGRES_HOST / POSTGRES_PORT / POSTGRES_USER / POSTGRES_PASSWORD /
# POSTGRES_DB from environment (or .env). Defaults target a local install on
# 127.0.0.1:5432, db `ai_stock_forecasts`.
#
# Assumes a running postgres server and that `psql` is on PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

POSTGRES_HOST="${POSTGRES_HOST:-127.0.0.1}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-ai_stock_forecasts}"

export PGPASSWORD="$POSTGRES_PASSWORD"

PSQL_BASE=(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -v ON_ERROR_STOP=1)

echo "ensuring database $POSTGRES_DB exists"
DB_EXISTS=$("${PSQL_BASE[@]}" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '$POSTGRES_DB'")
if [[ "$DB_EXISTS" != "1" ]]; then
    echo "creating database $POSTGRES_DB"
    "${PSQL_BASE[@]}" -d postgres -c "CREATE DATABASE \"$POSTGRES_DB\""
else
    echo "database $POSTGRES_DB already exists"
fi

echo "creating transactions table (if not exists)"
"${PSQL_BASE[@]}" -d "$POSTGRES_DB" <<'SQL'
CREATE TABLE IF NOT EXISTS transactions (
    id          BIGSERIAL PRIMARY KEY,
    model_id    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    price       NUMERIC(18, 6) NOT NULL,
    count       INTEGER NOT NULL,
    side        TEXT NOT NULL CHECK (side IN ('buy', 'sell'))
);

ALTER TABLE transactions
    ADD COLUMN IF NOT EXISTS model_id TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS transactions_symbol_timestamp_idx
    ON transactions (symbol, timestamp);

CREATE INDEX IF NOT EXISTS transactions_model_id_timestamp_idx
    ON transactions (model_id, timestamp);
SQL

echo "postgres setup complete"
