import os
import psycopg
from psycopg.rows import dict_row

def get_conn():
    dsn = os.environ["DATABASE_URL"]
    return psycopg.connect(dsn, row_factory=dict_row)
