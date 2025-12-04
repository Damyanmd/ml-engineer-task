import psycopg2

conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="localhost"
)
cur = conn.cursor()

with open('pgvector_schema.sql', 'r') as f:
    sql = f.read()
    cur.execute(sql)

conn.commit()
cur.close()
conn.close()