from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2.extras import RealDictCursor
import json

from mcp.server.lowlevel import  server

# 创建 MCP 服务器
mcp = FastMCP("PostgreSQL Server",
              debug=True,
              host="0.0.0.0",
              port=8002)

# 数据库连接配置
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "123456",
    "host": "localhost",
    "port": "5432"
}

# 获取数据库连接对象
def get_db_connection():
    """创建数据库连接"""
    return psycopg2.connect(**DB_CONFIG)


# 定义资源：获取所有表名
@mcp.resource("db://tables")
def list_tables() -> str:
    """获取所有表名列表"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cur.fetchall()]
            return json.dumps(tables)


# 定义资源：获取表数据
@mcp.resource("db://tables/{table_name}/data/{limit}")
def get_table_data(table_name: str, limit: int = 100) -> str:
    """获取指定表的数据

    参数:
    table_name: 表名
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 使用参数化查询防止 SQL 注入
            cur.execute(f"SELECT * FROM %s LIMIT %s",
                        (psycopg2.extensions.AsIs(table_name), limit))
            rows = cur.fetchall()
            return json.dumps(list(rows), default=str, ensure_ascii=False)


# 定义资源：获取表结构
@mcp.resource("db://tables/{table_name}/schema")
def get_table_schema(table_name: str) -> str:
    """获取表结构信息

    参数:
    table_name: 表名
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                select c.column_name, 
                       c.data_type, 
                       c.character_maximum_length,
                       pgd.description as column_comment
                from information_schema.columns c
                left join pg_catalog.pg_statio_all_tables st 
                on c.table_schema = st.schemaname and c.table_name = st.relname
                left join pg_catalog.pg_description pgd 
                on pgd.objoid = st.relid 
                   and pgd.objsubid = c.ordinal_position
                where c.table_name = %s
                order by c.ordinal_position
            """, (table_name,))
            columns = [{"name": row[0], "type": row[1], "max_length": row[2]}
                       for row in cur.fetchall()]
            return json.dumps(columns)

# 中国省份介绍
@mcp.prompt()
def introduce_china_province(province: str) -> str:
    """介绍中国省份

    参数:
    province: 省份名称
    """
    return f"""
    请介绍这个省份：{province}

    要求介绍以下内容：
    1. 历史沿革
    2. 人文地理、风俗习惯
    3. 经济发展状况
    4. 旅游建议
    """

@mcp.tool()
def add(a: float, b: float) -> float:
    """加法运算

    参数:
    a: 第一个数字
    b: 第二个数字

    返回:
    两数之和
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """减法运算

    参数:
    a: 第一个数字
    b: 第二个数字

    返回:
    两数之差 (a - b)
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """乘法运算

    参数:
    a: 第一个数字
    b: 第二个数字

    返回:
    两数之积
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """除法运算

    参数:
    a: 被除数
    b: 除数

    返回:
    两数之商 (a / b)

    异常:
    ValueError: 当除数为零时
    """
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


if __name__ == "__main__":
    # 本地测试
    mcp.run("stdio")
