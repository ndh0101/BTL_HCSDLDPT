import pyodbc
SERVER_NAME = r'(localdb)\MSSQLLocalDB'  # Tên Server
DATABASE_NAME = 'Bird_Metadata'    # Tên CSDL trong SQL Server
USERNAME = 'sa'
PASSWORD = '01012004'

def get_connection():
    """
    Thiết lập và trả về đối tượng kết nối đến SQL Server.
    """
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={SERVER_NAME};"
        f"DATABASE={DATABASE_NAME};"
        f"UID={USERNAME};"
        f"PWD={PASSWORD};"
    )
    try:
        conn = pyodbc.connect(conn_str)
        return conn
    except pyodbc.Error as e:
        print(f"Lỗi kết nối CSDL: {e}")
        return None

def init_db():
    """
    Kiểm tra và tạo bảng Bird_Metadata nếu chưa tồn tại trong CSDL.
    """
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            create_table_query = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Bird_Metadata' AND xtype='U')
            BEGIN
                CREATE TABLE Bird_Metadata (
                    Image_ID VARCHAR(100) PRIMARY KEY,
                    Species_Label NVARCHAR(150) NOT NULL,
                    File_Path NVARCHAR(500) NOT NULL,
                    Feature_Vector VARBINARY(MAX) NOT NULL
                )
                PRINT 'Đã tạo thành công bảng Bird_Metadata.'
            END
            ELSE
            BEGIN
                PRINT 'Bảng Bird_Metadata đã tồn tại.'
            END
            """
            cursor.execute(create_table_query)
            conn.commit()
        except pyodbc.Error as e:
            print(f"Lỗi khi thao tác với CSDL: {e}")
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    print("Đang kiểm tra kết nối SQL Server...")
    init_db()