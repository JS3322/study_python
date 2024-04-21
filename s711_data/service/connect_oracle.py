import cx_Oracle
import os

def connect_via_wallet(wallet_path, db_tns_name):
    """
    Connect to Oracle Database using Oracle Wallet.

    Parameters:
    - wallet_path: Path to the directory containing the Oracle Wallet.
    - db_tns_name: TNS alias of the database defined in tnsnames.ora.

    Returns:
    - connection object if successful, None otherwise.
    """
    try:
        # Set the TNS_ADMIN environment variable to the directory of the wallet
        os.environ['TNS_ADMIN'] = wallet_path

        # Create a connection using the TNS entry and the Wallet for authentication
        connection = cx_Oracle.connect(dsn=db_tns_name)

        return connection
    except cx_Oracle.DatabaseError as e:
        error, = e.args
        print("Oracle-Error-Code:", error.code)
        print("Oracle-Error-Message:", error.message)
        return None

def start():
    wallet_path = '/path/to/your/oracle/wallet'
    db_tns_name = 'your_db_tns_entry'
    connection = connect_via_wallet(wallet_path, db_tns_name)

    if connection:
        print("Successfully")
        cursor = connection.cursor()

        # Example query
        cursor.execute("SELECT * FROM TEST")
        for row in cursor:
            print(row)

        cursor.close()
        connection.close()
    else:
        print("Connection failed!")

