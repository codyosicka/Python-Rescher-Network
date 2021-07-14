import mysql.connector
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import pymysql
import os
import General

#user_name = os.environ.get('DB_USER')
#password = os.environ.get('DB_PASS')

#connection = mysql.connector.connect(
#	host = "localhost",
#	user = "root",
#	passwd = "Help181320!",
#	db = "equations_database")

#equations_conn = create_engine("mysql+pymysql://unwp2wrnzt46hqsp:b95S8mvE5t3CQCFoM3ci@bh10avqiwijwc8nzbszc-mysql.services.clever-cloud.com/bh10avqiwijwc8nzbszc")
#equations_conn.execute("CREATE TABLE IF NOT EXISTS equations_table (equation_name text, equation text, score real, x_variables text)")  

#shillerpe_regression = General.gp_symbolic_regression(data='shillerpe_regression_data_nodates.xlsx', data_type='xlsx', separator=',', y_variable='Shiller PE Ratio by Month', equation_name='shiller_pe')


def uploadto_equations_database(result_df):

  # Connect to database (or create it if it does not exist)

  equations_conn = create_engine("mysql+pymysql://root:Help181320!@localhost/equations_database")
  #equations_c = equations_conn.cursor()

  sql = "SELECT * FROM equations_table"
  read_sql = pd.read_sql(sql, equations_conn)


  if read_sql.isin([result_df['equation_name'][0]]).any().any():

    res1 = read_sql[read_sql['equation_name']==result_df['equation_name'][0]]
    previous_score = res1['score'].values[0]
    new_score = result_df['score'][0]

    if new_score > previous_score:
      #print(new_score > previous_score)
      result_df.to_sql('equations_table', equations_conn, if_exists='replace', index=False)
    else:
      pass

  else:
    result_df.to_sql('equations_table', equations_conn, if_exists='append', index=False)


  #equations_c.close()
  #equations_conn.close()

  return


#uploadto_equations_database(result_df = shillerpe_regression)

#connection = create_engine("mysql+pymysql://root:Help181320!@localhost/equations_database")
#table = pd.read_sql_query("SELECT * FROM equations_table", connection)
#print(table)