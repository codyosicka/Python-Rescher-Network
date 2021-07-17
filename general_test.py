import General 

#shillerpe_regression = General.gp_symbolic_regression(data='shillerpe_regression_data_nodates.xlsx', data_type='xlsx', separator=',', y_variable='Shiller PE Ratio by Month')
#wine_quality_regression = General.gp_symbolic_regression(data='winequality-red.csv', data_type='csv', separator=';', y_variable='quality')
#realestate_regression = General.gp_symbolic_regression('Realestate.csv','csv', ',', 'house price of unit area')
#us_infl_rate = General.gp_symbolic_regression('m2-monthly-growth-vs-infl-nodates.xlsx','xlsx', ',', 'US Inflation Rate by Month')

#General.uploadto_equations_database(result_df = shillerpe_regression)
#General.uploadto_equations_database(result_df = wine_quality_regression)
#General.uploadto_equations_database(realestate_regression)
#General.uploadto_equations_database(us_infl_rate)


connection = General.create_engine("mysql+pymysql://unwp2wrnzt46hqsp:b95S8mvE5t3CQCFoM3ci@bh10avqiwijwc8nzbszc-mysql.services.clever-cloud.com/bh10avqiwijwc8nzbszc")
table = General.pd.read_sql_query("SELECT * FROM equations_table", connection)
#print(table)

y_eq = table.loc[table['equation_name']=='Shiller PE Ratio by Month']['equation'].values.tolist()[0]
y_v = table.loc[table['equation_name']=='Shiller PE Ratio by Month']['x_variables'].str.split(",").to_list()[0]
print(y_eq)
print(y_v)
y_exp = General.sp.parsing.sympy_parser.parse_expr(y_eq)
y_otherv = list(map(str, list(y_exp.free_symbols)))
print(y_otherv)
#print(y_exp.subs(list(y_exp.free_symbols)[0], 1))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#x = General.complete_structures()
#print(x)

#wheat_array = General.np.array([[1,1,1,0,0], [0,1,0,1,1], [1,0,0,0,0], [0,0,1,0,0], [0,0,0,0,1]])
#wheat_df = General.pd.DataFrame(wheat_array, columns=['R', 'W', 'F', 'P', 'N'])

#wheat_structure_df = General.static_matrix_constructor(wheat_df)
#print(wheat_structure_df)

