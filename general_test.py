from General import General
General.pd.set_option('display.max_columns', None)
General.pd.set_option('display.max_rows', None)

#shillerpe_regression = General.gp_symbolic_regression(data='shillerpe_regression_data_nodates.xlsx', y_variable='Shiller PE Ratio by Month')
#wine_quality_regression = General.gp_symbolic_regression(data='winequality-red.csv', y_variable='quality')
#realestate_regression = General.gp_symbolic_regression(data='Realestate.csv', y_variable='house price of unit area')
#us_infl_rate = General.gp_symbolic_regression(data='m2-monthly-growth-vs-infl-nodates.xlsx', y_variable='US Inflation Rate by Month')


#General.uploadto_equations_database(result_df = shillerpe_regression)
#General.uploadto_equations_database(result_df = wine_quality_regression)
#General.uploadto_equations_database(realestate_regression)
#General.uploadto_equations_database(us_infl_rate)


#connection = General.create_engine("mysql+pymysql://unwp2wrnzt46hqsp:b95S8mvE5t3CQCFoM3ci@bh10avqiwijwc8nzbszc-mysql.services.clever-cloud.com/bh10avqiwijwc8nzbszc")
#table = General.pd.read_sql_query("SELECT * FROM equations_table", connection)
#print(table)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#graph = General.nx.read_gexf('G_causal_network.gexf')

#print(graph.adj)
#print(len(graph.adj))

#General.plt.figure(1)
#General.nx.draw_planar(graph,
#                node_color='red', # draw planar means the nodes and edges are drawn such that not edges cross
#                arrows=True, with_labels=True)
#General.plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

x = General.complete_structures()
print(x)

#wheat_array = General.np.array([[1,1,1,0,0], [0,1,0,1,1], [1,0,0,0,0], [0,0,1,0,0], [0,0,0,0,1]])
#wheat_df = General.pd.DataFrame(wheat_array, columns=['R', 'W', 'F', 'P', 'N'])

#wheat_structure_df = General.static_matrix_constructor(wheat_df)
#print(wheat_structure_df)

