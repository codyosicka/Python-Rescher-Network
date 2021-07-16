import pandas as pd
import io
import os

import gplearn
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import itertools
import math
import pandas_datareader as web

import mysql.connector
import pymysql
from sqlalchemy import create_engine
import sqlalchemy
from sqlalchemy import Integer, String, Float
import time
import datetime
from datetime import timedelta
import random
import matplotlib.dates as mdates
from matplotlib import style
style.use('fivethirtyeight')

import sympy as sp
from sympy import sympify
from sympy import log, sin, cos, tan, Mul, Add, Max, Min, sqrt, Abs, exp
from sympy import symbols

from scipy.optimize import minimize

import networkx as nx

import mysql.connector
import databases
from databases import Database

import fuzzywuzzy
from fuzzywuzzy import fuzz, process

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)




# FINALIZED GP: SYMBOLIC REGRESSION:

def gp_symbolic_regression(data, data_type, separator, y_variable):

  if data_type=='csv':
    df = pd.read_csv(data, sep=separator)
    df = df.dropna(how = 'any')
    df = df.astype(float)
  elif data_type=='xlsx':
    df = pd.read_excel(data)
    df = df.dropna(how = 'any')
    df = df.astype(float)

  y_variable_df = df[y_variable] 
  x_variables_df = df.drop(columns=y_variable)
  number_of_variables = len(x_variables_df.columns)
  x_variables_list = x_variables_df.columns.values.tolist()
  x_variables_str = ','.join(x_variables_list)
  x_variables_input = ' '.join(x_variables_list)

  # create a numpy array of all the variables and their values
  y_variable_array = y_variable_df.to_numpy()
  x_variables_array = x_variables_df.to_numpy()

  # Training samples
  X_train = x_variables_array[:round(len(x_variables_array)/2)]
  y_train = y_variable_array[:round(len(y_variable_array)/2)]

  # Testing samples
  X_test = x_variables_array[round(len(x_variables_array)/2):]
  y_test = y_variable_array[round(len(y_variable_array)/2):]


  # Symbolic Regression:

  # ,'sqrt','log','abs','inv','max','min','sin','cos'

  est_gp = SymbolicRegressor(population_size=7000, #the number of programs in each generation
                            generations=50, stopping_criteria=0.01, #The required metric value required in order to stop evolution early.
                            p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, #0.05, The probability of performing hoist mutation on a tournament winner. Hoist mutation takes the winner of a tournament and selects a random subtree from it. A random subtree of that subtree is then selected and this is ‘hoisted’ into the original subtrees location to form an offspring in the next generation. This method helps to control bloat.
                            p_point_mutation=0.1,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0.01, random_state=0, function_set=('add','sub','mul','div','log','sqrt','sin','cos','max','min','tan','abs','neg','inv'), 
                            warm_start=False, tournament_size=20)
  est_gp.fit(X_train, y_train)

  # Results:

  # These are other methods of estimating the equation using the data

  est_tree = DecisionTreeRegressor()
  est_tree.fit(X_train, y_train)
  est_rf = RandomForestRegressor(n_estimators=10)
  est_rf.fit(X_train, y_train)


  # Here is how the methods compare to the genetic programming Symbolic Regressor

  score_gp = est_gp.score(X_test, y_test)
  score_tree = est_tree.score(X_test, y_test)
  score_rf = est_rf.score(X_test, y_test)


  # Input equation into a text file

  equation = est_gp
  #text = "{}".format(equation)
  equation_file = open("{}.txt".format(y_variable), "w") # open for writing, truncating the file first
  print(equation, file=equation_file)
  equation_file.close()

  equation_df = pd.read_csv("{}.txt".format(y_variable), delimiter = "\t", header=None)
  equation_to_string = equation_df.to_string(index=False, header=False)
  os.remove("{}.txt".format(y_variable))

  # sympify
  #symbols_ = symbols(x_variables_input)
  locals = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x / y,
    'mul': Mul,
    'add': Add,
    'log': log,
    'sin': sin,
    'cos': cos,
    'sqrt': sqrt,
    'max': Max,
    'min': Min,
    'abs': Abs,
    #'inv': invert,
    'tan': tan,
    'neg': lambda x    : -x,
    'pow': lambda x, y : x**y,
    'exp': exp
    }
  equation_string = sympify(equation_to_string, locals=locals)
  #equation_string = sympify(equation_string, symbols_)
  equation_df = pd.DataFrame(data=[equation_string], dtype='string')

  x_variables_str_df = pd.DataFrame(data=[x_variables_str], dtype='string')
  name_df = pd.DataFrame(data=[y_variable], columns=['equation_name'], dtype='string')
  score_df = pd.DataFrame(data=[score_gp], columns=['score'], dtype='float')
  result_df = pd.concat([name_df, equation_df, score_df, x_variables_str_df], axis=1)
  result_df.columns = ['equation_name', 'equation', 'score', 'x_variables']

  return result_df#, score_gp, score_tree, score_rf



# Store the equations into an SQL database


def uploadto_equations_database(result_df):

  #equations_conn = create_engine("mysql+pymysql://root:Help181320!@localhost/equations_database")
  equations_conn = create_engine("mysql+pymysql://unwp2wrnzt46hqsp:b95S8mvE5t3CQCFoM3ci@bh10avqiwijwc8nzbszc-mysql.services.clever-cloud.com/bh10avqiwijwc8nzbszc")
  #equations_conn.execute("CREATE TABLE IF NOT EXISTS equations_table (eq_id int, equation_name text, equation text, score real, x_variables text)")

  #sql = "SELECT * FROM equations_table"
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



def complete_structures():
  
  equations_conn = create_engine("mysql+pymysql://unwp2wrnzt46hqsp:b95S8mvE5t3CQCFoM3ci@bh10avqiwijwc8nzbszc-mysql.services.clever-cloud.com/bh10avqiwijwc8nzbszc")

  sql = "SELECT * FROM equations_table"
  read_sql = pd.read_sql(sql, equations_conn)

  all_variables_df = pd.DataFrame()

  all_variables_df['all_variables'] = read_sql['equation_name'].str.cat(read_sql['x_variables'], sep=",")

  all_variables_dict = {}
  for i in range(len(read_sql)):
    all_variables_dict[i] = read_sql.loc[i]['x_variables'].split(",")
    all_variables_dict[i].append(read_sql.loc[i]['equation_name'])
    i+=1
  all_variables_list = []
  for j in range(len(all_variables_dict)):
    all_variables_list.append(all_variables_dict[j])
    j+=1
  all_variables_list = [var for sublist in all_variables_list for var in sublist]

  def find_matches(variable):
    matches = all_variables_df.apply(lambda row: (fuzz.partial_ratio(row['all_variables'], variable) == 100), axis=1)
    return [k for k, x in enumerate(matches) if x]
  
  for v_num in range(len(all_variables_list)):
    matches_list = []
    matches_list.append(find_matches(all_variables_list[v_num]))
    v_num+=1

  matches_array = np.array(matches_list)
  matches_series = pd.Series(list(matches_array)) # this is a pandas Series of all the matches, indexed, so they may be accessed easily


  # Now create the structures:
  #   definition: A structure is a set of m functions involving n variables (where n >= m), such that:
        # (a) In any subset k function of the structure, at least k different variables appear.
        # (b) In any subset of k function in which r (r >= k) variables appear, if the values of any (r-k) variables are chosen arbitrarily,
        # then the values of the remaining k variables are determined uniquely. (Finding these unique values is a matter of solving the equations for them.)

  structures_dict = {}
  for match in range(len(matches_series)):
    col_list = []
    for eq in range(len(matches_series[match])):
      structures_dict[match] = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in all_variables_dict.items() ])).transpose().drop(index=matches_series[match])
      structures_dict[match] = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in all_variables_dict.items() ])).transpose().drop(index=structures_dict[match].index).dropna(axis='columns', how='all')
      structures_dict[match].index = list(range(len(matches_series[match])))
      col_list.append(structures_dict[match].loc[eq].dropna().values.tolist())

      eq+=1
    match+=1
  
  col_list = [item for sublist in col_list for item in sublist]
  for match in range(len(matches_series)):
    if len(col_list) > len(structures_dict[match].columns):
      for newcol in range(len(col_list) - len(structures_dict[match].columns)):
        structures_dict[match]["newcol{}".format(newcol)] = np.nan
        newcol+=1

      structures_dict[match].columns = col_list
      structures_dict[match] = structures_dict[match].loc[:, ~structures_dict[match].columns.duplicated()]
    match+=1

  for key in range(len(structures_dict)):
    check_list = []
    for index in range(len(structures_dict[key])):
      check_list.append(structures_dict[key].columns.isin(structures_dict[key].loc[index].values.tolist()) * 1.0)
      structures_dict[key].loc[index] = check_list[index]
      index+=1
    key+=1


  return structures_dict



# Apply Rescher Causal Ordering:

# Define complete structures:


def static_matrix_constructor(df): # need to modify this to incorporate it into model building

  # Step 1: create static structure and functions from given variables and functions

  # Requires that the structure is self-contained (number of functions = number of variables)

  variables = df.columns.to_list()

  j = 0
  functions=[]
  while j < len(variables):
    functions.append('f'+'('+variables[j]+')')
    j += 1

  structure_matrix_array = df.to_numpy()
  variables = np.array(variables)
  functions = np.array(functions)

  structure_matrix_df = pd.DataFrame(structure_matrix_array, index=functions, columns=variables)

  return structure_matrix_df


def static_self_contained_causal_structure(structure_matrix_df):
  # Now we reduce the matrix to determine causal ordering

  # Step 2: reduce the matrix by eliminating functions with only one variable (self-contained structures)

  list_v = structure_matrix_df.columns.to_list() # sort the dataframe by values
  for v in range(len(list_v)):
    structure_matrix_df = structure_matrix_df.sort_values(by=list_v[v], ascending=True)
    v += 1
  
  df_reduced_first_order = structure_matrix_df.drop(index=structure_matrix_df[structure_matrix_df.sum(axis=1) != 1].index, columns=structure_matrix_df[structure_matrix_df.sum(axis=1) != 1])

  equations_dict = {}
  equations_df = pd.DataFrame()
  for f in range(len(structure_matrix_df)):
    equations_dict[f] = structure_matrix_df[f:f+1]
    equations_dict[f] = equations_dict[f].loc[:, (equations_dict[f] != 0).any(axis=0)]
    #equations_dict[f] = equations_dict[f] * equations_dict[f].columns

    f += 1
    list_f = structure_matrix_df.index.to_list()
    equations_dict = dict(zip(list_f, list(equations_dict.values())))

  list_reduced_first_order_functions = df_reduced_first_order.index.to_list()

  list_reduced_first_order_variables = []
  for h in list_reduced_first_order_functions:
    list_reduced_first_order_variables.append(equations_dict[h].columns.to_list())

  list_reduced_first_order_variables = [i[0] for i in list_reduced_first_order_variables]

  derived_structure_first_order = structure_matrix_df.drop(index=list_reduced_first_order_functions, columns=list_reduced_first_order_variables)


  # Step 3: reduce the matrix by eliminating pairs of functions with the same variables (self-contained structures) and can be solved through a system of equations that are a subset of the first derived matrix

  equations_reduced_first_dict = {}
  equations_reduced_first_df = pd.DataFrame()
  for f in range(len(derived_structure_first_order)):
    equations_reduced_first_dict[f] = derived_structure_first_order[f:f+1]
    equations_reduced_first_dict[f] = equations_reduced_first_dict[f].loc[:, (equations_reduced_first_dict[f] != 0).any(axis=0)]
    equations_reduced_first_dict[f] = equations_reduced_first_dict[f] * equations_reduced_first_dict[f].columns

    f += 1
    list_f2 = derived_structure_first_order.index.to_list()
    equations_reduced_first_dict = dict(zip(list_f2, list(equations_reduced_first_dict.values())))

  pairs_kept = derived_structure_first_order.drop_duplicates(keep=False)
  dropped_pairs = derived_structure_first_order.drop(index=pairs_kept.index)
  dropped_pairs = dropped_pairs.loc[:, (dropped_pairs != 0).any(axis=0)]

  list_reduced_second_order_functions = dropped_pairs.index.to_list()
  list_reduced_second_order_functions = list(filter(lambda x: x, list_reduced_second_order_functions))
  list_reduced_second_order_variables = dropped_pairs.columns.to_list()
  list_reduced_second_order_variables = list(filter(lambda x: x, list_reduced_second_order_variables))

  list_reduced_first_order_functions = list(filter(lambda x: x, list_reduced_first_order_functions))
  list_reduced_first_order_variables = list(filter(lambda x: x, list_reduced_first_order_variables))

  derived_structure_second_order = derived_structure_first_order.drop(index=list_reduced_second_order_functions, columns=list_reduced_second_order_variables)

  # Next, repeat the process until all functions are eliminated

  n = len(derived_structure_second_order.index)
  if n == 0:
    return derived_structure_second_order, list_reduced_first_order_functions, list_reduced_first_order_variables, list_reduced_second_order_functions, list_reduced_second_order_variables
  else:
    while n > 0:
      return list_reduced_first_order_functions, list_reduced_first_order_variables, list_reduced_second_order_functions, list_reduced_second_order_variables, static_self_contained_causal_structure(derived_structure_second_order)
      if n == 0:
        break

# Now, causal order may be determined

def static_causal_order(original_df):

  equations_dict = {}
  equations_df = pd.DataFrame()
  for f in range(len(original_df)):
    equations_dict[f] = original_df[f:f+1]
    equations_dict[f] = equations_dict[f].loc[:, (equations_dict[f] != 0).any(axis=0)]
    equations_dict[f] = equations_dict[f] * equations_dict[f].columns

    f += 1
    list_f = original_df.index.to_list()
    equations_dict = dict(zip(list_f, list(equations_dict.values())))

  def flatten(l, ltypes=(list, tuple)): # http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

  # the static_self_contained_causal_structure(original_df) produces a tuple that needs to be flattened
  merged = list(flatten(static_self_contained_causal_structure(original_df)[-1], tuple)) 
  merged2 = list(flatten(static_self_contained_causal_structure(original_df)[:-1], tuple))
  merged_final = merged2
  for i in merged:
    merged_final.append(i)
  if len(merged_final) <= 4:
    merged_final.append([]) # this is to replace the [] that was removed
  merged_final.pop(-5) # this removes the empty dataframe from any len(merged_final)

  merged_final_index = list(range(len(merged_final)))

  def splitevenodd(A):
    evenlist = [] 
    oddlist = [] 
    for i in A: 
      if (i % 2 == 0):
        evenlist.append(i) 
      else: 
        oddlist.append(i)
    return evenlist, oddlist
  
  functions_are_even = splitevenodd(merged_final_index)[0]
  variables_are_odd = splitevenodd(merged_final_index)[1]

  static_functions_eliminated = []
  for i in functions_are_even:
    static_functions_eliminated.append(merged_final[i])
  static_functions_eliminated = [x for x in static_functions_eliminated if x]
  
  static_variables_eliminated = []
  for i in variables_are_odd:
    static_variables_eliminated.append(merged_final[i])
  static_variables_eliminated = [x for x in static_variables_eliminated if x]

  function_layers_dict = {}
  for g in range(len(static_functions_eliminated)):
    function_layers_dict[g] = static_functions_eliminated[g]
    g += 1

  variable_layers_dict = {}
  for e in range(len(static_variables_eliminated)):
    variable_layers_dict[e] = static_variables_eliminated[e]
    e += 1

  causal_order_dict = {}
  for v in range(len(static_variables_eliminated)):
    causal_order_dict[v] = {}
    for a in range(len(variable_layers_dict[v])):
      causal_order_dict[v][a] = pd.DataFrame()
      causal_order_dict[v][a] = pd.read_csv(io.StringIO(variable_layers_dict[v][a]), header=None) # causal_order_dict[0][0] is the dataframe
      variable_layers_dict[v] = pd.Series(variable_layers_dict[v])
  
      if v > 0 and v < len(static_variables_eliminated):
        
        equations_dict[static_functions_eliminated[v][a]] = equations_dict[static_functions_eliminated[v][a]].drop(columns=causal_order_dict[v][a][0])
        equations_dict[static_functions_eliminated[v][a]] = equations_dict[static_functions_eliminated[v][a]].reset_index(drop=True).transpose().reset_index(drop=True)
        causal_order_dict[v][a] = pd.concat([equations_dict[static_functions_eliminated[v][a]], causal_order_dict[v][a]], ignore_index=True, axis=1).fillna(0)

      a += 1
    v += 1

  return causal_order_dict






# CAUSAL NETWORK GRAPH:


# Initialize mini network:

# initialize_causal_network should only be called once EVER and NEVER again
def initialize_mini_network(original_df, sco, name_of_network): # name_of_network is string, # sco is the result from static_causal_order function
  
  for i in range(len(sco)):
    for j in range(len(sco[i])):
      if i > 0:
        sco[i][j][1] = sco[i][j][1][0] # fill the 'caused' variable column with the 'caused' variable, this is so "causal pairs" can be easily made
        sco[i][j] = sco[i][j].values.tolist()
    i+=1

  edges_dict = {}
  for i in range(len(sco)):
    if i > 0:
      edges_dict[i] = sco[i]
    i+=1

  def flatten_dict(pyobj, keystring=''): # https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/
    if type(pyobj) == dict:
        keystring = keystring + '_' if keystring else keystring
        for k in pyobj:
            yield from flatten_dict(pyobj[k], keystring + str(k))
    else:
        yield keystring, pyobj

  flat_edges_dict = { k:v for k,v in flatten_dict(edges_dict) }
  key_list = list(range(len(flat_edges_dict)))
  final_dict = dict(zip(key_list, list(flat_edges_dict.values())))

  for key in range(len(final_dict)):
    final_dict[key] = [tuple(l) for l in final_dict[key]]
    key+=1

  static_functions = original_df.index.values.tolist()
  static_variables = original_df.columns.values.tolist()

  # Create an empty directed graph
  G_causal_variables = nx.DiGraph() # nx.DiGraph for directed graph (graph that has nodes that point in a direction), nx.Graph for undirected graph
  G_causal_variables.add_nodes_from(static_variables)

  for key in range(len(final_dict)):
    G_causal_variables.add_edges_from(final_dict[key]) # from file for large graphs

  nx.write_gexf(G_causal_variables, "C:\\Users\\Xaos\\Desktop\\Python\\causal_networks_folder\\initialized_causal_network_{}.gexf".format(name_of_network))

  return G_causal_variables



# Modifying a mini network:

def modify_mini_network(original_df, sco, previous_causal_network, name_of_network): # previous_causal_network is the title of the existing causal newtork as a 'string.gexf'

  for i in range(len(sco)):
    for j in range(len(sco[i])):
      if i > 0:
        sco[i][j][1] = sco[i][j][1][0] # fill the 'caused' variable column with the 'caused' variable, this is so "causal pairs" can be easily made
        sco[i][j] = sco[i][j].values.tolist()
    i+=1

  edges_dict = {}
  for i in range(len(sco)):
    if i > 0:
      edges_dict[i] = sco[i]
    i+=1

  def flatten_dict(pyobj, keystring=''): # https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/
    if type(pyobj) == dict:
        keystring = keystring + '_' if keystring else keystring
        for k in pyobj:
            yield from flatten_dict(pyobj[k], keystring + str(k))
    else:
        yield keystring, pyobj

  flat_edges_dict = { k:v for k,v in flatten_dict(edges_dict) }
  key_list = list(range(len(flat_edges_dict)))
  final_dict = dict(zip(key_list, list(flat_edges_dict.values())))

  for key in range(len(final_dict)):
    final_dict[key] = [tuple(l) for l in final_dict[key]]
    key+=1

  static_functions = original_df.index.values.tolist()
  static_variables = original_df.columns.values.tolist()

  # Save a copy of the old version in case a mistake is made in the current modification
  G_causal_variables_old = nx.read_gexf(previous_causal_network)
  nx.write_gexf(G_causal_variables_old, "old_causal_network.gexf")

  # Now modify the old version to make the new one
  G_causal_variables_new = nx.read_gexf(previous_causal_network)
  G_causal_variables_new.add_nodes_from(static_variables)

  for key in range(len(final_dict)):
    G_causal_variables_new.add_edges_from(final_dict[key]) # from file for large graphs

  nx.write_gexf(G_causal_variables_new, "C:\\Users\\Xaos\\Desktop\\Python\\causal_networks_folder\\modified_causal_network_{}.gexf".format(name_of_network))


  return G_causal_variables_new



# Now construct the causal network graph from the mini networks


def build_causal_network():
  
  #folder_path = r'C:\\Users\\Xaos\\Desktop\\Python\\causal_networks_folder'
  folder_path = r'C:\\Users\\Buff14\\Desktop\\Python\\causal_networks_folder'


  def listDir(dir):
    fileNames = os.listdir(dir)
    files_list = []
    for fileName in fileNames:
      files_list.append(fileName)
      #print('File Name: ' + fileName)
      #print('Folder Path: ' + os.path.abspath(os.path.join(dir, fileName)), sep='\n')
    return files_list

  #if __name__ == '__main__':
    #mini_networks_list = listDir(folder_path)

  mini_networks_list = listDir(folder_path)

  causal_network_nodes_list0 = []
  causal_network_edges_list0 = []
  for i in range(len(mini_networks_list)):
    causal_network_nodes_list0 = []
    causal_network_nodes_list0.append(list(nx.read_gexf(folder_path + '\\{}'.format(mini_networks_list[i])).nodes()))
    causal_network_edges_list0.append(list(nx.read_gexf(folder_path + '\\{}'.format(mini_networks_list[i])).edges()))
    i += 1

  causal_network_nodes_list = [node for listnodes in causal_network_nodes_list0 for node in listnodes]
  causal_network_edges_list = [edge for listedges in causal_network_edges_list0 for edge in listedges]

  G_causal_network = nx.DiGraph()
  G_causal_network.add_nodes_from(causal_network_nodes_list)
  G_causal_network.add_edges_from(causal_network_edges_list)

  #nx.write_gexf(G_causal_network, "C:\\Users\\Xaos\\Desktop\\Python\\G_causal_network.gexf")
  nx.write_gexf(G_causal_network, "C:\\Users\\Buff14\\Desktop\\Python\\causal_networks_folder")


  return G_causal_network


#print(build_causal_network())


#plt.figure(1)
#nx.draw_planar(initialize_causal_network(),
#                node_color='red', # draw planar means the nodes and edges are drawn such that not edges cross
#                #node_size=size_map, 
#                arrows=True, with_labels=True)
#plt.show()



def simulator(equation_name, variables):
  pass
  return



def optimizer(equation_name, objective, constraints, initial_condition): # objective is either min or max; constraints is a list; initial condition is a guess for values of variables
  
  equations_conn = create_engine("mysql+pymysql://unwp2wrnzt46hqsp:b95S8mvE5t3CQCFoM3ci@bh10avqiwijwc8nzbszc-mysql.services.clever-cloud.com/bh10avqiwijwc8nzbszc")

  sql = "SELECT * FROM equations_table"
  read_sql = pd.read_sql(sql, equations_conn)

  selected_eq = read_sql.loc[read_sql['equation_name']==equation_name]['equation'].values.tolist()[0]
  selected_variables = read_sql.loc[read_sql['equation_name']=='quality']['x_variables'].str.split(",").to_list()[0]


  #x_str = 'X2/0.5*X10+2*X9-X1'
  #x_exp = parse_expr(x_str)
  #x_v = list(x_exp.free_symbols) # list({some expression}.free_symbols) yeilds a list of all variables in the equations in descending (by number) order
  #print(x_exp)
  #print(x_v)
  #print(x_exp.subs({x_v[0]: 1, x_v[1]: 1, x_v[2]: 1, x_v[3]: 1}))


  return



print("General is DONE!")