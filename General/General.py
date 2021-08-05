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
import csv

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)




# FINALIZED GP: SYMBOLIC REGRESSION:

def gp_symbolic_regression(data, y_variable):

  if 'csv' in data:
    data_type = 'csv'
    with open(data, newline='') as csvfile:
      sniffer = csv.Sniffer()
      dialect = sniffer.sniff(csvfile.read(1024))
      separator = dialect.delimiter
  elif 'xlsx' in data:
    data_type = 'xlsx'
    separator = ''

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

  # WILL HAVE TO FIX THE PROBLEM OF NOT ALL OF THE VARIABLES LISTED IN THE CSV AND EXCEL FILES WILL SHOW UP IN THE EQUATIONS
  # SO IT DOES NOT MAKE SENSE THAT THE MISSING VARIABLES ARE INCLUDED IN THE complete_structures FUNCTIONS AND SHOW UP IN THE CAUSAL NETWORK AFTERWARDS
  
  equations_conn = create_engine("mysql+pymysql://unwp2wrnzt46hqsp:b95S8mvE5t3CQCFoM3ci@bh10avqiwijwc8nzbszc-mysql.services.clever-cloud.com/bh10avqiwijwc8nzbszc")

  sql = "SELECT * FROM equations_table"
  read_sql = pd.read_sql(sql, equations_conn)

  # Select and organize all of the variables for each equation
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

  # remove duplicates from all_variables_list
  seen = set()
  result = []
  for item in all_variables_list:
    if item not in seen:
      seen.add(item)
      result.append(item)
  all_variables_list = result

  series = all_variables_df['all_variables']
  all_variables_series = series.str.split(pat=',')
  all_variables_df_split = pd.DataFrame(item for item in all_variables_series)
  all_df = pd.DataFrame(all_variables_list)

  # Determine which variables appear in the equations
  expression_series = sympify(table['equation'])
  symbols_series = expression_series.apply(lambda x: list(x.free_symbols))
  symbols_series = symbols_series.apply(lambda x: list(map(str, x)))
  for x in range(len(symbols_series.index)):
    for y in range(len(symbols_series[x])):
      symbols_series[x][y] = symbols_series[x][y].replace('X','')
  symbols_series = symbols_series.apply(lambda x: list(map(int, x)))
  symbols_df = pd.DataFrame(item for item in symbols_series)

  for column in range(len(symbols_df.columns)):
    for index in range(len(symbols_df.index)):
      if np.isnan(symbols_df[column].loc[index]):
        continue
      else:
        symbols_df[column].loc[index] = all_variables_df_split[symbols_df[column].loc[index]].loc[index]

  def find_matches(variable_name):
    lst = []
    #for i in range(len(all_variables_df_split.index.values.tolist())):
    for i in range(len(symbols_df.index.values.tolist())):
      #if all_variables_df_split.loc[i].isin([variable_name]).any().any() == True:
      if symbols_df.loc[i].isin([variable_name]).any().any() == True:
        lst.append(i)
      i+=1
    return lst

  matches_list = []
  for j in range(len(all_variables_list)):
    matches_list.append(find_matches(all_variables_list[j]))
    j+=1
  all_df[1] = matches_list

  matches_df = all_df[1]
  for x in range(len(matches_df.index.values.tolist())):
    if len(matches_df[x]) < 2:
      matches_df = matches_df.drop(labels=[x])
    x+=1

  matches_series = matches_df.reset_index().drop(columns=['index'])[1]
  matches_df = matches_series.to_frame()
  matches_df = pd.DataFrame(np.unique(matches_df), columns=matches_df.columns) # remove duplicates
  matches_df.columns = [0]
  matches_series = matches_df[0] # this is a pandas Series of all the matches, indexed, so they may be accessed easily

  list_of_matches = []
  for i in range(len(matches_series)):
    list_of_matches.append(matches_series[i])

  transform0 = []
  for i in list_of_matches:
    transform0.append(set(i))

  transform1 = []
  for i in range(len(transform0)):
    for j in range(i+1, len(transform0)):
      i_set = transform0[i]
      j_set = transform0[j]
      if (i_set & j_set):
        comb = list(set(list(i_set) + list(j_set)))
        transform1.append(comb)

  # Now create the structures:
  #   definition: A structure is a set of m functions involving n variables (where n >= m), such that:
        # (a) In any subset k function of the structure, at least k different variables appear.
        # (b) In any subset of k function in which r (r >= k) variables appear, if the values of any (r-k) variables are chosen arbitrarily,
        # then the values of the remaining k variables are determined uniquely. (Finding these unique values is a matter of solving the equations for them.)

  # preping the structures_dict to look like the 1s and 0s representation of structures

  structures_dict = {}
  col_list = []
  for match in range(len(transform1)):
    for eq in range(len(transform1[match])):
      structures_dict[match] = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in all_variables_dict.items() ])).transpose().drop(index=transform1[match])
      structures_dict[match] = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in all_variables_dict.items() ])).transpose().drop(index=structures_dict[match].index).dropna(axis='columns', how='all')
      structures_dict[match].index = list(range(len(transform1[match])))
      col_list.append(structures_dict[match].loc[eq].dropna().values.tolist())
      eq+=1
    match+=1
  
  col_list = [item for sublist in col_list for item in sublist] # flatten the list
  col_list = list(set(col_list)) # remove duplicates
  for match in range(len(transform1)):
    if len(col_list) > len(structures_dict[match].columns):
      for newcol in range(len(col_list) - len(structures_dict[match].columns)):
        structures_dict[match]["newcol{}".format(newcol)] = np.nan
        newcol+=1
      structures_dict[match].columns = col_list
      structures_dict[match] = structures_dict[match].loc[:, ~structures_dict[match].columns.duplicated()]
    match+=1

  check_list = []
  for key in range(len(structures_dict)):
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


# the simulator needs the User to choose an equation and input the values for its component variables and choose a target variable
# then, the simulator needs to create static values for the effected equations to simulate the effects of the User inputs and assumptions on the target variable
def simulator(equation_name, variable_values, target_variable): # User chooses equation_name from menu and inputs variable_values (will be a dictionary on my end)
  
  equations_conn = create_engine("mysql+pymysql://unwp2wrnzt46hqsp:b95S8mvE5t3CQCFoM3ci@bh10avqiwijwc8nzbszc-mysql.services.clever-cloud.com/bh10avqiwijwc8nzbszc")

  sql = "SELECT * FROM equations_table"
  read_sql = pd.read_sql(sql, equations_conn)

  whole_graph = nx.read_gexf('G_causal_network.gexf')
  node_connections = whole_graph.adj

  for node in range(len(node_connections)):
    pass


  #x_str = 'X2/0.5*X10+2*X9-X1'
  #x_exp = parse_expr(x_str)
  #x_v = list(x_exp.free_symbols) 
  #print(x_exp)
  #print(x_v)
  #print(x_exp.subs({x_v[0]: 1, x_v[1]: 1, x_v[2]: 1, x_v[3]: 1}))

  pass
  return



# on the website make each of the inputs for this optimizer function a menu of choices
# Ex: objective: minimize or maximize; constraints: =, <, >, =<, >=, != (?); etc.
def optimizer(chosen_variable, equation_name, objective, constraints, variable_bounds, initial_condition): # objective is either min or max; constraints is a list; initial condition is a guess for values of variables
  
  equations_conn = create_engine("mysql+pymysql://unwp2wrnzt46hqsp:b95S8mvE5t3CQCFoM3ci@bh10avqiwijwc8nzbszc-mysql.services.clever-cloud.com/bh10avqiwijwc8nzbszc")

  sql = "SELECT * FROM equations_table"
  read_sql = pd.read_sql(sql, equations_conn)

  # if chosen_variable is equal to equation_name then...
  # if not, then...
  selected_eq = read_sql.loc[read_sql['equation_name']==equation_name]['equation'].values.tolist()[0]
  selected_variables = read_sql.loc[read_sql['equation_name']==equation_name]['x_variables'].str.split(",").to_list()[0]
  expression = sp.parsing.sympy_parser.parse_expr(selected_eq)
  eq_symbols = list(map(str, list(expression.free_symbols))) # list({some expression}.free_symbols) yeilds a list of all variables in the equations by order of which they appear in the equation
  
  eq_symbols_nums = []
  for v in range(len(eq_symbols)):
    eq_symbols_nums.append(eq_symbols[v].replace('X',''))
    v+=1
  eq_symbols_nums = list(map(int, eq_symbols_nums))
  sorted_variables = eq_symbols_nums.sort()
  sorted_variables = list(map(str, eq_symbols_nums))
  sorted_variables = ["X" + sortv for sortv in sorted_variables]

  eq_actual_variables = []
  for n in eq_symbols_nums.sort():
    eq_actual_variables.append(selected_variables[n])


  keys = sorted_variables
  values = []
  for s in range(len(sorted_variables)):
    values.append('x[{}]'.format(s))
    s+=1

  dict_of_xs = {keys[i]: values[i] for i in range(len(sorted_variables))}


  def f(x):
    list_to_execute = []
    for key, value in dict_of_xs.items():
      list_to_execute.append('{} = {}'.format(key, value))
    for ex in list_to_execute:
      exec(ex)

    y = eval(selected_eq)

    if objective == "minimize":
      y = y
    elif objective == "maximize":
      y = -y

    return y


  # Initial Condition for the Function:
  x_start = initial_condition


  # Constraints of the the Function:
  #cons = ({'type': 'eq', 'fun': lambda x: })


  # Bounds of the Variables:
  bnds = tuple(variable_bounds)


  #solution = minimize(f, x_start, method='SLSQP', bounds=bnds, constraints=cons)

  #y_sol = solution.fun
  #xs_sol = solution.x


  return



print("General is DONE!")