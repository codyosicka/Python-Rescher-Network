import General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys


wheat_array = np.array([[1,1,1,0,0], [0,1,0,1,1], [1,0,0,0,0], [0,0,1,0,0], [0,0,0,0,1]])

wheat_df = pd.DataFrame(wheat_array, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

wheat_structure_df = General.static_matrix_constructor(wheat_df)

d = General.initialize_mini_network(wheat_structure_df, General.static_causal_order(wheat_structure_df), 'wheat_test')

print(d)

'''plt.figure(1)
nx.draw_planar(d,
                node_color='red', # draw planar means the nodes and edges are drawn such that not edges cross
                #node_size=size_map, 
                arrows=True, with_labels=True)
plt.show()'''



# Build folder for mini networks
#import os

#path = 'C:\\Users\\Xaos\\Desktop\\Python'
#os.chdir(path)

#Newfolder = 'Tutorial-01'
#os.makedirs(Newfolder)

folder_path = r'C:\\Users\\Xaos\\Desktop\\Python\\Tutorial-01'

def listDir(dir):
	fileNames = os.listdir(dir)
	files_list = []
	for fileName in fileNames:
		files_list.append(fileName)
		#print('File Name: ' + fileName)
		#print('Folder Path: ' + os.path.abspath(os.path.join(dir, fileName)), sep='\n')

	print(files_list)

if __name__ == '__main__':
	listDir(folder_path)
