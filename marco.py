from codac import Interval, IntervalVector, CtcDist, Function,CtcFunction
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import networkx as nx
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation as Rot
import copy

def get_edm(nodes,t, graph):
    n = len(nodes)
    matrix = np.zeros((n,n))
    for k in range(n-1):
        for l in range(k+1,n):
            i = nodes[k]
            j = nodes[l]
            low = graph.get_edge_data(i,j)["mini"]
            up = graph.get_edge_data(i,j)["maxi"]
            matrix[k,l] = (1-t[k,l])*low + t[k,l]*up
            matrix[l,k] = (1-t[k,l])*low + t[k,l]*up
    return matrix

def get_x(EDM):
    U, S, Vh = np.linalg.svd(EDM)
    x = U[:,:3]
    return x
lamb = 1
tau = 0.1

def theta(x,i,j,c):
    return (lamb**2 *(c-np.sqrt(np.linalg.norm(x[3*i:3*(i+1)]-x[3*j:3*(j+1)])**2 + tau**2))**2 + tau**2)

def phi(x,i,j, low,up):
    value = lamb*(low - up) + theta(x,i,j,low) + theta(x,i,j,up)
    return value

def cost(x,nodes,graph):
    n = len(nodes)
    cost_value = 0
    for k in range(n-1):
        for l in range(k+1,n):
            i = nodes[k]
            j = nodes[l]
            low = graph.get_edge_data(i,j)["mini"]
            up = graph.get_edge_data(i,j)["maxi"]
            cost_value += phi(x,k,l,low,up)
    return cost_value

def get_position_sub(sub):
    t = np.ones((len(sub),len(sub))) * 0.7
    D = get_edm(list(sub.nodes),t,sub)
    x = get_x(D)
    res = minimize(cost,x.flatten(),args=(list(sub.nodes),sub),method='BFGS')
    X = res.x.reshape(len(sub),3)
    return X

def get_dico_sub(sub):
    dico_index_to_X = {}
    dico_X_to_index = {}
    count = 0
    for node in list(sub.nodes):
        dico_index_to_X[node] = count
        count += 1
    count = 0
    for node in list(sub.nodes):
        dico_X_to_index[count] = node
        count += 1
    return dico_index_to_X, dico_X_to_index

def visualize_step(list_sub, dico_interval):
    list_atomes_interested = []
    for i in list_sub:
        print(i.sub.nodes)
        i.test_constraint(dico_interval)
        list_atomes_interested += list(i.sub.nodes)
    print(f'{len(list_atomes_interested)} atomes for {len(set(list_atomes_interested))} different ones. {len(list_atomes_interested) - len(set(list_atomes_interested))} atome(s) to merge ')

if True:
    def get_edm(nodes,t, graph):
        n = len(nodes)
        matrix = np.zeros((n,n))
        for k in range(n-1):
            for l in range(k+1,n):
                i = nodes[k]
                j = nodes[l]
                low = graph.get_edge_data(i,j)["mini"]
                up = graph.get_edge_data(i,j)["maxi"]
                matrix[k,l] = (1-t[k,l])*low + t[k,l]*up
                matrix[l,k] = (1-t[k,l])*low + t[k,l]*up
        return matrix

    def get_x(EDM):
        U, S, Vh = np.linalg.svd(EDM)
        x = U[:,:3]
        return x
    lamb = 1
    tau = 0.1

    def theta(x,i,j,c):
        return (lamb**2 *(c-np.sqrt(np.linalg.norm(x[3*i:3*(i+1)]-x[3*j:3*(j+1)])**2 + tau**2))**2 + tau**2)

    def phi(x,i,j, low,up):
        value = lamb*(low - up) + theta(x,i,j,low) + theta(x,i,j,up)
        return value

    def cost(x,nodes,graph):
        n = len(nodes)
        cost_value = 0
        for k in range(n-1):
            for l in range(k+1,n):
                i = nodes[k]
                j = nodes[l]
                low = graph.get_edge_data(i,j)["mini"]
                up = graph.get_edge_data(i,j)["maxi"]
                cost_value += phi(x,k,l,low,up)
        return cost_value

    def get_position_sub(sub):
        t = np.ones((len(sub),len(sub))) * 0.7
        D = get_edm(list(sub.nodes),t,sub)
        x = get_x(D)
        res = minimize(cost,x.flatten(),args=(list(sub.nodes),sub),method='BFGS')
        X = res.x.reshape(len(sub),3)
        return X

    def get_dico_sub(sub):
        dico_index_to_X = {}
        dico_X_to_index = {}
        count = 0
        for node in list(sub.nodes):
            dico_index_to_X[node] = count
            count += 1
        count = 0
        for node in list(sub.nodes):
            dico_X_to_index[count] = node
            count += 1
        return dico_index_to_X, dico_X_to_index

    def visualize_step(list_sub, dico_interval):
        list_atomes_interested = []
        for i in list_sub:
            print(i.sub.nodes)
            i.test_constraint(dico_interval)
            list_atomes_interested += list(i.sub.nodes)
        print(f'{len(list_atomes_interested)} atomes for {len(set(list_atomes_interested))} different ones. {len(list_atomes_interested) - len(set(list_atomes_interested))} atome(s) to merge ')

def align_figures(figure1, figure2, point1, point2):
    # Calculer les vecteurs de translation pour chaque point en commun
    vector = figure1[point1] - figure2[point2]

    return vector
def calculate_rotation_matrix(fixed_point, target_point, desired_position, verbose):
    # Calculate vectors from fixed_point to target_point and fixed_point to desired_position
    vector_target = target_point - fixed_point
    vector_desired = desired_position - fixed_point

    if verbose > 1:
        print(f"test_lenght : {np.linalg.norm(vector_target)} , {np.linalg.norm(vector_desired)}")
    
    # Calculate the rotation axis and angle
    rotation_axis = np.cross(vector_target, vector_desired)
    rotation_axis /= np.linalg.norm(rotation_axis)

    #atan2(crossproduct.length,scalarproduct)
    angle = np.arctan2(np.linalg.norm(np.cross(vector_target, vector_desired)), np.dot(vector_target, vector_desired))

    # Create the rotation matrix
    rotation_matrix = Rot.from_rotvec(angle * rotation_axis)
    
    return rotation_matrix
def calculate_sec_rotation_matrix(segment_point1, segment_point2, target_point, desired_position, verbose):
    # Calculate vectors along the line segment
    vector_segment = segment_point2 - segment_point1
    vector_segment /= np.linalg.norm(vector_segment)

    vector_target = target_point - np.dot(target_point,vector_segment) * vector_segment
    vector_desired = desired_position - np.dot(desired_position,vector_segment) * vector_segment
    point_rotation = segment_point1 - np.dot(segment_point1, vector_segment) * vector_segment

    return calculate_rotation_matrix(point_rotation, vector_target, vector_desired, verbose), vector_segment, point_rotation
def compute_rotation_2p(segment_point1, segment_point2, sub1, sub2, constraints, verbose):
    # Calculate vectors along the line segment
    vector_segment = segment_point2 - segment_point1
    vector_segment /= np.linalg.norm(vector_segment)

    def cost_angle_function(angle, X1, X2, axis, point, constraint):
        rotation = Rot.from_rotvec(angle * axis)
        cost = 0
        for i, j, mini, maxi in constraint:
            norm = np.linalg.norm(X1[i] - (np.dot(X2[j]- point, axis) * axis + rotation.apply((X2[j] - (X2[j] - point) @ axis * axis) - point) + point))
            if mini < norm < maxi:
                # cost += (norm - mini)*(maxi - norm) * 0
                pass
            else:
                cost += 50*(norm - mini)**2*(maxi - norm)**2
        return cost

    point_rotation = segment_point1 - np.dot(segment_point1, vector_segment) * vector_segment

    # res_angle = minimize_scalar(cost_angle_function, bounds=(0, 2*np.pi), args=(sub1.X, sub2.X, vector_segment, point_rotation, constraints), method='bounded', options={'xopt': 0.0000001})
    res_angle = minimize_scalar(cost_angle_function, bounds=(0, 2*np.pi), args=(sub1.X, sub2.X, vector_segment, point_rotation, constraints), method='bounded')
    if verbose >1:
        print(res_angle)
        print(f'minimizing_cost_function: {cost_angle_function(np.array(res_angle.x), sub1.X, sub2.X, vector_segment, point_rotation, constraints)}')
    rotation = Rot.from_rotvec(res_angle.x * vector_segment)
    
    return rotation, vector_segment, point_rotation
def compute_rotation_1p(fixed_point, sub1, sub2, constraints, verbose):

    def cost_angles_function(angles, X1, X2, point, constraint):
        angle_theta, angle_phi = angles
        rotation_theta = Rot.from_rotvec(angle_theta * np.array([0,0,1]))
        rotation_phi = Rot.from_rotvec(angle_phi * rotation_theta.apply(np.array([0,1,0])))
        rotation = np.dot(rotation_theta, rotation_phi)
        cost = 0
        for i, j, mini, maxi in constraint:
            norm = np.linalg.norm(X1[i] - (rotation.apply(X2[j] - point) + point))
            if mini < norm < maxi:
                cost -= 16 * ((norm - mini)*(maxi - norm)/(maxi - mini))**2
                pass
            else:
                cost += 50*(norm - mini)**2*(maxi - norm)**2
        return cost

    init_angle = np.array([0,0])
    # res_angle = minimize_scalar(cost_angle_function, bounds=(0, 2*np.pi), args=(sub1.X, sub2.X, vector_segment, point_rotation, constraints), method='bounded', options={'xopt': 0.0000001})
    res_angle = minimize(cost_angles_function, init_angle.flatten(), args=(sub1.X, sub2.X, fixed_point, constraints), method='Nelder-Mead')
    if verbose > 0:
        print(res_angle)
        print(f'minimizing_cost_function: {cost_angles_function(np.array(res_angle.x), sub1.X.copy(), sub2.X.copy(), fixed_point, constraints)}')
    angle_theta, angle_phi = res_angle.x
    rotation_theta = Rot.from_rotvec(angle_theta * np.array([0,0,1]))
    rotation_phi = Rot.from_rotvec(angle_phi * rotation_theta.apply(np.array([0,1,0])))
    rotation = np.dot(rotation_theta, rotation_phi)    
    return rotation
class sub_fc:

    def __init__(self, fc_subgraph,dico_interval) -> None:
        sub = nx.Graph()
        for node in fc_subgraph:
            sub.add_node(node)

        for (i,j), (mini, maxi) in dico_interval.items():
            if i in fc_subgraph and j in fc_subgraph:
                sub.add_edges_from([(i,j,{"mini":mini, "maxi":maxi})])

        self.sub = sub
        self.X = get_position_sub(sub).astype(np.float64)
        self.index_to_X, self.X_to_index = get_dico_sub(sub)


    def test_communs_positions(self, sub2):
        res = list(set(self.sub.nodes).intersection(set(sub2.sub.nodes)))
        return res
    
    def test_communs_constraints(self, sub2, dico_interval):
        nodes1 = list(self.sub.nodes)
        nodes2 = list(sub2.sub.nodes)
        nodes1 = [x for x in nodes1 if x not in nodes2]
        nodes2 = [x for x in nodes2 if x not in nodes1]
        list_constraint = []
        for i in nodes1:
            for j in nodes2:
                try:
                    if len(dico_interval[(i,j)]):

                        list_constraint.append((self.index_to_X[i],sub2.index_to_X[j], dico_interval[(i,j)][0], dico_interval[(i,j)][1]))
                except KeyError:
                    pass
        return list_constraint

    

    def apply_translation(self, translation):
        for i in range(len(self.X)):
            self.X[i] += translation


    def apply_rotation_point(self, rotation, point):
        for i in range(len(self.X)):
            self.X[i] = rotation.apply(self.X[i] - point) + point
        

    def apply_rotation_axis(self, rotation, axis, point):
        for i in range(len(self.X)):
            self.X[i] = np.dot(self.X[i], axis) * axis + rotation.apply((self.X[i] - self.X[i] @ axis * axis) - point) + point

    def visualisation (self, dico_interval):
        for i in self.sub.nodes:
            for j in self.sub.nodes:
                try:
                    if len(dico_interval[(i,j)]) > 1:
                        print(f"({i}, {j}) -- lenght: {np.linalg.norm(self.X[self.index_to_X[i]] - self.X[self.index_to_X[j]])}, interval: {dico_interval[(i,j)]}")
                except KeyError:
                    pass
        print()


    def update_position_3p(self, sub2, common_points, verbose):
        #translation
        translation = align_figures(self.X, sub2.X, self.index_to_X[common_points[0]], sub2.index_to_X[common_points[0]])
        sub2.apply_translation(translation)


        #rotation 1 
        if verbose >1:
            print(common_points[:2])
        first_rotation = calculate_rotation_matrix(
            sub2.X[sub2.index_to_X[common_points[0]]],
            sub2.X[sub2.index_to_X[common_points[1]]],
            self.X[self.index_to_X[common_points[1]]],
            verbose
            )
        sub2.apply_rotation_point(first_rotation, sub2.X[sub2.index_to_X[common_points[0]]])


        #rotation 2
        second_rotation, axis, point = calculate_sec_rotation_matrix(
            sub2.X[sub2.index_to_X[common_points[0]]],
            sub2.X[sub2.index_to_X[common_points[1]]],
            sub2.X[sub2.index_to_X[common_points[2]]],        
            self.X[self.index_to_X[common_points[2]]],
            verbose
            )
        sub2.apply_rotation_axis(
            second_rotation,
            axis,
            point)
        
        if verbose>0:
            print(f"results: {np.linalg.norm(sub2.X[sub2.index_to_X[common_points[0]]] - self.X[self.index_to_X[common_points[0]]])} --- {np.linalg.norm(sub2.X[sub2.index_to_X[common_points[1]]] - self.X[self.index_to_X[common_points[1]]])} ---{np.linalg.norm(sub2.X[sub2.index_to_X[common_points[2]]] - self.X[self.index_to_X[common_points[2]]])}")


    def update_position_2p(self, sub2, common_points, constraints, verbose):
            #translation
            translation = align_figures(self.X, sub2.X, self.index_to_X[common_points[0]], sub2.index_to_X[common_points[0]])
            sub2.apply_translation(translation)


            #rotation 1 
            first_rotation = calculate_rotation_matrix(
                sub2.X[sub2.index_to_X[common_points[0]]],
                sub2.X[sub2.index_to_X[common_points[1]]],
                self.X[self.index_to_X[common_points[1]]],
                verbose
                )
            sub2.apply_rotation_point(first_rotation, sub2.X[sub2.index_to_X[common_points[0]]])


            #rotation 2 (segment_point1, segment_point2, sub1, sub2, constraints, verbose)
            second_rotation, axis, point = compute_rotation_2p(
                sub2.X[sub2.index_to_X[common_points[0]]],
                sub2.X[sub2.index_to_X[common_points[1]]],
                self,
                sub2,
                constraints, 
                verbose
                )
            sub2.apply_rotation_axis(
                second_rotation,
                axis,
                point)
            
            if verbose>0:
                print(f"results: {np.linalg.norm(sub2.X[sub2.index_to_X[common_points[0]]] - self.X[self.index_to_X[common_points[0]]])} --- {np.linalg.norm(sub2.X[sub2.index_to_X[common_points[1]]] - self.X[self.index_to_X[common_points[1]]])}")
                for (i,j, min, max) in constraints:
                    print(f'constraint ({i}, {j}) --- norm: {np.linalg.norm(sub2.X[j] - self.X[i])}, constraint: {[min, max]}')


    def update_position_1p(self, sub2, common_points, constraints, verbose):
            #translation
            translation = align_figures(self.X, sub2.X, self.index_to_X[common_points[0]], sub2.index_to_X[common_points[0]])
            sub2.apply_translation(translation)


            #rotation 1
            #fixed_point, sub1, sub2, constraints, verbose
            rotation = compute_rotation_1p(
                sub2.X[sub2.index_to_X[common_points[0]]],
                self,
                sub2,
                constraints,
                verbose
                )

            sub2.apply_rotation_point(rotation, sub2.X[sub2.index_to_X[common_points[0]]])

            if verbose>0:
                print(f"results: {np.linalg.norm(sub2.X[sub2.index_to_X[common_points[0]]] - self.X[self.index_to_X[common_points[0]]])}")
                for (i,j, min, max) in constraints:
                    print(f'constraint ({i}, {j}) --- norm: {np.linalg.norm(sub2.X[j] - self.X[i])}, constraint: {[min, max]}')


    def reversing_position_3p(self, sub2, common_points):
        plan_vect_1 = sub2.X[sub2.index_to_X[common_points[0]]] - sub2.X[sub2.index_to_X[common_points[1]]]
        plan_vect_2 = sub2.X[sub2.index_to_X[common_points[0]]] - sub2.X[sub2.index_to_X[common_points[2]]]
        
        inversion_axis = np.cross(plan_vect_1, plan_vect_2)
        inversion_axis /= np.linalg.norm(inversion_axis)
        distance = np.dot(inversion_axis, sub2.X[sub2.index_to_X[common_points[0]]]) * inversion_axis

        for i in range(len(sub2.X)):
            comp_to_inverse = np.dot(sub2.X[i] - distance, inversion_axis)
            sub2.X[i] -= 2 * inversion_axis * comp_to_inverse


    def reversing_position_1_2p(self, sub2):
        for i in range(len(sub2.X)):
            sub2.X[i][0] = -sub2.X[i][0]


    def test_constraint_merge(self, sub2, dico_interval, verbose):
        nodes1 = list(self.sub.nodes)
        nodes2 = list(sub2.sub.nodes)
        
        for i in nodes1 + nodes2:
            for j in nodes1 + nodes2:
                try :
                    if dico_interval[(i,j)][0] < \
                        np.linalg.norm(self.X[self.index_to_X[i]] - sub2.X[sub2.index_to_X[j]])\
                        < dico_interval[(i,j)][1]:
                        continue
                    else:
                        if verbose > 0:
                            print(f"atomes ({i},{j}): {dico_interval[(i,j)]} --- {np.linalg.norm(self.X[self.index_to_X[i]] - sub2.X[sub2.index_to_X[j]])}")
                        return False
                except KeyError:
                    continue
        return True


    def test_constraint(self, dico_interval):
        nodes = list(self.sub.nodes)
        for i in nodes:
            for j in nodes[i+1:]:
                try :
                    if dico_interval[(i,j)][0] < \
                        np.linalg.norm(self.X[self.index_to_X[i]] - self.X[self.index_to_X[j]])\
                        < dico_interval[(i,j)][1]:
                        continue
                    else:
                        print(f"atomes ({i},{j}): {dico_interval[(i,j)]} --- {np.linalg.norm(self.X[self.index_to_X[i]] - self.X[self.index_to_X[j]])}")
                        return False
                except KeyError:
                    continue
        return True

    def merge_X(self, sub2, sub):
        X = []
        for node in sub.nodes:
            pos1, pos2 = [], []
            try:
                pos1 = np.array(self.X[self.index_to_X[node]])
            except KeyError:
                pass
            try:
                pos2 = np.array(sub2.X[sub2.index_to_X[node]])
            except KeyError:
                pass
            if len(pos1) == 0:
                X.append(pos2)
            elif len(pos2) == 0:
                X.append(pos1)
            else:
                X.append((pos1 + pos2) / 2)
        index_to_X, X_to_index = get_dico_sub(sub)
        return X, index_to_X, X_to_index


    def merge(self, sub2, dico_interval, add_constraint = True):
        sub = nx.Graph()
        for node in self.sub.nodes:
            sub.add_node(node)
        for node in sub2.sub.nodes:
            if node not in self.sub.nodes:
                sub.add_node(node)

        for (i,j), (mini, maxi) in dico_interval.items():
            if i in sub.nodes and j in sub.nodes:
                sub.add_edges_from([(i,j,{"mini":mini, "maxi":maxi})])

        self.sub = sub
        self.X, self.index_to_X, self.X_to_index = self.merge_X(sub2, sub)

        if add_constraint:
            delta = 0.01
            for i in sub.nodes:
                for j in sub.nodes:
                    if i==j:
                        continue
                    try:
                        dico_interval[(i,j)]
                    except KeyError:
                        new_contraint = np.linalg.norm(self.X[self.index_to_X[i]] - self.X[self.index_to_X[j]])
                        dico_interval[(i,j)] = [new_contraint - delta, new_contraint + delta]
def test_and_merge_3p(i,j,list_sub, dico_interval, verbose=0):
    """ verbose has 3 stades: 0 no verbose, 1 mini verbose for each merging, 2 total verbose"""
    sub1 = list_sub[i]
    sub2 = list_sub[j]
    commun_points = sub1.test_communs_positions(sub2)
    if len(commun_points) < 3:
        return False
    else:
        if verbose > 0:
            print()
            print(f"3p_test i = {i}, j = {j}: n_communs: {len(commun_points)} --- {sub1.sub.nodes} --- {sub2.sub.nodes}")
            if verbose > 1:
                sub1.visualisation(dico_interval)
                sub2.visualisation(dico_interval)
        sub1.update_position_3p(sub2, commun_points, verbose)


        if verbose > 1:
            for j in list(sub1.sub.nodes) + list(sub1.sub.nodes):
                print(f'atome {j}')
                try: 
                    print(sub1.X[sub1.index_to_X[j]])
                except KeyError:
                    pass
                try:
                    print(sub2.X[sub2.index_to_X[j]])
                except KeyError:
                    pass


        if sub1.test_constraint_merge(sub2, dico_interval, verbose):
            sub1.merge(sub2, dico_interval)
            return True
        else:
            sub1.reversing_position_3p(sub2, commun_points)
            if sub1.test_constraint_merge(sub2, dico_interval, verbose):
                sub1.merge(sub2, dico_interval)
        return False
def merging_3p(list_sub, verbose, dico_interval):
    n = len(list_sub)
    merge_succeed, has_merged = True, False
    while merge_succeed:
        merge_succeed = False
        i, j = 0, 1
        while i < len(list_sub):
            while j < len(list_sub):
                if test_and_merge_3p(i,j,list_sub, dico_interval, verbose):
                    list_sub.pop(j)
                    merge_succeed, has_merged = True, True
                else:
                    j +=1
            i+=1
            j = i + 1
    print(f"{n} => {len(list_sub)}")
    return list_sub, has_merged

