from scipy.spatial import distance, distance_matrix
import pandas as pd
import numpy as np
import time
import random
import os

def shift(sol):  # the position of one random city is changed randomly (shift)
    pos = list(np.random.choice(len(sol), 2, replace=False))  # select two random positions from route

    mi = min(pos)
    ma = max(pos)
    pos[0] = mi  # smaller of the selected two positions
    pos[1] = ma  # larger of the selected two positions

    job = sol[pos[1]]  # take the city which is at the larger position

    for i in range(pos[1], pos[0], -1):  # shift the position of cities that are in between the selected positions
        sol[i] = sol[i - 1]

    sol[pos[0]] = job  # put the city at the smaller position

    return sol

def roulette(gaincandidates):    #roulette_wheel selection is used in insertion and deletion
    max = sum(gaincandidates)
    pick = random.uniform(0, max)   #generate a random value between 0 and sum of the gains
    current = 0
    for idx in range(len(gaincandidates)):
        current += gaincandidates[idx]
        if current > pick:
            a = idx      # the selection probability of a node is proportional to its benefit of deletion/insertion
            break

    return a

def tourLen(seq, dist):   #calculate the tour length
    frm = seq
    to = seq[1:] + [seq[0]]
    return sum(dist[[frm, to]])

def calculateObjective(tour,dist,profits):   # calculate the objective value
    cost = tourLen(tour,dist)
    profit = sum(profits[tour])
    return profit-cost

def two_opt(seq, dist_mat): #n : number of cities, dist_mat: distance matrix for cities
    n=len(seq)
    seq=seq.copy()
    delta_min = -1
    #continue until there is no improvement in tour length
    while delta_min < 0:
        delta = []
        arcs = []
        #find all possible arcs that can be replaced
        for i in range(0, n):
            for k in range(i + 2, n):
                j = (i + 1) % n
                k = k % n
                l = (k + 1) % n
                #cycling is prevented by following if conditions
                if i == l:
                    continue
                if j == k:
                    continue
                d_ik = dist_mat[seq[i],seq[k]]
                d_jl = dist_mat[seq[j],seq[l]]
                d_ij = dist_mat[seq[i],seq[j]]
                d_kl = dist_mat[seq[k],seq[l]]
                delta_current=d_ik+d_jl-d_ij-d_kl
                delta.append(delta_current)
                arcs.append([i,j,k,l])
        delta_min = min(delta)
        delta_min_i = np.argmin(delta)
        arc_min = arcs[delta_min_i]
        #for the best improvement swap nodes
        if delta_min < 0:
            seq[arc_min[1]] , seq[arc_min[2]] = seq[arc_min[2]] , seq[arc_min[1]]
    return seq

def initialization(distMatrix,profits,nstart,construction_method="nearest insert",sortType="mergesort"):

    #Here, a sequence for number of customers is created
    #according to given problem size
    #this is going to be used to iterate through the for loop
    lower = int(len(profits) * 0.2)
    upper = int(len(profits) * 1)
    ncustomers = np.linspace(lower,upper,nstart,dtype=int)

    profs = profits.copy()
    profs[0]=-999 #profit of the depot is set to a very low number
    tours = [[]]*nstart
    tours_after2opt = [[]] * nstart
    initial_objectives = []
    initial_objectives_after2opt = []
    for idx,ncust in enumerate(ncustomers):
        tours[idx] = [0] #tour is initialized with depot
        while len(tours[idx])<=ncust-1:
            best = []
            best_values = []
            if construction_method == 'nearest insert': #implement nearest insert algorithm
                #SELECTION STEP
                for i,j in enumerate(tours[idx]):
                    # candidate customers are sorted according to
                    # non-increasing (profit/distance) values
                    candid = (-(profs)/distMatrix[j,:]).argsort(kind=sortType)
                    # select the candidate with max (profit/distance) value, that is not already in the tour
                    best.append([x for x in candid if x not in tours[idx]][0])
                    best_values.append(profs[best[i]]/distMatrix[j,best[i]])
                selected=best[np.argmax(best_values)]
                if len(tours[idx])<=2:
                    tours[idx].append(selected)
                    continue #skip the insertion step if there is only 1 cust and the depot in the tour
                #INSERTION STEP
                temp_vertices = tours[idx]+[tours[idx][0]] #temporarily add the depot at the very end of the tour
                arc_candid = []
                #find all possible insertions and their costs=(increase in distance when added)
                for a in range(len(temp_vertices)-1):  #try all possible insertions
                    c_ij=distMatrix[temp_vertices[a], temp_vertices[a+1]]
                    c_ik=distMatrix[temp_vertices[a], selected]
                    c_kj=distMatrix[selected, temp_vertices[a+1]]
                    arc_candid.append(c_ik+c_kj-c_ij)
                index = np.argmin(arc_candid)+1
                tours[idx].insert(index, selected)

            if construction_method == 'nearest neighbor': #implement nearest neighbor algorithm
                neigh = (-(profs)/distMatrix[tours[idx][-1], :]).argsort(kind=sortType)
                neighbors = [x for x in neigh if x not in tours[idx]]  # find neighbors that have not been selected yet
                tours[idx].append(neighbors[0])

        tours[idx] = shift(tours[idx]) #add randomness to resulting sequence for diversity
        initial_objectives.append(calculateObjective(tours[idx],distMatrix,profits))
        tours_after2opt[idx] = two_opt(tours[idx],distMatrix)
        initial_objectives_after2opt.append(calculateObjective(tours_after2opt[idx],distMatrix,profits))
    summary= pd.DataFrame(data={'initial_tour_len':ncustomers,
                                'initial_obj':initial_objectives,
                                'initial_obj_after2opt':initial_objectives_after2opt},
                          index=None)
    #print(summary)
    #print('Initialization time:',time.time()-starttime)
    return tours,tours_after2opt,initial_objectives,initial_objectives_after2opt

def h(seq): # a function to implement argsort on lists
    return sorted(range(len(seq)), key=seq.__getitem__)

def insertion(dat, tour,distMatrix,profits): #insertion operator
    tour_new = tour.copy()
    chosen_candid = None # initialize chosen candidate which is to be inserted to the given tour
    N = len(distMatrix)
    if len(tour)!=len(distMatrix):  # prevent insertion when the tour includes all cities
        candidates = [] # initialize candidate list
        candidates = [i for i in range(N) if i not in tour] # candidates are the cities that are not yet visited
        gain_candidates = [] # list for gain parameter of the cities which is defined below

        # choose the best candidate randomly among best 2 candidates according to
        # their contribution to the tour which is inversely propotional to distance added to the tour
        # and proportional to their profit (=profit/distance to nearest city in the tour)
        for candid in candidates: # calculate gain for each candidate city
            profit_candid = profits[candid]
            coord_candid = np.array(dat.iloc[candid][['x','y']]).reshape(1,-1)
            coord_tour = np.array(dat.iloc[tour][['x','y']])
            dist_to_tour = distance_matrix(coord_candid,coord_tour,p=2)
            gain_candid = np.max(profit_candid / dist_to_tour)
            gain_candidates.append(gain_candid)

        #best candidate is chosen via roulette wheel selection method
        chosen_candid = candidates[roulette(gain_candidates)]
        #insert the chosen candidate to the tour
        temp_vertices = (tour+[tour[0]]).copy()   #add the depot to the tour
        arc_candid = []
        selected = chosen_candid

        #find all possible insertions and their costs=(increase in distance)
        #then insert where the cost is minimum
        for a in range(len(temp_vertices)-1):
            c_ij=distMatrix[temp_vertices[a], temp_vertices[a+1]]
            c_ik=distMatrix[temp_vertices[a], selected]
            c_kj=distMatrix[selected, temp_vertices[a+1]]
            arc_candid.append(c_ik+c_kj-c_ij)
        index = np.argmin(arc_candid)+1
        tour_new.insert(index, selected)
    return tour_new,chosen_candid

def deletion(tour, distMatrix,profits): #deletion operator
    tour_new = tour.copy()
    chosen_candid = None
    if len(tour) >= 1: # do not delete when there is less than 2 cities in the tour
        candidates = []
        candidates = tour.copy()
        candidates.remove(0) #remove depot from the candidates
        gain_candidates = []

        # choose the best candidate via roulette wheel according to
        # their contribution(=profit loss/distance reduction)
        for candid in candidates:
            profit_candid = profits[candid]
            idx = tour.index(candid)

            if candid != tour[-1]:  # if candidate is not the last city on tour
                d_ij = distMatrix[tour[idx-1], tour[idx]]
                d_ik = distMatrix[tour[idx-1], tour[idx+1]]
                d_jk = distMatrix[tour[idx], tour[idx + 1]]

            else:
                idx=-1
                d_ij = distMatrix[tour[idx - 1], tour[idx]]
                d_ik = distMatrix[tour[idx - 1], tour[idx + 1]]
                d_jk = distMatrix[tour[idx], tour[idx + 1]]

            distance_gain = -(d_ik - d_ij - d_jk)
            # calculate gain of deletion
            total_gain = -profit_candid / (distance_gain + 0.0001)  #0.0001 term is added in order to avoid any possible division by zero error

            gain_candidates.append(total_gain)

        minimum = min(gain_candidates)
        adding = 0
        if minimum < 0:   #we set the negative gain as 0 and add this value to other gains to normalize the gains
            adding = -minimum
        gain_candidates = [x+adding for x in gain_candidates]

        # chose the candidate city to delete via roulette wheel selection method
        chosen_candid = candidates[roulette(gain_candidates)]

        #delete the chosen candidate from the tour
        tour_new.remove(chosen_candid)

    return tour_new,chosen_candid

#decide whether deletion or insertion is more profitable

def chooseBestAction(tour, dat, distMatrix, profits,incumbentObj,tabuList):
    #calculate objective function values of actions
    tour_old = tour.copy()
    tourDeleted, citytobeDeleted = deletion(tour_old,distMatrix,profits)
    tourInserted, citytobeInserted = insertion(dat, tour_old,distMatrix,profits)
    insertionObj = calculateObjective(tourInserted,distMatrix,profits)
    deletionObj = calculateObjective(tourDeleted,distMatrix,profits)

    if citytobeInserted == None:      # when there is no city left to be inserted, implemement deletion
        tour_new = tourDeleted.copy()

    elif citytobeDeleted == None: # when deletion is not permitted, perform insertion and add the inserted city to tabu list
        tour_new = tourInserted.copy()
        tabuList[citytobeInserted] = np.random.randint(10, 20)

    else:
        # if deletion if it gives better obj value than insertion
        if deletionObj>insertionObj:
            # check if it gives better than incumbent or not in tabu list, then implement deletion
            # else do not perform deletion
            if deletionObj>incumbentObj or citytobeDeleted not in tabuList.keys():
                tour_new = tourDeleted.copy()
            else:
                tour_new = tour.copy()
        # if deletion does not give better obj value than insertion and inserted city is not in the tabu list
        # perform insertion and update the tabu list
        elif citytobeInserted not in tabuList.keys():    #we again check the tabu list for insertion to avoid putting again newly deleted node with aspiration criterion
            tour_new = tourInserted.copy()
            tabuList[citytobeInserted] = np.random.randint(10, 20)
        # if all fails, do not make any change to the tour
        else:
            tour_new = tour
    return tour_new

#MAIN SOLVER

def main_loop(initial, highlow, N, construction_method, nstart, iterations):
    fname = "dataset-" + highlow + ".xls"
    dat = pd.read_excel(fname, sheet_name="eil" + str(N), header=None, index_col=0)
    dat.columns = ['x', 'y', 'profit']
    distances = distance_matrix(dat[['x', 'y']], dat[['x', 'y']], p=2)  # find euclidean distance
    profits = np.array(dat['profit'])

    tours,tours_after2opt,initial_objectives,initial_objectives_after2opt = initialization(distances,profits,nstart,construction_method)
    if initial is None:
        #tour = tours[np.random.randint(0,len(tours))]   #select the constructed initial tour randomly
        tour = tours[4]
        tour = shift(tour)
    else:
        tour = initial.copy()

    tabuList = {}
    incumbentSoln = tour.copy()
    incumbentObj = calculateObjective(incumbentSoln,distances,profits)

    # Start the timer
    starttime = time.time()

    count = 0   # this is a count of the iterations in which an improvement is to the incumbent solution is obtained
    for i in range(iterations):
        tour = chooseBestAction(tour, dat, distances, profits,incumbentObj,tabuList)
        tour = two_opt(tour,distances)
        currentObj = calculateObjective(tour,distances,profits)
        if currentObj > incumbentObj:
            count = 0
            currentObj = calculateObjective(tour,distances,profits)
            incumbentObj = currentObj
            incumbentSoln = tour.copy()
        else:
            count += 1
        #UPDATE THE TABU LIST
        for key, value in list(tabuList.items()):
                v = tabuList
                tabuList[key] = tabuList[key] - 1
                if tabuList[key] == 0:
                    del(tabuList[key])
        '''
        #SHAKING PART (OPTIONAL)
        if count == 40:
            random.shuffle(tour)
            tabuList.clear()
        '''

    timediff = time.time() - starttime
    #print("Total CPU time: " + str(timediff))
    #print("Number of cities visited: (with depot):" + str(len(incumbentSoln)))

    #Reput the depot at the beginning of the solution representation for easy reading
    x = incumbentSoln.index(0)
    first = incumbentSoln[x:]
    first.extend(incumbentSoln[0: x])
    incumbentSoln = first

    print(incumbentObj, incumbentSoln, len(incumbentSoln), timediff)

for i in range(5):    #PRINT THE RESULTS
    main_loop(None, 'HP', 51, 'nearest insert', 10, 1000)
