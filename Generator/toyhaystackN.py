'''
    number of lines
    number of simulated events
'''


import sys
import math
import random
import ROOT as root
import multilines as ML
from utility import geometrycheck as gcheck

root.gROOT.ProcessLine(
"struct DataStruct{\
   vector<double>* dirx;\
   vector<double>* diry;\
   vector<double>* dirz;\
   vector<double>* pointx;\
   vector<double>* pointy;\
   vector<double>* pointz;\
   vector<double>* breakpointx;\
   vector<double>* breakpointy;\
   vector<double>* bpangle;\
   vector<double>* radius;\
   vector<double>* wirex;\
   vector<double>* wirey;\
   vector<double>* wirez;\
   vector<int>*    grid_id;\
   vector<int>*    grid_side;\
   vector<int>*    grid_layer;\
   vector<int>*    grid_column;\
   vector<int>*    break_layer;\
   vector<int>*    charge;\
   vector<int>*    calo_id;\
   vector<int>*    calo_type;\
   vector<int>*    calo_side;\
   vector<int>*    calo_wall;\
   vector<int>*    calo_column;\
   vector<int>*    calo_row;\
};");

Nsims = int(sys.argv[2]) # Number of simulated lines
Nlines = int(sys.argv[1])

# Set up ROOT data structures for file output and storage
file = root.TFile("haystack.root","recreate")
tree = root.TTree("hit_tree","Hit data")
tree.SetDirectory(file)

dataStruct = root.DataStruct()
dataStruct.dirx = root.std.vector('double')()
dataStruct.diry = root.std.vector('double')()
dataStruct.dirz = root.std.vector('double')()
dataStruct.pointx = root.std.vector('double')()
dataStruct.pointy = root.std.vector('double')()
dataStruct.pointz = root.std.vector('double')()
dataStruct.bpointx = root.std.vector('double')()
dataStruct.bpointy = root.std.vector('double')()
dataStruct.bangle = root.std.vector('double')()
dataStruct.radius = root.std.vector('double')()
dataStruct.wirex = root.std.vector('double')()
dataStruct.wirey = root.std.vector('double')()
dataStruct.wirez = root.std.vector('double')()
dataStruct.gridid = root.std.vector('int')()
dataStruct.gridside = root.std.vector('int')()
dataStruct.gridlayer = root.std.vector('int')()
dataStruct.gridcolumn = root.std.vector('int')()
dataStruct.breaklayer = root.std.vector('int')()
dataStruct.charge = root.std.vector('int')()
dataStruct.caloid = root.std.vector('int')()
dataStruct.calotype = root.std.vector('int')()
dataStruct.calowall = root.std.vector('int')()
dataStruct.caloside = root.std.vector('int')()
dataStruct.calorow = root.std.vector('int')()
dataStruct.calocolumn = root.std.vector('int')()
dataStruct.trackSplit = root.std.vector('int')() # HACK split lines

tree.Branch('dirx', dataStruct.dirx)
tree.Branch('diry', dataStruct.diry)
tree.Branch('dirz', dataStruct.dirz)
tree.Branch('pointx', dataStruct.pointx)
tree.Branch('pointy', dataStruct.pointy)
tree.Branch('pointz', dataStruct.pointz)
tree.Branch('breakpointx', dataStruct.bpointx)
tree.Branch('breakpointy', dataStruct.bpointy)
tree.Branch('bpangle', dataStruct.bangle)
tree.Branch('radius', dataStruct.radius)
tree.Branch('wirex',  dataStruct.wirex)
tree.Branch('wirey',  dataStruct.wirey)
tree.Branch('wirez',  dataStruct.wirez)
tree.Branch('grid_id', dataStruct.gridid)
tree.Branch('grid_side', dataStruct.gridside)
tree.Branch('grid_layer', dataStruct.gridlayer)
tree.Branch('grid_column', dataStruct.gridcolumn)
tree.Branch('break_layer', dataStruct.breaklayer)
tree.Branch('charge', dataStruct.charge)
tree.Branch('calo_id', dataStruct.caloid)
tree.Branch('calo_type', dataStruct.calotype)
tree.Branch('calo_side', dataStruct.caloside)
tree.Branch('calo_wall', dataStruct.calowall)
tree.Branch('calo_row', dataStruct.calorow)
tree.Branch('calo_column', dataStruct.calocolumn)
tree.Branch('track_split',dataStruct.trackSplit) # HACK split lines

wgr = ML.demonstratorgrid()
tgen = ML.track_generator()
dcalo = gcheck.demonstratorcalo()

for i in range(Nsims):
    tgen.haystack(Nlines)
    all = tgen.getLines()
    lines = []
    for entry in all:
        lrtracker = random.randint(0,1) # pick left or right
        lines.append((entry, lrtracker))

    # all hits related truth data in cluster
    cluster = wgr.multi_track_hits(lines)
    cluster2= dcalo.multi_calohits(lines)

    while len(cluster2) < Nlines: # too few calo hits, try again
        tgen.haystack(Nlines)
        all = tgen.getLines()
        lines = []
        for entry in all:
            lrtracker = random.randint(0,1) # XXX pick left or right
            lines.append((entry, lrtracker))
        cluster = wgr.multi_track_hits(lines)
        cluster2= dcalo.multi_calohits(lines)

    file.cd()
    # Prepare data structure for this line
    dataStruct.dirx.clear()
    dataStruct.diry.clear()
    dataStruct.dirz.clear()
    dataStruct.pointx.clear()
    dataStruct.pointy.clear()
    dataStruct.pointz.clear()
    dataStruct.bpointx.clear()
    dataStruct.bpointy.clear()
    dataStruct.bangle.clear()
    dataStruct.radius.clear()
    dataStruct.wirex.clear()
    dataStruct.wirey.clear()
    dataStruct.wirez.clear()
    dataStruct.gridid.clear()
    dataStruct.gridside.clear()
    dataStruct.gridlayer.clear() 
    dataStruct.gridcolumn.clear()
    dataStruct.breaklayer.clear()
    dataStruct.charge.clear()
    dataStruct.caloid.clear()
    dataStruct.calotype.clear()
    dataStruct.caloside.clear()
    dataStruct.calowall.clear()
    dataStruct.calorow.clear() 
    dataStruct.calocolumn.clear()
    dataStruct.trackSplit.clear() # HACK split lines
    
    counter = 0
    for k,val in cluster.iteritems():
        line = lines[k-1][0]  # line3 object
        cells = val[0] # first list in cluster tuple 
        radii = val[1] # as list
        info = val[2]  # as list

        dataStruct.dirx.push_back(line.v.x)
        dataStruct.diry.push_back(line.v.y)
        dataStruct.dirz.push_back(line.v.z)
        dataStruct.charge.push_back(0)

        ci, point = cluster2[k]
        calo_hit_point = point[0]
        type = ci[0][1]
        for w,r,mi in zip(cells,radii,info):
            if type == 1 and abs(w[1]) > abs(calo_hit_point.y): 
                continue # dismiss geiger hits outside xwall
            if type == 2 and abs(w[2]) > abs(calo_hit_point.z): 
                continue # dismiss geiger hits outside gveto
            dataStruct.radius.push_back(r)
            dataStruct.wirex.push_back(w[0])
            dataStruct.wirey.push_back(w[1])
            dataStruct.wirez.push_back(w[2])
            dataStruct.gridid.push_back(counter)
            gside = mi[0] # wire side
            grow = mi[1] # wire column
            gcol = mi[2] # wire layer
            dataStruct.gridlayer.push_back(grow)
            dataStruct.gridcolumn.push_back(gcol)
            dataStruct.gridside.push_back(gside) # not covered yet 
            counter += 1 # count up all hits for entire event
        dataStruct.trackSplit.push_back(counter) # HACK split lines
        
    for k,val in cluster2.iteritems():
        ci, point = val
        type = ci[0][1]
        side = ci[0][3]
        col  = ci[0][4]
        row  = ci[0][5]
        wall = ci[0][6]
        dataStruct.caloid.push_back(k-1)
        dataStruct.calorow.push_back(row)
        dataStruct.calocolumn.push_back(col)
        dataStruct.calotype.push_back(type)
        dataStruct.caloside.push_back(side)
        dataStruct.calowall.push_back(wall)
        dataStruct.pointx.push_back(point[0].x)
        dataStruct.pointy.push_back(point[0].y)
        dataStruct.pointz.push_back(point[0].z)
        
        
    tree.Fill() # data structure fully filled, line done
    
tree.Write() # write all lines to disk
file.Close()
