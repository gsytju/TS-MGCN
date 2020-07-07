import os
import numpy as np
import sys
from rdkit import Chem

np.set_printoptions(threshold=np.inf)

def getNameMatrix(filename):
    nums = len(filename)
    nameMatrix = []

    for i in range(nums):
        line = filename[i]
        if '_' in line:      #judge TS or Mol
            underline = line.index('_')
            if '+' in line[0:underline]:    #H-abstraction or R-addition_multibond
                plus1 = line.index('+')
                rmol1 = line[0:plus1]
                rmol2 = line[plus1+1:underline]
                product = line[underline+1:]
                if '+' in product:     #H-abstraction
                    plus2 = product.index('+')
                    pmol1 = product[0:plus2]
                    pmol2 = product[plus2+1:]
                    nameMatrix.append([rmol1, rmol2, pmol1, pmol2])
                else:      #R-addition_multibond
                    pmol = product
                    nameMatrix.append([rmol1, rmol2, pmol, '0'])
            else:         #Intra_H-migration
                rmol = line[0:underline]
                pmol = line[underline+1:]
                nameMatrix.append([rmol, '0', pmol, '0'])
        else:       #Molecules or radicals
            mol = line
            nameMatrix.append([mol, '0', '0', '0'])
    # print(count)
    # print(np.asarray(nameMatrix))
    return np.asarray(nameMatrix)

def ExtendConvertToGraph(filename):
    maxNumAtoms = 12
    nameMatrix = getNameMatrix(filename)
    adj = []
    features = []

    for i in range(len(nameMatrix)):
        iFeature = np.zeros((maxNumAtoms, 36))
        iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
        # print(nameMatrix[i, :])
        if nameMatrix[i, 2] != '0':     # TS
            if nameMatrix[i, 3] != '0':    #H_abstraction
                if '[' in nameMatrix[i, 0] and '[' not in nameMatrix[i, 1]:
                    nameMatrix[i, 0], nameMatrix[i, 1] = nameMatrix[i, 1], nameMatrix[i, 0]
                if '[' in nameMatrix[i, 3] and '[' not in nameMatrix[i, 2]:
                    nameMatrix[i, 2], nameMatrix[i, 3] = nameMatrix[i, 3], nameMatrix[i, 2]
                if nameMatrix[i, 1] != '[H]' and nameMatrix[i, 2] != '[H]':
                    rmol1 = Chem.MolFromSmiles(nameMatrix[i, 0])
                    rmol2 = Chem.MolFromSmiles(nameMatrix[i, 1])
                    pmol1 = Chem.MolFromSmiles(nameMatrix[i, 2])
                    pmol2 = Chem.MolFromSmiles(nameMatrix[i, 3])
                    # Adj
                    iAdjTmpr1 = Chem.rdmolops.GetAdjacencyMatrix(rmol1)
                    iAdjTmpr2 = Chem.rdmolops.GetAdjacencyMatrix(rmol2)
                    len1 = len(iAdjTmpr1)
                    len2 = len(iAdjTmpr2)
                    # estimate the maxnum of atoms needed for the Adjacency Matrix of TS
                    # print(rmol1.GetNumAtoms() + rmol2.GetNumAtoms())
                    # Feature
                    # Feature-preprocessing
                    iFeatureTmpr1, iFeatureTmpp1 = FeaturePreProcess_TS_1(rmol1, pmol1)
                    x_index, iFeature1 = FeatureProcess(iFeatureTmpr1, iFeatureTmpp1)
                    iFeatureTmpr2, iFeatureTmpp2 = FeaturePreProcess_TS_1(rmol2, pmol2)
                    y_index, iFeature2 = FeatureProcess(iFeatureTmpr2, iFeatureTmpp2)

                    iFeature[0:len1, 0:36] = iFeature1
                    iFeature[len1:len1+len2, 0:36] = iFeature2
                    features.append(iFeature)
                    # Adj-preprocessing
                    iAdj[0:len1, 0:len1] = iAdjTmpr1 + np.eye(len1)
                    iAdj[len1:len1+len2, len1:len1+len2] = iAdjTmpr2 + np.eye(len2)
                    adj.append(np.asarray(iAdj))
                else:
                    if nameMatrix[i, 1] == '[H]':
                        rmol = Chem.MolFromSmiles(nameMatrix[i, 0])
                        pmol = Chem.MolFromSmiles(nameMatrix[i, 2])
                    if nameMatrix[i, 2] == '[H]':
                        rmol = Chem.MolFromSmiles(nameMatrix[i, 1])
                        pmol = Chem.MolFromSmiles(nameMatrix[i, 3])
                    iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(rmol)
                    len_mol = len(iAdjTmp)
                    iFeatureTmp1, iFeatureTmp2 = FeaturePreProcess_TS_1(rmol, pmol)
                    x_index, iFeature_mol = FeatureProcess(iFeatureTmp1, iFeatureTmp2)
                    iFeature[0:len_mol, 0:36] = iFeature_mol
                    iFeature[x_index, 34] = 1      #align for H_abstraction
                    # print(iFeature)
                    features.append(iFeature)
                    iAdj[0:len_mol, 0:len_mol] = iAdjTmp + np.eye(len_mol)
                    adj.append(np.asarray(iAdj))

            elif nameMatrix[i, 1] == '0':    #Intra_H-migration
                rmol_m = Chem.MolFromSmiles(nameMatrix[i, 0])
                pmol_m = Chem.MolFromSmiles(nameMatrix[i, 2])
                iAdjTmp_m = Chem.rdmolops.GetAdjacencyMatrix(rmol_m)
                len_m = len(iAdjTmp_m)
                iFeatureTmp1_m, iFeatureTmp2_m = FeaturePreProcess_TS_1(rmol_m, pmol_m)
                x_index_m, iFeature_m = FeatureProcess(iFeatureTmp1_m, iFeatureTmp2_m)
                iFeature[0:len_m, 0:36] = iFeature_m
                features.append(iFeature)
                iAdj[0:len_m, 0:len_m] = iAdjTmp_m + np.eye(len_m)
                adj.append(np.asarray(iAdj))

            elif nameMatrix[i, 1] != '0' and nameMatrix[i, 3] == '0':    #R-addition_multibond
                rmol1_r = Chem.MolFromSmiles(nameMatrix[i, 0])
                pmol_r = Chem.MolFromSmiles(nameMatrix[i, 2])
                iAdjTmp_r = Chem.rdmolops.GetAdjacencyMatrix(pmol_r)
                len_r = len(iAdjTmp_r)
                if nameMatrix[i, 1] != '[H]':
                    rmol2_r = Chem.MolFromSmiles(nameMatrix[i, 1])
                    iFeatureTmp1_r, iFeatureTmp2_r = FeaturePreProcess_TS_2(rmol1_r, rmol2_r, pmol_r)
                    x_index_r, iFeature_r = FeatureProcess(iFeatureTmp1_r, iFeatureTmp2_r)
                    iFeature[0:len_r, 0:36] = iFeature_r
                else:
                    iFeatureTmp1_r, iFeatureTmp2_r = FeaturePreProcess_TS_1(rmol1_r, pmol_r)
                    x_index_r, iFeature_r = FeatureProcess(iFeatureTmp1_r, iFeatureTmp2_r)
                    iFeature[0:len_r, 0:36] = iFeature_r
                    iFeature[x_index_r, 35] = 1      # align for R-addition_multibond
                features.append(iFeature)
                iAdj[0:len_r, 0:len_r] = iAdjTmp_r + np.eye(len_r)
                adj.append(np.asarray(iAdj))

        if nameMatrix[i, 2] == '0':     #Mol
            mol = Chem.MolFromSmiles(nameMatrix[i, 0])
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(mol)
            leni = len(iAdjTmp)
            if leni <= maxNumAtoms:
                iAdj[0:leni, 0:leni] = iAdjTmp + np.eye(leni)
                adj.append(np.asarray(iAdj))
                iFeaturetmp = FeatureProcess_Mol(mol)
                iFeature[0:leni, 0:36] = iFeaturetmp
                features.append(iFeature)

    features = np.asarray(features)
    adj = np.asarray(adj)
    return features, adj

def FeaturePreProcess_TS_1(mol1, mol2):
    iFeatureTmp1 = []
    iFeatureTmp2 = []
    for atom1 in mol1.GetAtoms():
        iFeatureTmp1.append(GetAtomFeature(atom1))
    for atom2 in mol2.GetAtoms():
        iFeatureTmp2.append(GetAtomFeature(atom2))
    iFeatureTmp1 = np.asarray(iFeatureTmp1)
    iFeatureTmp2 = np.asarray(iFeatureTmp2)
    return iFeatureTmp1, iFeatureTmp2

def FeaturePreProcess_TS_2(rmol1, rmol2, pmol):
    iFeatureTmp1 = []
    iFeatureTmp2 = []
    for atom1 in rmol1.GetAtoms():
        iFeatureTmp1.append(GetAtomFeature(atom1))
    for atom2 in rmol2.GetAtoms():
        iFeatureTmp1.append(GetAtomFeature(atom2))
    for atom3 in pmol.GetAtoms():
        iFeatureTmp2.append(GetAtomFeature(atom3))
    iFeatureTmp1 = np.asarray(iFeatureTmp1)
    iFeatureTmp2 = np.asarray(iFeatureTmp2)
    return iFeatureTmp1, iFeatureTmp2

def FeatureProcess(iFeatureTmp1, iFeatureTmp2):
    iFeature = []
    index = np.arange(0, len(iFeatureTmp1))
    x_index = index[iFeatureTmp1[:, 2] != iFeatureTmp2[:, 2]]   # Recognize the reaction site
    Num_ring = 0
    for i in range(len(iFeatureTmp1)):
        if iFeatureTmp1[i, 4] == 'True':
            Num_ring += 1
    # print(x_index)
    iFeatureTmp = iFeatureTmp1
    iFeatureTmp[:, 1] = (iFeatureTmp1[:, 1].astype(float) + iFeatureTmp2[:, 1].astype(float)) / 2
    iFeatureTmp[:, 2] = (iFeatureTmp1[:, 2].astype(float) + iFeatureTmp2[:, 2].astype(float)) / 2
    iFeatureTmp[:, 3] = (iFeatureTmp1[:, 3].astype(float) + iFeatureTmp2[:, 3].astype(float)) / 2
    # print(iFeatureTmp)
    # print(iFeatureTmp.shape)
    for i in range(len(iFeatureTmp)):
        AtomFeature_line = np.array(one_of_k_encoding_unk(iFeatureTmp[i, 0], ['C', 'O']) +  # Atom Type (feature AT)
                    one_of_k_encoding(iFeatureTmp[i, 1].astype(float), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) +  # Atom Connections (feature AC)
                    one_of_k_encoding_unk(iFeatureTmp[i, 2].astype(float), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) +  # Atom TotalNumHs (feature NH)
                    one_of_k_encoding_unk(iFeatureTmp[i, 3].astype(float), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) +  # Atom TotalValence (feature NV)
                    [0, 0, 0, 0] + [iFeatureTmp[i, 5] == str(True)] + [0] + [0])  # (2, 9, 9, 9, 1, 1, 1, 1)               # Atom IsLipidRing, Atom NumRing, Atom IsAromatics, TS modification (feature LR, AR, MTS)
        if iFeatureTmp[i, 4] == str(True):
            if Num_ring == 3:
                AtomFeature_line[30] = 1
            if Num_ring == 4:
                AtomFeature_line[31] = 1
            if Num_ring == 5:
                AtomFeature_line[32] = 1
        iFeature.append(AtomFeature_line)
    iFeature = np.asarray(iFeature)
    iFeature = iFeature.astype(int)
    return x_index, iFeature
    
def FeatureProcess_Mol(mol):
    iFeatureTmp = []
    iFeature = []
    for atom in mol.GetAtoms():
        iFeatureTmp.append(GetAtomFeature(atom))
    iFeatureTmp = np.asarray(iFeatureTmp)
    # print(iFeatureTmp)
    # print(iFeatureTmp.shape)
    Num_ring = 0
    for i in range(len(iFeatureTmp)):
        if iFeatureTmp[i, 4] == 'True':
            Num_ring += 1
    for i in range(len(iFeatureTmp)):
        AtomFeature_line = np.array(one_of_k_encoding_unk(iFeatureTmp[i, 0], ['C', 'O']) +  # Atom Type (feature AT)
                    one_of_k_encoding(iFeatureTmp[i, 1].astype(float), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) +  # Atom Connections (feature AC)
                    one_of_k_encoding_unk(iFeatureTmp[i, 2].astype(float), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) +  # Atom TotalNumHs (feature NH)
                    one_of_k_encoding_unk(iFeatureTmp[i, 3].astype(float), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) +  # Atom TotalValence (feature NV)
                    [0, 0, 0, 0] + [iFeatureTmp[i, 5] == str(True)] + [0] + [0])  # (2, 9, 9, 9, 1, 1, 1, 1)               # Atom IsLipidRing, Atom NumRing, Atom IsAromatics, TS modification (feature LR, AR, MTS)
        if iFeatureTmp[i, 4] == str(True):
            if Num_ring == 3:
                AtomFeature_line[30] = 1
            if Num_ring == 4:
                AtomFeature_line[31] = 1
            if Num_ring == 5:
                AtomFeature_line[32] = 1
        iFeature.append(AtomFeature_line)
    iFeature = np.asarray(iFeature)
    iFeature = iFeature.astype(int)
    return iFeature

def GetAtomFeature(atom):
    return np.array([atom.GetSymbol()] + [atom.GetDegree()] + [atom.GetTotalNumHs()] +
                    [atom.GetTotalValence()] + [atom.IsInRing()] + [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
