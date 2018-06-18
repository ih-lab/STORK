from collections import defaultdict
import numpy as np


def read_values():
	filename = 'all_pgmc.txt'
	values = defaultdict(list)
	algs_dict = dict()
	algs_int = dict()
	f = open(filename, 'r')
	r = 0
	for line in f:
		parsed = line.replace('\n', '').split('\t')
		cnt = 0
		for val in parsed:
			if cnt == 0:
				alg = val
			else:
				values[alg].append(float(val))
			cnt+=1
			algs_dict[r] = alg
			algs_int[alg] = r
		r+=1

	filename = 'F_pgmc.txt'
	valuesF = defaultdict(list)
	algs_dict = dict()
	algs_int = dict()
	f = open(filename, 'r')
	r = 0
	for line in f:
		parsed = line.replace('\n', '').split('\t')
		cnt = 0
		for val in parsed:
			if cnt == 0:
				alg = val
			else:
				valuesF[alg].append(float(val))
			cnt+=1
			algs_dict[r] = alg
			algs_int[alg] = r
		r+=1

	filename = 'P_pgmc.txt'
	valuesP = defaultdict(list)
	algs_dict = dict()
	algs_int = dict()
	f = open(filename, 'r')
	r = 0
	for line in f:
		parsed = line.replace('\n', '').split('\t')
		cnt = 0
		for val in parsed:
			if cnt == 0:
				alg = val
			else:
				valuesP[alg].append(float(val))
			cnt+=1
			algs_dict[r] = alg
			algs_int[alg] = r
		r+=1

	filename = 'C_pgmc.txt'
	valuesC = defaultdict(list)
	algs_dict = dict()
	algs_int = dict()
	f = open(filename, 'r')
	r = 0
	for line in f:
		parsed = line.replace('\n', '').split('\t')
		cnt = 0
		for val in parsed:
			if cnt == 0:
				alg = val
			else:
				valuesC[alg].append(float(val))
			cnt+=1
			algs_dict[r] = alg
			algs_int[alg] = r
		r+=1

	algs = {'F':'FUSE', 'B': 'BEAMS', 'S': 'SMETANA', 'C':'CSRW', 'I': 'IsoRankN'}
	S_algs = []
	A_algs = []
	x = []
	for i in range(23):
		l = 40 + i * 5
		x.append(l)
		algs[str(l)] = '$\\rm{MPROPER}_{'+str(l)+'}$'
		A_algs.append(str(l))
		algs['0_'+str(l)] = '$\\rm{SeedGeneration}_{'+str(l)+'}$'
		S_algs.append(str('0_'+str(l)))


	cols = {'c5':13, 'nbc5': 46, 'a5': 9, 'IC5': 27, 'nent5': 23, 'aD5': 31}
	allcolls = cols.copy()
	cols = {'c4':12, 'nbc4': 45, 'a4': 8, 'IC4': 26, 'nent4': 22, 'aD4': 30}
	allcolls.update(cols)
	cols = {'c3':11, 'nbc3': 44, 'a3': 7, 'IC3': 25, 'nent3': 21, 'aD3': 29}
	allcolls.update(cols)
	cols = {'c2':10, 'nbc2': 43, 'a2': 6, 'IC2': 24, 'nent2': 20, 'aD2': 28}
	allcolls.update(cols)
	allcolls['ciq'] = 32
	allcolls['m2'] = 2
	allcolls['m3'] = 3
	allcolls['m4'] = 4
	allcolls['m5'] = 5
	allcolls['tot'] = 0
	allcolls['nbc'] = 1
	allcolls['p2'] = 43
	allcolls['p3'] = 44
	allcolls['p4'] = 45
	allcolls['p5'] = 46
	allcolls['cprot'] = 15
	allcolls['cpair'] = 14
	allcolls['ent2'] = 16
	allcolls['ent3'] = 17
	allcolls['ent4'] = 18
	allcolls['ent5'] = 19
	
	allcolls['tcedge'] = 34
	allcolls['ccedge'] = 35
	allcolls['ttedge'] = 39
	allcolls['ctedge'] = 40

	lent = (len(values['F']))
	if lent > 48:
		allcolls['mp2'] = 48
		allcolls['mp3'] = 49
		allcolls['mp4'] = 50
		allcolls['mp5'] = 51
	allcolls['tac'] = lent + 0 #total annotated clusters
	allcolls['tcc'] = lent + 1 #total consistent clusters
	allcolls['tspec'] = lent + 2
	allcolls['avgspec'] = lent + 3 
	allcolls['2spec'] = lent + 4
	allcolls['3spec'] = lent + 5
	allcolls['4spec'] = lent + 6
	allcolls['5spec'] = lent + 7
	allcolls['avgIC2'] = lent + 8
	allcolls['avgIC3'] = lent + 9
	allcolls['avgIC4'] = lent + 10
	allcolls['avgIC5'] = lent + 11
	allcolls['ECc'] = lent + 12
	allcolls['ECt'] = lent + 13


	
	col_proc=['nent2', 'nent3', 'nent4', 'nent5']
	for alg in values:
		string = ''
		string+= (alg + '\t')
		for i in range(4):
			#values[alg][allcolls[col_proc[i]]] = ((nent_null[i] - values[alg][allcolls[col_proc[i]]]) / nent_null[i])
			string+= (str(values[alg][allcolls[col_proc[i]]])  + '\t')
	#print (string)

	for alg in values:
		#print (alg, len(values[alg]), values[alg][48])
		Cs = ['c2', 'c3', 'c4', 'c5']
		As = ['a2', 'a3', 'a4', 'a5']

		specs = []
		tac = 0
		tcc = 0
		avgspec = 0
		for i in range(4):
			ac = values[alg][allcolls[As[i]]] * 1.0
			cc = values[alg][allcolls[Cs[i]]] * 1.0
			tac+=ac
			tcc+=cc
			#acc = ((cc / ac - null_model[i]) / null_model[i])
			acc = cc / ac
			specs.append(acc)
			avgspec+= (acc)
		spec = tcc / tac
		values[alg].append(tac)
		values[alg].append(tcc)
		values[alg].append(spec)
		values[alg].append(avgspec/4)
		for val in specs:
			values[alg].append(val)

	for alg in values:
		ics = ['avgIC2', 'avgIC3', 'avgIC4' ,'avgIC5']
		denum  = ['aD2', 'aD3', 'aD4', 'aD5']
		num = ['IC2', 'IC3', 'IC4' ,'IC5']
		for i in range(4):
			values[alg].append(values[alg][allcolls[num[i]]] / values[alg][allcolls[denum[i]]])
	for alg in values:
		values[alg].append(values[alg][allcolls['ccedge']] / float(values[alg][allcolls['tcedge']]))
		values[alg].append(values[alg][allcolls['ctedge']] / float(values[alg][allcolls['ttedge']]))
	return algs, algs_int, A_algs, S_algs, allcolls, values, valuesP, valuesF, valuesC