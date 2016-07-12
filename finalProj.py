#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

def randMap(map_size,msg,M,N):

	print 'Begin search for valid initial mapping...'
	#set valid map flag to false
	valid = False

	#while a valid initial mapping is not found
	while not valid:
		#generate a random mapping
		map = np.random.permutation(map_size)
		#apply mapping to the message
		new_msg = applyMap(map,msg,map_size)
		#get likelihood of mapping
		log_PI = logPI(new_msg,M,N)

		
		#if the log of the likelihood is not -inf
		if log_PI != float('-inf'):
			#the map has a non-zero likelihood PI, and is considered a valid starting point
			valid = True
			print 'Valid map found.'

	return map, new_msg


def applyMap(f,msg,map_size):

	new_msg = np.copy(msg)
	
	#for each of the 26 letters
	for i in range(map_size):
		
		#find the location of current letter in the message
		letter_loc = np.nonzero(msg == i)[0]
		#replace the current letter i at those locations according to map f
		new_msg[letter_loc] = f[i]
	
	return new_msg


def logPI(msg,M,N):
	#find the beginning of each transition
	first = np.copy(msg[0:N-2])
	#find the end of each transition
	second = np.copy(msg[1:N-1])

	#get the frequency of all transitions
	pair_freq = M[first,second]


	#if any of the frequencies are zero, the PI = 0, and logPI = -inf
	if np.sum(pair_freq == 0) != 0:
		log_PI = float('-inf')
	
	#else, take the log of each frequency and sum for the likelihood PI
	else:
		log_pair_freq = np.log(pair_freq)
		log_PI = np.sum(log_pair_freq)
	
	return log_PI

def letterSwap(letter_p,letter_q,f,msg):
	
	#swap for message
	#find the locations of p and q in the most recent msg
	p_loc = np.nonzero(msg == letter_p)[0]
	q_loc = np.nonzero(msg == letter_q)[0]
	
	new_msg = np.copy(msg)
	#at the p locations, assign letter q
	new_msg[p_loc] = letter_q
	#at the q locations, assign letter p
	new_msg[q_loc] = letter_p
	
	
	#swap for mapping
	#find the index where f = p, and the index where f = q
	p_idx = np.nonzero(f == letter_p)[0][0]
	q_idx = np.nonzero(f == letter_q)[0][0]
	
	new_f = np.copy(f)
	#assign q to p's index
	new_f[p_idx] = letter_q
	#assign p to q's index
	new_f[q_idx] = letter_p

	return new_msg, new_f


def acceptance(old_PI,new_PI):
	
	#if new log_PI is -inf, new PI is zero, new map is rejected
	if new_PI == float('-inf'):
		acc = False
		return acc
	
	#if new logPI is greater than old logPI, the new PI is higher and the move is accepted
	if new_PI > old_PI:
		acc = True
	else:
		acc = False

	return acc

def decodeDriver(MCS,f,msg,M,N):

	num_succ = 0
	
	#initial mapping, message, and likelihood
	old_f = f
	old_msg = np.copy(msg)
	old_PI = logPI(old_msg,M,N)

	for i in range(MCS):
	
		#randomly pick two letters to swap
		swapped = np.random.randint(26,size = (2,1))
		#apply the swap to the message/map
		new_msg, new_f = letterSwap(swapped[0],swapped[1],old_f,old_msg)
		#get the new likelihood
		new_PI = logPI(new_msg,M,N)
		#check for acceptance
		accept = acceptance(old_PI,new_PI)
		
		if accept:
			num_succ += 1
			#new becomes current/old
			old_f = new_f
			old_msg = new_msg
			old_PI = new_PI


	acc_ratio = float(num_succ)/float(MCS)
	final_msg = old_msg
	final_map = old_f

	return final_msg,final_map,acc_ratio

#print output header
print 'Message Decoder now running...\n'



#import the initial message from file as an array of characters
msg_file = 'coded_msg.dat'
infile = open(msg_file,'r')
msg_alpha = np.loadtxt(infile,dtype = 'c',delimiter = False)
infile.close()

#convert characters to integers
msg_int = np.copy(msg_alpha.view(np.int8)) - 65
space_loc = np.nonzero(msg_int < 0)
msg_int[space_loc] = 26
N = len(msg_int)

print 'Initial encrypted message'
print 'Message length is {0:d} characters (including spaces)'.format(N)
print ''.join(msg_alpha)
print '\n'

#import the pair freq table and store as 27 x 27 matrix
pairfreq_file = 'pairFreq.dat'
infile = open(pairfreq_file,'r')
M = np.loadtxt(infile,dtype = 'f')
infile.close()

#normalize M values as probabilities
total = np.sum(M)
M = M/total



#propose initial mapping randomly
map_size = 26

t = time.time()
f,msg = randMap(map_size,msg_int,M,N)
elapsed = time.time() - t


print 'Map size is {0:d} characters long'.format(map_size)
print 'Map generated in {0:f} seconds'.format(elapsed)
print 'Initial mapping:'
print f
print ''


MCS = 5000
print 'Ready to start MCMC for optimization...'
print 'Number of MC steps is {0:d}'.format(MCS)

t = time.time()
final_msg,final_map,acc_ratio = decodeDriver(MCS,f,msg,M,N)
elapsed = time.time() - t
print 'All MC steps completed in {0:f} seconds.'.format(elapsed)
print 'Acceptance ratio = {0:f}\n\n'.format(acc_ratio)

#convert message back to alpha
final_msg_int = final_msg + 65
space_loc = np.nonzero(final_msg_int == 26+65)
final_msg_int[space_loc] = 32
final_msg_alpha = np.copy(final_msg_int.view('c'))

print 'Best mapping found'
print final_map


#check the final map by applying it to the initial message
print '\nApply the best mapping found to initial encrypted message...'
msg_check = applyMap(final_map,msg_int,map_size)
msg_check_int = msg_check + 65
space_loc = np.nonzero(msg_check_int == 26+65)
msg_check_int[space_loc] = 32
msg_check_alpha = np.copy(msg_check_int.view('c'))
print 'Decoded message:'
print ''.join(msg_check_alpha)
print'\n\n---Program fully executed---\n'





