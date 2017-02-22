import os,sys

if __name__ == '__main__':
	lines = open(sys.argv[1])
	mapp = {}
	count = 1
	for line in lines:
		# print line
		mapp[line.strip()] = count 
		count = count + 1
	# print mapp

	lines = open(sys.argv[2])
	for line in lines:
		print mapp[line.strip()]

