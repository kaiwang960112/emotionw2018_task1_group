import os, sys
import numpy as np
import pdb

run_num, score_files, weights_str = sys.argv[1], sys.argv[2:10], sys.argv[10:18]

#pdb.set_trace()
weights = [float(i) for i in weights_str]
submit_folder = 'run'+run_num
if not os.path.exists(submit_folder):
	os.mkdir(submit_folder)

int2label={2:'Positive', 1:'Neutral', 0:'Negative'}

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def main():
	score_dict = {}
	for i, f in enumerate(score_files):
		with open(f, 'r') as fid:
			for line in fid:
				splits = line.rstrip().split()
				
				# some image name has spaces
				k = [s+' ' for s in splits[:-3]]
				#pdb.set_trace()
				k = ''.join(k)
				filename, ext = os.path.splitext(k)
				k = filename + '.txt' #target_file 
				if not k in score_dict:
					score = np.array(splits[-3:]).astype(np.float32)
					if abs(np.sum(score)-1.0)>0.01:
						score_dict[k] = weights[i] * softmax(score) #score #
					else:
						score_dict[k] = weights[i] * score

				else:
					pdb.set_trace()
					score = np.array(splits[-3:]).astype(np.float32)
					if abs(np.sum(score)-1.0)>0.01:
						score_dict[k] += weights[i] *softmax(score) #
					else:
						score_dict[k] += weights[i] *score # 
	write_submit(score_dict)

def write_submit(score_dict):
	for k in score_dict:
		pred = np.argmax(score_dict[k])
		with open(os.path.join(submit_folder, k), 'w') as fid:
			fid.write(int2label[pred]+' '+str(score_dict[k]))

if __name__ == '__main__':
	main()