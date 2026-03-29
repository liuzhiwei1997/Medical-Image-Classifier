import argparse
import json
import os
import faulthandler

def get_parser():
	# create parser object
	parser = argparse.ArgumentParser(
		description='Neural network classifier for medical imaging using Tensorflow',
		epilog='For questions and bug reports, contact Jacky Ko <jkmailbox1991@gmail.com>')

	# add arguments
	parser.add_argument(
		'-v', '--verbose',
		dest='verbose',
		help='Show verbose output',
		action='store_true')
	parser.add_argument(
		'-p','--phase', 
		dest='phase', 
		help='Training phase (default=TRAIN)',
		choices=['TRAIN','PREDICT'],
		default='TRAIN',
		metavar='[TRAIN PREDICT]')
	parser.add_argument(
		'--config_json',
		dest='config_json',
		help='JSON file for model configuration',
		type=str,
		default='./configs/config.json', 
		metavar='FILENAME'
		)
	parser.add_argument(
		'--gpu',
		dest='gpu',
		default='0',
		type=str,
		help='Select GPU device(s) (default = 0)',
		metavar='GPU_IDs')

	args = parser.parse_args()

	if args.verbose:
		args_dict = vars(args)
		for key in sorted(args_dict):
			print("{} = {}".format(str(key), str(args_dict[key])))

	return args

def main(args):
	try:
		import tensorflow.compat.v1 as tf
		tf.disable_v2_behavior()
		from core.model import MedicalImageClassifier
	except ModuleNotFoundError as e:
		raise ModuleNotFoundError(
			"Missing dependency: {}. Please install project dependencies before running training/prediction. "
			"For Linux/macOS Python 3.10 use tensorflow==2.15.* + SimpleITK; "
			"for Windows native GPU use tensorflow==2.10.* + CUDA 11.2 + cuDNN 8.1; "
			"for Python 3.7 use tensorflow==1.15.* + SimpleITK.".format(e.name)
		) from e

	# select gpu
	# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # e.g. "0,1,2", "0,2" 

	# read configuration file
	with open(args.config_json) as config_json:
		config = json.load(config_json)

	# session config
	config_proto = tf.ConfigProto()
	config_proto.gpu_options.allow_growth = True
	# config_proto.log_device_placement = True

	with tf.Session(config=config_proto) as sess:
		model = MedicalImageClassifier(sess,config)
		if args.phase == "TRAIN":
			model.train()
		else:
			model.predict() 

if __name__=="__main__":
	# for seg fault trace
	faulthandler.enable()
	args = get_parser()
	main(args)
