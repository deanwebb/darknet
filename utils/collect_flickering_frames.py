import os
import rosbag
import rospy
import make_snippet_bag
import pickle
import flicker_detector


PICKLE_CACHE = os.path.join(os.getcwd(), 'pickle')

if __name__ == '__main__':
	"""
	"""

	top_dir = '/data/kache-workspace/bags'
	all_bag_dirs = os.listdir(top_dir)
	save_dir = '/data/kache-workspace/processed_frames/'
	bag_data = {}
	index = 1


	for dirs in [d for d in all_bag_dirs if os.path.isdir(os.path.join(top_dir, d))]:
		sub_dirs = os.listdir(os.path.join(top_dir, dirs))
		for bag_dir in sub_dirs:
			fpath = os.path.join(top_dir, dirs, bag_dir)
			print("python flicker_detector.py --bag_dir={} --save_dir={} --skip_rate=1".format(fpath, save_dir))
			try:
				os.system("python flicker_detector.py --bag_dir={} --save_dir={} --skip_rate=1".format(fpath, os.path.join(save_dir, "frames{}".format(index))))
				index += 1
			except:
				continue
