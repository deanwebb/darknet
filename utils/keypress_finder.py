import os
import rosbag
import rospy
import make_snippet_bag
import pickle

from keyboard.msg import Key

# keys of interest
# Key.KEY_c
# Key.KEY_v
# Key.KEY_l
# Key.KEY_i
# Key.KEY_s
# Key.KEY_SPACE

PICKLE_CACHE = os.path.join(os.getcwd(), 'pickle')


def make_bags(directory_name, event_log):
	"""
	"""

	top_dir = event_log['top_dir']

	for bag_name in event_log[directory_name]:
		bag_file_name = top_dir + directory_name + bag_name

		print('loading bag ' + bag_file_name)
		bag = rosbag.Bag(bag_file_name)
		start_time_secs = get_start_secs(bag)

		event_times = event_log[directory_name][bag_name]

		construction_count = 1
		for event_time in event_times:
			prefix, extension = os.path.splitext(bag_name)
			construction_bag_nme = prefix + '_' + str(construction_count) + extension
			construction_bag = rosbag.Bag(construction_bag_name, 'w+')

			construction_count += 1

			t0 = event_time - PRECEEDING_SECONDS
			t1 = event_time + FOLLOWING_SECONDS

			for topic, msg, t in tqdm(bag.read_messages()):
				time_from_beginning = t.to_sec() - start_time_secs

				if (time_from_beginning > t0) and (time_from_beginning < t1):
					construction_bag.write(topic, msg, t)

	return



def do_stuff(bag_full_path, bag):
	"""
	"""

	bag_snaps = []
	pickle_file = '{}.pickle'.format(bag_full_path.replace('/','_'))
	pickle_out = os.path.join(PICKLE_CACHE, pickle_file)

	if os.path.exists(pickle_out):
		pickle_in = open(pickle_out, 'rb')
		pickle_dict = pickle.load(pickle_in)
		return pickle_dict['bag_snaps']

	for topic, msg, t in bag.read_messages():
		if topic == '/dbw/toyota_dbw/car_state':
			try:
				if msg.keypressed == Key.KEY_c:
					# print('c key pressed at {}'.format(t))
					bag_snaps.append(('C', t))




				if msg.keypressed == Key.KEY_v:
				 	# print('v key pressed at {}'.format(t))
					bag_snaps.append(('V', t))
			except AttributeError:
				continue




	with open(pickle_out, 'w+') as f:
		print(bag_snaps)
		pickle_dict = {'bag_snaps': bag_snaps}
		pickle.dump(pickle_dict, f)


	return bag_snaps


if __name__ == '__main__':
	"""
	"""

	# change this depending on where you have sawmill mounted
	# and what directory you want to search in

	# list of all the bag file names in top_dir
	top_dir = '/data/kache-workspace/bags'
        all_bag_dirs = os.listdir(top_dir)
	save_dir = '/data/kache-workspace/processed_frames/'
	bag_data = {}
	index = 1

        for dirs in [d for d in all_bag_dirs if os.path.isdir(os.path.join(top_dir, d))]:
		sub_dirs = os.listdir(os.path.join(top_dir, dirs))
		for bag_dir in sub_dirs:
			fpath = os.path.join(top_dir, dirs, bag_dir)
		        print("python bag_to_images_from_keys.py --bag_dir={} --save_dir={} --skip_rate=2".format(fpath, save_dir))
			#if not os.path.exists("../construction_frames/{}".format(bag_dir)):
                        try:
				os.system("python bag_to_images_from_keys.py --bag_dir={} --save_dir=../{} --skip_rate=2".format(fpath, os.path.join(save_dir, "frames{}".format(index))))
				index += 1
			except:
				continue
