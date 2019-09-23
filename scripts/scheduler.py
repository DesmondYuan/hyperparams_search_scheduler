import argparse
import os
import pickle
import csv
import json
import sys
import subprocess
from os.path import dirname, realpath
import random
sys.path.append(dirname(dirname(realpath(__file__))))
import utils.parsing as parsing
import time

######################################################### Global variables
CONFIG_NOT_FOUND_MSG = "ERROR! {}: {} config file not found."
RESULTS_PATH_APPEAR_ERR = 'WARNING! Existing results for the same config(s). \nJob {} skipped. Use --rerun_experiments to override.'
SUCESSFUL_SCHEDULE_JOB_STR = "This job {} is launched on this machine {}"
SUCESSFUL_SCHEDULE_FINISH_STR = "SUCCESS! All worker experiments scheduled!"
MACHINES = ["machine_{}".format(str(machine_i).zfill(4)) for machine_i in range(20)]
SLEEP_TIME_SCHEDULED = 5 # seconds
SLEEP_TIME_SCHEDULED = 1 # seconds
SLEEP_TIME_BUSY = 60
# imagine we have ssh access to 20 VMs


######################################################### dispatcher args
parser = argparse.ArgumentParser(description='Regev Lab Grid Search Dispatcher.')
parser.add_argument("--experiment_config_path", required=False, type=str, help="Path of the master experiment config")
parser.add_argument("--debugging", action='store_true', default=False, help="Set to true for demo purpose.")

parser.add_argument('--log_dir', type=str, default="logs", help="path to store logs and detailed job level result files")
parser.add_argument('--results_path', type=str, default="results", help="path to store grid_search table. This is preferably on shared storage")
parser.add_argument('--rerun_experiments', action='store_true', default=False, help='whether to rerun experiments with the same result file location')
parser.add_argument('--shuffle_experiment_order', action='store_true', default=False, help='whether to shuffle order of experiments')

parser.add_argument('--check_queue', action='store_true', default=False, help='check job running and waiting')
parser.add_argument('--drop_queue', action='store_true', default=False, help='drop the job waiting list')
parser.add_argument('--refresh_queue', action='store_true', default=False, help='run jobs in the queue that are not luanched because a previous scheduler crashed')
parser.add_argument('--kill_queue', action='store_true', default=False, help='kill the running jobs')

args = parser.parse_args()


######################################################### scheduler functions
def test_ssh():
	if args.debugging:
		return 0
	try:
		assert os.system('ssh {} "exit"'.format(MACHINES[0])) is 0
		assert os.system('ssh {} "exit"'.format(MACHINES[-1])) is 0
	except:
		raise Exception("ABORT! SSH pipe is broken. ")


def get_machine():
	'''
	Check available machines, return an available one
	'''
	if args.debugging:
		return "LOCAL_MACHINE"

	for machine in MACHINES:
		try:
			process_running = "python scripts/main" in str(subprocess.check_output('ssh ' + machine + ' "ps aux"', shell=True))
		except:
			process_running = True

		if process_running:
			pass
		else:
			print ("This machine is available: ",machine)
			return machine
	return False


def launch_job(job_id, machine='machine_0'):

	cmd = ['python', 'scripts/main.py',
					'--results_path={}/{}.summary'.format(args.results_path, job_id)]

	try:
		assert not os.path.exists('{}/{}.log'.format(args.log_dir, job_id)) or args.rerun_experiments
	except:
		print(RESULTS_PATH_APPEAR_ERR.format(job_id))
		return 0

	cmd = cmd + ['>> {}/{}.log'.format(args.log_dir, job_id)]
	cmd = ' '.join(cmd)

	if not args.debugging:
		cmd = 'source /virtualenv/bin/activate && cd  /home/user/wdr/ && (nohup {} &>>results/{}.nohup.out &) &&exit'.format(cmd, job_id)
		cmd = 'ssh {} "{}" '.format(machine, cmd)
		test_ssh()

	# This is where the job is deployed using ssh
	os.system(cmd)
	print (SUCESSFUL_SCHEDULE_JOB_STR.format(job_id, machine))


def run_queue():

	test_ssh()
	try:
		os.mkdir(args.log_dir)
		os.mkdir(args.results_path)
	except:
		pass

	while True:
		with open('waiting_list.txt','r') as f:
			worker = f.readline().strip()
		if len(worker)==0:
			break
		machine = get_machine()
		if machine:
			launch_job(worker, machine=machine)
			os.system('sed -i "" "1d" waiting_list.txt')
			time.sleep(SLEEP_TIME_SCHEDULED)
		else:
			print('No available machines found.')
	sys.exit(0)

def check_queue():

	try:
		test_ssh()
		machine_used = sum(["python scripts/main" in str(subprocess.check_output('ssh ' + machine + ' "ps aux"', shell=True)) for machine in MACHINES])
		machine_free = [machine for machine in MACHINES if "python scripts/main" not in str(subprocess.check_output('ssh ' + machine + ' "ps aux"', shell=True))]
	except:
		machine_used = 0
		machine_free = []

	try:
		with open('waiting_list.txt', 'r') as f:
			job_waiting = len(f.readlines())
	except:
		job_waiting = []
	print ("There are {} machine used at the moment and {} job waiting in the queue.".format(machine_used, job_waiting))
	print ("These machines are currently not running jobs: ", machine_free)
	sys.exit(0)

def drop_queue():
	drop_safe = input("Are you sure you wanna drop the current queue?")
	if drop_safe=='Y' or drop_safe=='y' or drop_safe=='True':
		try:
			os.remove('waiting_list.txt')
		except:
			print('No waiting list detected. Nothing happened.')
	else:
		print("Need confirmation to proceed. (Use Y or True)")

	sys.exit(0)

def kill_queue():
	kill_safe = input("Are you sure you wanna kill the current runnning jobs?")
	if kill_safe=='Y' or kill_safe=='y' or kill_safe=='True':
		for machine in MACHINES:
			print('Killing for machine {}'.format(machine))
			kill_cmd = 'ssh ' + machine + ' "echo \'KILLING\' && pkill -9 python"'
			if args.debugging:
				print("DEBUGGING...\n", kill_cmd)
			else:
				test_ssh()
				os.system(kill_cmd)
	else:
		print("Need confirmation to proceed. (Use Y or True)")
	sys.exit(0)

def parse_config(args):
	assert args.experiment_config_path is not None, \
		"Experiment config path is required! Otherwise please specify --[cmd]_queue."

	assert os.path.exists(args.experiment_config_path), \
		CONFIG_NOT_FOUND_MSG.format("master", args.experiment_config_path)

	experiment_config = json.load(open(args.experiment_config_path, 'r'))
	job_list, experiment_axies = parsing.parse_dispatcher_config(experiment_config)

	if args.shuffle_experiment_order:
		random.shuffle(job_list)

	workers = [parsing.md5(job) for job in job_list]
	print("Schduling {} jobs!".format(len(job_list)))
	return job_list, workers


def prepare_results_path(args):
	if not os.path.exists(args.results_path):
		os.mkdir(args.results_path)
	results_path = args.results_path + '/master.{}.exp'.format(parsing.md5(''.join(args.job_list)))
	with open(results_path,'w') as out_file:
		out_file.write("worker_id, flag_string\n")
		for worker, job in zip(args.workers, args.job_list):
			out_file.write("{}, {}\n".format(worker, job))

if __name__ == "__main__":

	if args.check_queue:
		check_queue()

	if args.refresh_queue:
		run_queue()

	if args.drop_queue:
		drop_queue()

	if args.kill_queue:
		kill_queue()

	args.job_list, args.workers = parse_config(args)

	prepare_results_path(args)

	with open('waiting_list.txt','a') as f:
		for worker in args.workers:
			f.write('{}\n'.format(worker))
		print('Waiting list has been updated!')

	with open('waiting_list.txt','r') as f:
		launch_workers = len(f.readlines())==len(args.workers) or args.refresh_queue
		if not launch_workers:
			print('Warning! Another scheduler is detected. Abort it and clean the waitlist or use --refresh_queue to override!')
			sys.exit(1)

	print(SUCESSFUL_SCHEDULE_FINISH_STR)

	if launch_workers:
		run_queue()
