from subprocess import run
import argparse 
import os
# from sample_videos import main

parser = argparse.ArgumentParser()
parser.add_argument('--pt_input', type=str, help='Path of directory with the video directories for test')
parser.add_argument('--pt_output', type=str, help='Path of directory where to deposit predictions')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')
parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for dataloading')

args = parser.parse_args()

print(args)
#assert os.path.isdir(args.pt_input), 'The test directory {} does not exist'.format(args.pt_input)


if os.getenv('CSV_PATH') is None or os.getenv('CSV_PATH') == '':
    os.environ['CSV_PATH'] = os.path.join(args.pt_output)

if os.getenv('TEST_DIR') is None or os.getenv('TEST_DIR') == '':
    os.environ['TEST_DIR'] = os.path.abspath(args.pt_input)

if os.getenv('BATCH') is None or os.getenv('BATCH') == '':
    os.environ['BATCH'] = str(args.batch_size)

if os.getenv('WORKERS') is None or os.getenv('WORKERS') == '':
    os.environ['WORKERS'] = str(args.num_workers)


# assert len(tasks)<=3, 'Incorrect number of tasks inputed {}'.format(tasks)
run(['sh','inference.sh'])