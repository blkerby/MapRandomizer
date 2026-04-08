import argparse, subprocess

argparser = argparse.ArgumentParser(description = 'Build asar')
argparser.add_argument('--reset', action = 'store_true', help = 'Hard reset asar submodule first')
args = argparser.parse_args()

if args.reset:
    subprocess.run(['git', '-C', '../../asar', 'reset', '--hard'], check = True)

subprocess.run(['cmake', '../../asar/src', '-B', '../../asar/build'], check = True)
subprocess.run(['cmake', '--build', '../../asar/build', '--config', 'Release'], check = True)
