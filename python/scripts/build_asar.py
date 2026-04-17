import argparse, os, pathlib, subprocess

os.chdir(pathlib.Path(__file__).parent)

argparser = argparse.ArgumentParser(description = 'Build asar')
argparser.add_argument('-j', action = 'store_true', help = 'Enable build parallelisation')
argparser.add_argument('--clean', action = 'store_true', help = 'Do a clean build')
argparser.add_argument('--reset', action = 'store_true', help = 'Hard reset asar submodule first')
args = argparser.parse_args()

if args.reset:
    subprocess.run(['git', '-C', '../../asar', 'reset', '--hard'], check = True)

if args.clean:
    subprocess.run(['git', '-C', '../../asar', 'clean', '-dfx'], check = True)

# Specify release here for single-config generators (ignored by multi-config generators)
subprocess.run(['cmake', '../../asar/src', '-B', '../../asar/build', '-DCMAKE_BUILD_TYPE=Release'], check = True)

# Specify release here for multi-config generators (ignored by single-config generators)
buildCommand = ['cmake', '--build', '../../asar/build', '--config', 'Release']
if args.j:
    buildCommand += ['-j']

subprocess.run(buildCommand, check = True)

# For multi-config generators, move the output binary to the location single-config generators use
directoryPath = pathlib.Path('../../asar/build/asar/bin/Release/')
if directoryPath.exists():
    asarPath = directoryPath / 'asar'
    if not asarPath.exists():
        asarPath = directoryPath / 'asar.exe'
        if not asarPath.exists():
            raise RuntimeError('Could not resolve path to asar binary')
    
    asarPath.move_into('../../asar/build/asar/bin/')
