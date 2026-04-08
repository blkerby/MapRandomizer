import glob, os, pathlib, subprocess

successes : list[tuple[str, str]] = []
failures : list[tuple[str, str]] = []

apply = lambda t, f: f(*t)

def runAsar(outputFilepath, inputFilepath):
    global successes, failures
    
    try:
        output = subprocess.run(
            [
                '../../asar/build/asar/bin/Release/asar', 
                '--fix-checksum=off', 
                '--no-title-check', 
                '--disable-read', 
                '--ips', outputFilepath, 
                inputFilepath, 
                'dummy.smc'
            ],
            capture_output = True,
            text = True
        )
    except Exception as e:
        failures += [(inputFilepath, f'Asar invocation failed: {e}')]
        return
    
    try:
        output.check_returncode()
    except subprocess.CalledProcessError as e:
        failures += [(inputFilepath, output.stderr)]
        return
    
    successes += [(outputFilepath, inputFilepath)]

def assemblePatches():
    for inputFilepath in glob.iglob('../../patches/src/**/*.asm', recursive = True):
        outputFilepath = pathlib.Path('../../patches/ips').joinpath(*pathlib.Path(inputFilepath).parts[4:]).with_suffix('.ips')
        runAsar(outputFilepath, inputFilepath)

    for inputFilepath in glob.iglob('../../Mosaic/Projects/Base/ASM/**/*.asm', recursive = True):
        outputFilepath = pathlib.Path('../../patches/ips').joinpath(*pathlib.Path(inputFilepath).parts[6:]).with_suffix('.ips')
        runAsar(outputFilepath, inputFilepath)

    pathlib.Path('dummy.smc').unlink(missing_ok = True)

def printResults():
    if successes:
        print()
        inputWidth = max(len(fp) for (_, fp) in successes)
        for (outputFilepath, inputFilepath) in successes:
            inputFilepath = f'{{:{inputWidth}}}'.format(inputFilepath)
            print(f'Assembled {inputFilepath} -> {outputFilepath}')

    if failures:
        print()
        for (failure, stderr) in failures:
            stderr = ''.join(filter(lambda line: "DEPRECATION NOTIFICATION" not in line, stderr.split('\n')))
            print(f'Failed to assemble: {failure}')
            print(stderr)
            print()

assemblePatches()
printResults()
