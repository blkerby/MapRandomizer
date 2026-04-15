import dump_rom_map
import argparse, glob, os, pathlib, subprocess

argparser = argparse.ArgumentParser(description = 'Build patches')
argparser.add_argument('-i', type = str, help = 'ASM file to build. If omitted, all patches are built')
argparser.add_argument('-e', '--fatal-errors', action = 'store_true', help = 'Stop if any patch fails to build')
argparser.add_argument('-q', '--quiet', action = 'store_true', help = 'Suppress asar output')
argparser.add_argument('-v', '--verbose', action = 'store_true', help = 'Print asar invocations')
argparser.add_argument('--no-filter', action = 'store_true', help = 'Enable printing deprecation notices')
argparser.add_argument('--no-summary', action = 'store_true', help = 'Disable printing the summary of results')
args = argparser.parse_args()

successes : list[tuple[str, str]] = []
failures : list[tuple[str, str]] = []
conflicts : list[tuple[str, int, dump_rom_map.Interval, dump_rom_map.Interval, str]] = []

apply = lambda t, f: f(*t)

ignoredPatches = [
    'hud_expansion_opaque'
]

def filterDeprecations(text : str):
    return '\n'.join(filter(lambda line: "DEPRECATION NOTIFICATION" not in line, text.split('\n')))
    
def checkForConflicts(outputFilepath : str, inputFilepath : str):
    global conflicts
    
    filename = pathlib.Path(inputFilepath).stem
    
    if filename in ignoredPatches:
        return
    
    newBankedChanges = dump_rom_map.BankedChanges()
    with open(outputFilepath, 'rb') as f:
        newBankedChanges.addFromPatch(filename, f)
    
    for (bank, newChanges) in newBankedChanges.changesMap.items():
        for newInterval in newChanges:
            changes = bankedChanges.changesMap[bank]
            for (interval, label) in changes.items():
                if (interval.begin <= newInterval.end and newInterval.begin <= interval.end
                    and not label.startswith(f'; {filename}') and f', {filename}' not in label
                    and all(not label.startswith(f'; {filename}') and f', {filename}' not in label for filename in ignoredPatches)
                ):
                    conflicts += [(inputFilepath, bank, newInterval, interval, label)]

def runAsar(outputFilepath : str, inputFilepath : str):
    global successes, failures
    
    try:
        invocation = [
            '../../asar/build/asar/bin/asar', 
            '--fix-checksum=off', 
            '--no-title-check', 
            '--disable-read', 
            '--ips', outputFilepath, 
            inputFilepath, 
            'dummy.smc'
        ]
        if args.verbose:
            print(' '.join(invocation))
        
        output = subprocess.run(invocation, capture_output = True, text = True)
    except Exception as e:
        failures += [(inputFilepath, f'Asar invocation failed: {e}')]
        return
    
    outputText = f'{output.stdout.strip()}\n{output.stderr.strip()}'.strip()
    if not args.quiet and outputText:
        if not args.no_filter:
            outputText = filterDeprecations(outputText)
            
        outputText = outputText.strip()
        if outputText:
            print(f'{inputFilepath}:')
            print(outputText)
            print()
    
    try:
        output.check_returncode()
    except subprocess.CalledProcessError as e:
        failures += [(inputFilepath, output.stderr)]
        return
    
    successes += [(outputFilepath, inputFilepath)]
    checkForConflicts(outputFilepath, inputFilepath)

def assemblePatches():
    if args.i:
        outputFilepath = pathlib.Path(args.i).with_suffix('.ips')
        runAsar(str(outputFilepath), args.i)
        return
        
    for inputFilepath in glob.iglob('../../patches/src/**/*.asm', recursive = True):
        if args.fatal_errors and failures:
            break
        
        outputFilepath = pathlib.Path('../../patches/ips').joinpath(*pathlib.Path(inputFilepath).parts[4:]).with_suffix('.ips')
        runAsar(str(outputFilepath), str(inputFilepath))

    for inputFilepath in glob.iglob('../../Mosaic/Projects/Base/ASM/**/*.asm', recursive = True):
        if args.fatal_errors and failures:
            break
        
        outputFilepath = pathlib.Path('../../patches/ips').joinpath(*pathlib.Path(inputFilepath).parts[6:]).with_suffix('.ips')
        runAsar(str(outputFilepath), str(inputFilepath))

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
            stderr = filterDeprecations(stderr)
            print(f'Failed to assemble: {failure}')
            print(stderr)
            print()
    
    if conflicts:
        print()
        print('Found conflicts with the ROM map:')
        for (filepath, bank, newInterval, interval, label) in conflicts:
            print(f'    ${bank:X}:{newInterval.begin:X}-{newInterval.end:X} from "{filepath}" conflicts with ${bank:X}:{interval.begin:X}-{interval.end:X} "{label}"')

os.chdir(pathlib.Path(__file__).parent)
bankedChanges = dump_rom_map.collectBankedChanges()
assemblePatches()
if not args.no_summary:
    printResults()
