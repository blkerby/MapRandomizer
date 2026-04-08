import dataclasses, glob, io, pathlib, re

hex2snes = lambda address: address << 1 & 0xFF0000 | address & 0xFFFF | 0x808000

@dataclasses.dataclass(order = True, frozen = True)
class Interval: # Closed interval
    begin : int
    end : int

class BankedChanges:
    def __init__(self):
        self.changesMap : dict[int, dict[Interval, str]] = {}
    
    def _getIntervalsFromPatch(self, patch : io.BufferedReader):
        if patch.read(5) != b'PATCH':
            raise RuntimeError('File does not begin with PATCH')

        intervals : list[Interval] = []
        while True:
            data = patch.read(3)
            if len(data) < 3:
                raise RuntimeError("Could not read an address")
                
            address = int.from_bytes(data, 'big')
            if address == 0x454F46:
                # 'EOF'
                break
            
            size = int.from_bytes(patch.read(2), 'big')
            if size != 0:
                patch.read(size)
            else:
                # RLE
                size = int.from_bytes(patch.read(2), 'big')
                int.from_bytes(patch.read(1), 'big')
                
            intervals += [Interval(address, address + size - 1)]
        
        return intervals
    
    def _mergeAdjacentIntervals(self, intervals : list[Interval]):
        mergedIntervals = []
        begin = intervals[0].begin
        for (interval, nextInterval) in zip(intervals[:-1], intervals[1:]):
            if interval.end + 1 != nextInterval.begin:
                mergedIntervals += [Interval(begin, interval.end)]
                begin = nextInterval.begin
        
        mergedIntervals += [Interval(begin, intervals[-1].end)]
        return mergedIntervals
    
    def addFromPatch(self, filename : str, patch : io.BufferedReader):
        patchIntervals = self._getIntervalsFromPatch(patch)
        if not patchIntervals:
            # Some ASM files make no changes, e.g. constants.asm
            return
            
        intervals = self._mergeAdjacentIntervals(patchIntervals)

        for interval in intervals:
            address = interval.begin
            endAddress = interval.end
            # Loop required for changes that span multiple banks
            while True:
                bank = hex2snes(address) >> 0x10
                endBank = hex2snes(endAddress) >> 0x10
                
                if endBank == bank:
                    interval = Interval(address & 0xFFFF | 0x8000, endAddress & 0xFFFF | 0x8000)
                else:
                    interval = Interval(address & 0xFFFF | 0x8000, 0xFFFF)
                    intervalSize = interval.end - interval.begin + 1
                    address += intervalSize
                
                if bank not in self.changesMap:
                    self.changesMap[bank] = {}
                
                if interval not in self.changesMap[bank]:
                    self.changesMap[bank][interval] = f'; {filename}'
                else:
                    self.changesMap[bank][interval] += f', {filename}'
                
                if endBank == bank:
                    break
    
    def addFromRomMap(self, bank : int, file : io.TextIOWrapper):
        label = ''
        interval = None
        for line in file:
            if ': ;' in line:
                continue
            
            match = re.match(r'([\dA-F]+) - ([\dA-F]+): (.+)', line, re.IGNORECASE)
            if not match:
                label += f'\n{line.rstrip()}'
                continue
                
            if interval is not None:
                if bank not in self.changesMap:
                    self.changesMap[bank] = {}
                
                if interval in self.changesMap[bank]:
                    raise RuntimeError(f'Found comment with the same address interval as a patch change: ${bank:X}:{interval.begin:X}-{interval.end:X}')
                    
                self.changesMap[bank][interval] = label
            
            interval = Interval(int(match[1], 0x10), int(match[2], 0x10))
            label = match[3]
        
        if interval is not None:
            if bank not in self.changesMap:
                self.changesMap[bank] = {}
            
            if interval in self.changesMap[bank]:
                raise RuntimeError(f'Found comment with the same address interval as a patch change: ${bank:X}:{interval.begin:X}-{interval.end:X}')
                
            self.changesMap[bank][interval] = label
    
    def addFromVanillaHooks(self, file : io.TextIOWrapper):
        def addNote(bank, interval, label):
            if interval is not None:
                if bank not in self.changesMap:
                    self.changesMap[bank] = {}
                
                if interval in self.changesMap[bank]:
                    raise RuntimeError(f'Found comment with the same address interval as a patch change: ${bank:X}:{interval.begin:X}-{interval.end:X}')
                    
                self.changesMap[bank][interval] = label
        
        label = ''
        interval = None
        bank = None
        for line in file:
            if ': ;' in line:
                continue
            
            match = re.match(r'\[BANK ([\dA-F]+)\]', line, re.IGNORECASE)
            if match:
                addNote(bank, interval, label)
                label = ''
                interval = None
                bank = int(match[1], 0x10)
                continue
            
            match = re.match(r'([\dA-F]+) - ([\dA-F]+): (.+)', line, re.IGNORECASE)
            if not match:
                label += f'\n{line.rstrip()}'
                continue
                
            addNote(bank, interval, label)
            interval = Interval(int(match[1], 0x10), int(match[2], 0x10))
            label = match[3]
        
        addNote(bank, interval, label)

freespaceMap = {
    0x80: [Interval(0xCD8E, 0xFFBF)],
    0x81: [Interval(0xEF1A, 0xFFFF)], # Not accounting for Genji
    0x82: [Interval(0xF70F, 0xFFFF)],
    0x83: [Interval(0xAD66, 0xFFFF)],
    0x84: [Interval(0xEFD3, 0xFFFF)],
    0x85: [Interval(0x9643, 0xFFFF)],
    0x86: [Interval(0xF4A6, 0xFFFF)],
    0x87: [Interval(0xC964, 0xFFFF)],
    0x88: [Interval(0xA206, 0xA265), Interval(0xEE32, 0xFFFF)],
    0x89: [Interval(0xAEFD, 0xFFFF)],
    0x8A: [Interval(0xE980, 0xFFFF)],
    0x8B: [Interval(0xF760, 0xFFFF)],
    0x8C: [Interval(0xF3E9, 0xFFFF)],
    0x8D: [Interval(0xFFF1, 0xFFFF)],
    0x8E: [Interval(0xE600, 0xFFFF)],
    0x8F: [Interval(0xE99B, 0xFFFF)],
    0x90: [Interval(0xF63A, 0xFFFF)],
    0x91: [Interval(0xFFEE, 0xFFFF)],
    0x92: [Interval(0xEDF4, 0xFFFF)],
    0x93: [Interval(0xF61D, 0xFFFF)],
    0x94: [Interval(0xB19F, 0xC7FF), Interval(0xDC00, 0xDFFF)],
    0x99: [Interval(0xEE21, 0xFFFF)],
    0x9A: [Interval(0xFC20, 0xFFFF)],
    0x9B: [Interval(0xCBFB, 0xDFFF), Interval(0xFDA0, 0xFFFF)],
    0x9C: [Interval(0xFA80, 0xFFFF)],
    0x9D: [Interval(0xF780, 0xFFFF)],
    0x9E: [Interval(0xF6C0, 0xFFFF)],
    0x9F: [Interval(0xF740, 0xFFFF)],
    0xA0: [Interval(0xF7D3, 0xFFFF)],
    0xA1: [Interval(0xEBD1, 0xFFFF)],
    0xA2: [Interval(0xF498, 0xFFFF)],
    0xA3: [Interval(0xF311, 0xFFFF)],
    0xA4: [Interval(0xF6C0, 0xFFFF)],
    0xA5: [Interval(0xF95A, 0xFFFF)],
    0xA6: [Interval(0xFEBC, 0xFFFF)],
    0xA7: [Interval(0xFF82, 0xFFFF)],
    0xA8: [Interval(0xF9BE, 0xFFFF)],
    0xA9: [Interval(0xFB70, 0xFFFF)],
    0xAA: [Interval(0xF7D3, 0xFFFF)],
    0xAB: [Interval(0xF800, 0xFFFF)],
    0xAC: [Interval(0xEE00, 0xFFFF)],
    0xAD: [Interval(0xF444, 0xFFFF)],
    0xAE: [Interval(0xFD20, 0xFFFF)],
    0xAF: [Interval(0xEC00, 0xFFFF)],
    0xB0: [Interval(0xEE00, 0xFFFF)],
    # $B1 has no free space
    0xB2: [Interval(0xFEAA, 0xFFFF)],
    0xB3: [Interval(0xED77, 0xFFFF)],
    0xB4: [Interval(0xF4B8, 0xFFFF)],
    0xB5: [Interval(0xF000, 0xFFFF)],
    0xB6: [Interval(0xF200, 0xFFFF)],
    0xB7: [Interval(0xFD00, 0xFFFF)],
    0xB8: [Interval(0x8000, 0xFFFF)], # Unused bank
    # $B9..$CD have no free space (compressed data)
    0xCE: [Interval(0xB22E, 0xFFFF)],
    # $CF..DD have no free space (music data)
    0xDE: [Interval(0xD1C0, 0xFFFF)],
    0xDF: [Interval(0x8000, 0xFFFF)] # Unused bank
    # $E0+ is omitted from this table
}

def collectBankedChanges():
    bankedChanges = BankedChanges()
    for filepath in glob.iglob('../../patches/ips/**/*.ips', recursive = True):
        filename = pathlib.Path(filepath).stem + '.asm'
        with open(filepath, 'rb') as f:
            bankedChanges.addFromPatch(filename, f)
        
    for bank in range(0x80, 0x100):
        path = pathlib.Path(f'../../patches/rom_map/Bank {bank:X}.txt')
        if not path.exists():
            continue
            
        with open(path, 'r') as f:
            bankedChanges.addFromRomMap(bank, f)
    
    with open('../../patches/rom_map/vanilla_hooks.txt', 'r') as f:
        bankedChanges.addFromVanillaHooks(f)
    
    return bankedChanges

def writeBankFiles(bankedChanges : BankedChanges):
    with open('../../patches/rom_map/vanilla_hooks.txt', 'w') as hooksFile:
        for (bank, changesMap) in sorted(bankedChanges.changesMap.items()):
            bankText = ''
            hooksText = ''
            
            for (interval, label) in sorted(changesMap.items()):
                label = label.rstrip()
                isHook = True
                if bank >= 0xE0:
                    isHook = False
                elif bank in freespaceMap:
                    for freespace in freespaceMap[bank]:
                        if freespace.begin <= interval.end and interval.end <= freespace.end:
                            isHook = False
                            break
                
                if isHook:
                    hooksText += f'{interval.begin:X} - {interval.end:X}: {label}\n'
                else:
                    bankText += f'{interval.begin:X} - {interval.end:X}: {label}\n'
            
            if bankText:
                with open(f'../../patches/rom_map/Bank {bank:X}.txt', 'w') as bankFile:
                    bankFile.write(bankText)
            
            if hooksText:
                hooksFile.write(f'[BANK {bank:X}]\n')
                hooksFile.write(hooksText)
                hooksFile.write('\n')

collectBankedChanges()
writeBankFiles(collectBankedChanges())
