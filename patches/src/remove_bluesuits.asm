; Removes a spikesuit state from samus 

arch snes.cpu
lorom


org $91DE5C ; replace the instruction that allows Samus to gain a bluesuit (speedbooster is only cancelled if running flag is 0)
		nop
		nop