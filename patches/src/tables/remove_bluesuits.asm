; Removes a spikesuit state from samus 

arch snes.cpu
lorom


org $91DE59 ; replace the instruction that allows Samus to gain a bluesuit (speedbooster is only cancelled if running flag is 0)
		STZ $0B3E
		LDA $0B3C
		BEQ $2F
		STZ $0B3C
		