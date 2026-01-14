; Removes bluesuit ability

arch snes.cpu
lorom

org $91D35C ; replace branch check of samus running momentum flag
nop 
nop