lorom
arch snes.cpu

; disable the SRAM check as it 1) conflicts with the RTA timer, 2) risks destroying save files if the system 
; is reset during the check.
org $808000
    dw $FFFF
