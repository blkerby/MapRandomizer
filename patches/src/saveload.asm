; Based on Scyzer's replacement of saving/loading, to fix map saving.
; From https://metroidconstruction.com/SMMM/Scyzer_map_fix_saveload.zip
arch snes.cpu
LoRom


org $819A47		;Fix File Copy for the new SRAM files
	LDA.l SRAMAddressTable,X : Skip 7 : LDA.l SRAMAddressTable,X : Skip 11 : CPY #$0A00
org $819CAE		;Fix File Clear for the new SRAM files
	LDA.l SRAMAddressTable,X : Skip 12 : CPY #$0A00
	
org $818000
	JMP SaveGame
org $818085
	JMP LoadGame
org $81EF20
SRAMAddressTable:
	DW $0010,$0A10,$1410
CheckSumAdd: CLC : ADC $14 : INC A : STA $14 : RTS

SaveGame: PHP : REP #$30 : PHB : PHX : PHY
	PEA $7E7E : PLB : PLB : STZ $14
    TAX         ; store save slot (0..2) in X
	LDA $D916
	CMP $078B   ; Are we at the same save station index as the last save on this slot?
	BNE .notsame
	LDA $D918
	CMP $079F   ; Are we in the same (vanilla) area as the last save on this slot?
	BEQ .noinc  ; Yes, which means we're at the same save station, so skip incrementing the slot.
.notsame
    TXA         ; load save slot (0..2) back into A
	AND #$0003
	INC
	CMP #$0003
	BNE .nowrap
	LDA #$0000  ; we were already on the third slot, so wrap back to the first one
.nowrap
    STA $0952   ; update the save slot
    STA $701FEC ; write to SRAM
    TAX
    EOR #$FFFF
    STA $701FEE ; write complement to SRAM for validation
.noinc
    TXA
	ASL A : STA $12
	LDA $1F5B : INC A : XBA : TAX : LDY #$00FE
SaveMap: LDA $07F7,Y : STA $CD50,X : DEX : DEX : DEY : DEY : BPL SaveMap		;Saves the current map
	LDY #$005E
SaveItems: LDA $09A2,Y : STA $D7C0,Y : DEY : DEY : BPL SaveItems				;Saves current equipment	
	LDA $078B : STA $D916		;Current save for the area
	LDA $079F : STA $D918		;Current Area
	LDA $1F5B : STA $7FFE00     ;Current Map-area
	LDA $1F5D : STA $7FFE04     ;Item set before escape (note that $7FFE04 gets wiped during the escape sequence)
	LDX $12
	LDA.l SRAMAddressTable,X : TAX : LDY #$0000		;How much data to save for items and event bits
SaveSRAMItems: LDA $D7C0,Y : STA $700000,X : JSR CheckSumAdd : INX : INX : INY : INY : CPY #$0160 : BNE SaveSRAMItems	
	LDY #$06FE		;How much data to save for maps
SaveSRAMMaps: LDA $CD52,Y : STA $700000,X : INX : INX : DEY : DEY : BPL SaveSRAMMaps	
	PEA $7F7F : PLB : PLB : LDY #$00FE		;How much extra data to save per save
SaveSRAMExtra: LDA $FE00,Y : STA $700000,X : INX : INX : DEY : DEY : BPL SaveSRAMExtra
	LDY #$00FE : LDX #$1E10					;How much extra data to save globally (affects all saves)
SaveSRAMExtraA: LDA $FF00,Y : STA $700000,X : INX : INX : DEY : DEY : BPL SaveSRAMExtraA
SaveChecksum: LDX $12 : LDA $14 : STA $700000,X : STA $701FF0,X : EOR #$FFFF : STA $700008,X : STA $701FF8,X
EndSaveGame: PLY : PLX : PLB : PLP : RTL

LoadGame: PHP : REP #$30 : PHB : PHX : PHY
	PEA $7E7E : PLB : PLB : STZ $14 : AND #$0003 : ASL A : STA $12
	TAX : LDA.l SRAMAddressTable,X : STA $16 : TAX : LDY #$0000		;How much data to load for items and event bits
LoadSRAMItems: LDA $700000,X : STA $D7C0,Y : JSR CheckSumAdd : INX : INX : INY : INY : CPY #$0160 : BNE LoadSRAMItems	
	LDY #$06FE		;How much data to load for maps
LoadSRAMMaps: LDA $700000,X : STA $CD52,Y : INX : INX : DEY : DEY : BPL LoadSRAMMaps
	PEA $7F7F : PLB : PLB : LDY #$00FE		;How much extra data to load per save
LoadSRAMExtra: LDA $700000,X : STA $FE00,Y : INX : INX : DEY : DEY : BPL LoadSRAMExtra
	LDY #$00FE : LDX #$1E10					;How much extra data to load globally (affects all saves)
LoadSRAMExtraA: LDA $700000,X : STA $FF00,Y : INX : INX : DEY : DEY : BPL LoadSRAMExtraA
LoadCheckSum: LDX $12 : LDA $700000,X : CMP $14 : BNE $0B : EOR #$FFFF : CMP $14 : BNE $02 : BRA LoadSRAM
	LDA $14 : CMP $701FF0,X : BNE SetupClearSRAM : EOR #$FFFF : CMP $701FF8,X : BNE SetupClearSRAM : BRA LoadSRAM
LoadSRAM: PEA $7E7E : PLB : PLB : LDY #$005E
LoadItems: LDA $D7C0,Y : STA $09A2,Y : DEY : DEY : BPL LoadItems		;Loads current equipment	
	LDA $D916 : STA $078B		;Current save for the area
	LDA $D918 : STA $079F		;Current Area
    LDA $7FFE00 : STA $1F5B     ;Current Map-area
	LDA $7FFE04 : STA $1F5D     ;Item set before escape
	PLY : PLX : PLB : PLP : CLC : RTL
SetupClearSRAM: LDX $16 : LDY #$09FE : LDA #$0000
ClearSRAM: STA $700000,X : INX : INX : DEY : DEY : BPL ClearSRAM
    LDA #$0000 : STA $7E078B : STA $7E079F
	PLY : PLX : PLB : PLP : SEC : RTL

    warnpc $81f100
