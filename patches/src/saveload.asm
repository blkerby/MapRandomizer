; Based on Scyzer's replacement of saving/loading, to fix map saving.
; From https://metroidconstruction.com/SMMM/Scyzer_map_fix_saveload.zip
arch snes.cpu
LoRom

!bank_8f_free_space_start = $8ffe40
!bank_8f_free_space_end = $8ffe80
!bank_8f_free_space2_start = $8fe99b
!bank_8f_free_space2_end = $8feb00
!bank_81_free_space_start = $81f1c0
!bank_81_free_space_end = $81f200

!seed_value_0 = $dfff00
!seed_value_1 = $dfff02

incsrc "constants.asm"

org $819A47		;Fix File Copy for the new SRAM files
	LDA.l SRAMAddressTable,X : Skip 7 : LDA.l SRAMAddressTable,X : Skip 11 : CPY #$0A00
org $819CAE		;Fix File Clear for the new SRAM files
	LDA.l SRAMAddressTable,X : Skip 12 : CPY #$0A00
    
org $819AE8
    JSR hook_file_copy
    
org $819D14
    JSR hook_file_clear
    
org $818000
	JMP SaveGame
org $818085
	JMP LoadGame
org $81EF20
SRAMAddressTable:
	DW $0010,$0A10,$1410

SaveGame: PHP : REP #$30 : PHB : PHX : PHY
	PEA $7E7E : PLB : PLB

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
	LDA $1F5B : INC A : XBA : TAX
SaveStoredFlashSuit:
	LDY #$0000
	LDA $0A68  ; check that the special Samus palette timer is non-zero (indicating the ability to spark)
	BEQ .skip
	LDA $0ACC  ; check that special Samus palette type is not a regular shinecharge
	CMP #$0001
	BEQ .skip
	STA $7EFE92
	INY        ; Y <- 1
.skip:
SaveStoredBlueuit:
	LDA $0B3E
	CMP #$0400 ; check that the dash counter is 4, with 0 speed boost timer
	BNE .finish
	TYA
	ORA #$0002
	TAY		    ; Y |= 2 (set blue suit bit)
.finish:
	TYA
	STA $7EFE90
SaveMap: 
	LDY #$00FE
.loop:
	LDA $07F7,Y : STA $CD50,X : DEX : DEX : DEY : DEY : BPL .loop		;Saves the current map
	LDY #$005E
SaveItems: LDA $09A2,Y : STA $D7C0,Y : DEY : DEY : BPL SaveItems				;Saves current equipment	
	LDA $078B : STA $D916		;Current save for the area
	LDA $079F : STA $D918		;Current Area
	LDA $1F5B : STA $7EFE00     ;Current Map-area
	LDA $1F5D : STA $7EFE04     ;Item set before escape
	LDX $12
	LDA.l SRAMAddressTable,X : TAX : LDY #$0000		;How much data to save for items and event bits
SaveSRAMItems: LDA $D7C0,Y : STA $700000,X : INX : INX : INY : INY : CPY #$0160 : BNE SaveSRAMItems	
	LDY #$06FE		;How much data to save for maps
SaveSRAMMaps: LDA $CD52,Y : STA $700000,X : INX : INX : DEY : DEY : BPL SaveSRAMMaps	
	PEA $7E7E : PLB : PLB : LDY #$00FE		;How much extra data to save per save
SaveSRAMExtra: LDA $FE00,Y : STA $700000,X : INX : INX : DEY : DEY : BPL SaveSRAMExtra
SaveSeed:
	LDX $12
	LDA !seed_value_0 : STA $700000, X
	LDA !seed_value_1 : STA $700008, X
SaveAreaMapCoord:
    JSL set_save_markers
EndSaveGame: PLY : PLX : PLB : PLP : RTL

LoadGame: PHP : REP #$30 : PHB : PHX : PHY
	PEA $7E7E : PLB : PLB
	
	AND #$0003 : ASL A : STA $12
	TAX : LDA.l SRAMAddressTable,X : STA $16 : TAX : LDY #$0000		;How much data to load for items and event bits
LoadSRAMItems: LDA $700000,X : STA $D7C0,Y : INX : INX : INY : INY : CPY #$0160 : BNE LoadSRAMItems	
	LDY #$06FE		;How much data to load for maps
LoadSRAMMaps: LDA $700000,X : STA $CD52,Y : INX : INX : DEY : DEY : BPL LoadSRAMMaps
	PEA $7E7E : PLB : PLB : LDY #$00FE		;How much extra data to load per save
LoadSRAMExtra: LDA $700000,X : STA $FE00,Y : INX : INX : DEY : DEY : BPL LoadSRAMExtra
LoadSeed:
	LDX $12
	LDA $700000, X : CMP !seed_value_0 : BNE SetupClearSRAM
	LDA $700008, X : CMP !seed_value_1 : BNE SetupClearSRAM
LoadSRAM: PEA $7E7E : PLB : PLB : LDY #$005E
LoadItems: LDA $D7C0,Y : STA $09A2,Y : DEY : DEY : BPL LoadItems		;Loads current equipment	
	LDA $D916 : STA $078B		;Current save for the area
	LDA $D918 : STA $079F		;Current Area
    LDA $7EFE00 : STA $1F5B     ;Current Map-area
	LDA $7EFE04 : STA $1F5D     ;Item set before escape
    LDA #$0000
    STA !last_samus_map_y  ; reset Samus map Y coordinate, to trigger minimap to update
    JSL set_marker_colors
    PLY : PLX : PLB : PLP : CLC : RTL
SetupClearSRAM: LDX $16 : LDY #$09FE : LDA #$0000
ClearSRAM: STA $700000,X : INX : INX : DEY : DEY : BPL ClearSRAM
    LDA #$0000 : STA $7E078B : STA $7E079F
	PLY : PLX : PLB : PLP : SEC : RTL

    warnpc $81f100

; hook north/south elevatube door ASM to check if it's spawn location before unlocking Samus
org $8fe301
    jsr fix_tube

org $8fe310
    jsr fix_tube

org !bank_8f_free_space_start
fix_tube:
    lda $0A44
    cmp #$E86A                    ; Samus appearance?
    bne .leave
    pla                           ; adjust stack
    rts

.leave
    lda #$0001                    ; replaced code
    rts

warnpc !bank_8f_free_space_end

;;; Save Marker patch
org !bank_8f_free_space2_start
set_save_markers:
    LDA $952                ; current save index
    ASL
    ASL
    TAX                     ; index*4

; adjust map coords for spawn point / ship
    LDA $0AF6
    AND #$FF00
    XBA
    CLC
    ADC $07A1
    PHA                     ; x
    
    LDA $0AFA
    XBA
    AND #$00FF
    CLC
    ADC $07A3
    PHA                     ; y
    
.check_save
    SEP #$20
    PHX                     ; save index
    LDA $702602
    CMP #$FF                ; check if all 3 areas are unset (= spawn)
    BNE .save_normal
    LDA $702606
    CMP #$FF
    BNE .save_normal
    LDA $70260A
    CMP #$FF
    BNE .save_normal
    JSL check_save_rooms    ; spawning in save room?
    BCS .save_normal
    LDA $80C4E1             ; spawn room == $91F8 (Landing Site)?
    CMP #$F8
    BNE .helmet
    LDA $80C4E2
    CMP #$91
    BNE .helmet
    LDA $80C4E7
    BEQ .save_normal        ; offset_x == 0 (ship)?
; possible sprite values:
; 8D = helmet
; 08 = yellow, 0E = orange, 0F = pink (save icons)
.helmet
    LDA #$8D                ; helmet icon
    BRA .write_icon
.save_normal
    LDA #$08                ; normal save
.write_icon
    PLX                     ; index
    STA $702603,X           ; icon
    LDA $1F5B
    STA $702602,X           ; area
    PLA
    STA $702605,X           ; y
    PLA
    PLA
    STA $702604,X           ; x
    PLA
    JSL set_marker_colors
    REP #$30
    RTL

; check current room against save stations
; set carry with result
check_save_rooms:
    PHP
    PHB
    PHK
    PLB
    REP #$30
    LDX #$0000
    
.read_lp
    LDA save_rooms,X
    BEQ .no_match
    CMP $79B
    BEQ .match
    INX : INX
    BRA .read_lp
.no_match
    PLB
    PLP
    CLC
    RTL
.match
    PLB
    PLP
    SEC
    RTL

save_rooms:
    dw $A184, $A201, $A22A, $A70B, $A734, $AAB5, $B0DD, $B167
    dw $B192, $B1BB, $B741, $93D5, $CE8A, $CED2, $D3DF, $D765
    dw $D81A, $DE23, $DF1B, $0000

; save icon logic:
;   - helmet represents spawn until 3rd save occurs; exception is spawn in save room
;   - yellow always represents most recent save; orange n-1 (next revert), pink n+1 (next overwrite)
set_marker_colors:
    PHP
    LDX #$0000
    TXY
; set relative colors
    SEP #$20
    LDA #$08            ; yellow color
    STA $2E
    LDA $952            ; save index
    CMP #$03            ; 0-2 valid; ignore file copy calls
    BCS .exit
    
.next_color_lp
    PHA                 ; index
    ASL
    ASL
    TAX                 ; index*4
    LDA $2E             ; color to use (08, 0E, 0F)
    PHA                 ; save current color
    CMP #$08
    BNE .is_orange
    CLC
    ADC #$06            ; orange
    BRA .write
.is_orange
    INC                 ; pink
.write
    STA $2E             ; next slot color
    LDA $702603,X
    CMP #$8D            ; helmet?
    BNE .not_helmet
    PLA                 ; fix stack
    BRA .skip_write
.not_helmet
    PLA                 ; current color
    STA $702603,X
.skip_write
    PLA                 ; index
    BNE .next_index
    LDA #$03            ; wrap
.next_index
    DEC
    INY
    CPY #$0003
    BNE .next_color_lp

    REP #$30
    LDA #$0008
    STA $2E
    JSR check_dupe      ; ensure yellow doesn't get overwritten
    LDA #$000E
    STA $2E
    JSR check_dupe      ; ensure orange doesn't get overwritten
.exit
    PLP
    RTL

; find color in $2e and duplicate it for any matching slots with same x,y
check_dupe:
    LDX #$0000
.find_lp
    LDA $702603,X
    AND #$00FF
    CMP $2E
    BEQ .found
    INX #4
    CPX #$000C
    BEQ .done
    BRA .find_lp

.found
    LDA $702604,X       ; coords
    LDX #$0000
    
.find_lp_2
    CMP $702604,X
    BEQ .match
.inc_x
    INX #4
    CPX #$000C
    BEQ .done
    BRA .find_lp_2
.match
    PHA
    SEP #$20
    LDA $2E
    STA $702603,X
    REP #$30
    PLA
    BRA .inc_x
.done
    RTS

warnpc !bank_8f_free_space2_end

;;; Copy/clear save markers when doing file copy/clear
org !bank_81_free_space_start
hook_file_copy:
    PHY
    LDA $19B7           ; src slot
    ASL
    ASL
    TAX                 ; src*4
    LDA $19B9           ; dst slot
    ASL
    ASL
    TAY                 ; dst*4
    LDA $702602,X
    PHA
    LDA $702604,X
    TYX
    STA $702604,X
    PLA
    STA $702602,X
    PLY
    LDX #$0510          ; replaced code
    RTS

hook_file_clear:
    LDA $19B7           ; src slot
    ASL
    ASL
    TAX                 ; slot*4
    LDA #$FFFF
    STA $702602,X
    STA	$702604,X
    LDX #$0500          ; replaced code
    RTS

warnpc !bank_81_free_space_end
