; From https://github.com/theonlydude/RandomMetroidSolver/blob/master/patches/common/src/decompression.asm
;$30FF-$3265

;;; Decompression optimization by Kejardon, with fixes by PJBoy and Maddo

;Compression format: One byte(XXX YYYYY) or two byte (111 XXX YY-YYYYYYYY) headers
;XXX = instruction, YYYYYYYYYY = counter

;Modified by Tundain (20/12/2023: DMY)

;changes:
;-when checking for overflow, the branch is now taken whenever there is overflow, rather than when there's not
;    because a non-taken branch is 1 cycle cheaper, and this happens more often
;    this leads to a bit more code, but i believe it's worth it (there's some space left before hitting the VRAM decomp routine
;-direct copy has been moved in front of the main loop, because that way a JMP doesn't need to get taken from there.
;-word fill has been modified to go straight into 16-bit mode, to avoid having to load a word whilst in 8-bit mode
;-Because storing happens more often than loading during decompression, DB is now set to the destination bank.
;    this means loading now uses indirect long, and storing indirect short
;    updating source bank is also simpler now


;doesn't change that much, but it is faster!


;Overhauled by Tundain (26/12/2024: DMY)
;Huge overhaul this time for significant performance improvement!
; I changed the way it writes data to use the WRAM I/O port ($2180)
; I also made the decompression commands use DMA transfers where applicable
; I also changed the DP during decompression to $4300 and moved all temporary variables usage to over there, since DMA channel RAM is apparently faster than WRAM.
; I also undid some changes in the previous version because they became impractical
; and a bunch of other small tweaks

;Also optimizes the VRAM decompression routine!

;unfortunately now requires a bit of freespace in bank $80 
;if any other patch you're using conflicts, feel free to repoint

incsrc "constants.asm"

!bank_80_free_space_start = $80E2A0
!bank_80_free_space_end = $80E3A0

!bank_89_free_space_start = $89B0C0
!bank_89_free_space_end = $89B110

;also uses one unused RAM variable
;if another patch also uses this RAM variable, find yourself a different one
!DecompFlag = $9B

lorom

;;; hooks

org $8095A7
    JMP checkIfInLoading

;change bottom of HUD hcounter, since IRQ does more work
org $8096A5
    LDX #$0078

org $80972F
    LDX #$0078

;some hijacks
org $809876
    JMP setDp
return:

org $809883
    JMP ResetDp
return2:

;HDMA channels aren't preserved during unpause, but since decomp uses DMA 2/3, those now need to be preserved
org $828D1A
    JSR preserveDMA_pause

org $80A15F
    JSR restoreDMA_unpause : NOP

org $82E7A8
    JSL VRAMdecomp

org $82E7BB
    JSL VRAMdecomp

org $89AC2E
    BEQ skip_fx_ptr                ; adjust stock branch for JSL hook @ 89AC34

org $89AC34
    JSL UploadFXTilemap : NOP #2
skip_fx_ptr:
    JSR fx_hook                    ; replaced code above

;;; new decompression func

org $80B0FF
    LDA $02, S
    STA $45
    LDA $01, S
    STA $44        ;Address of target address for data into $44
    CLC
    ADC #$0003
    STA $01, S    ;Skip target address for RTL
    LDY #$0001
    LDA [$44], Y
    STA $4C
    INY
    LDA [$44], Y
    STA $4D

warnpc $80B119

org $80B119; : Later JSL start if 4C (target address) is specified
{;regular decompression
namespace DEFAULT
Start:
    PHP : PHB : PHD
    JSR setup;prepare settings
    BRA NextByte

End:
    PLD
    STZ !DecompFlag
    PLB : PLP : RTL

    Option0:;this one is here so one of the commands doesn't need a JMP to go back to the start (ugly, but saves those extra 3 cycles)
    REP #$20
    STY $35;size
    STX $32;address
    TXA : CLC : ADC $18 : ORA #$8000 : TAX : BCC +
    AND #$7FFF : STA $70;size of data in next bank
    TYA : SEC : SBC $70 : STA $35;size of data in current bank
    SEP #$20
    LDA #$08 : STA $80420B
    LDA $70 : STA $35
    LDA #$80 : STA $33 : INC $34
    LDA $34 : PHA : PLB
    LDA $70 : BEQ start
    +
    SEP #$20
    LDA #$08 : STA $80420B
start:
    REP #$20
    LDA $22 : CLC : ADC $18 : STA $22;update destination address
    SEP #$20
NextByte:
    LDA $0000,x : INX : BNE +
    JSR IncrementBank
    +
    STA $08
    CMP #$FF : BEQ End
    CMP #$E0 : BCC ++
    ASL #3 : AND #$E0 : STA $0A
    LDA $08 : AND #$03 : XBA
    LDA $0000,x : INX : BNE +
    JSR IncrementBank
    +
    BRA +++
++
    AND #$E0 : STA $0A
    TDC : XBA
    LDA $08 : AND #$1F
+++
    TAY : INY : STY $18
    LDA $0A
    BMI Option4567
    BNE +
    JMP Option0 : +
    CMP #$20 : BEQ BRANCH_THETA
    CMP #$40 : BEQ BRANCH_IOTA
;X = 3: Store an ascending number (starting with the value of the next byte) Y times.
    LDA $0000,x : INX : BNE +
    JSR IncrementBank
    +
-
    STA $802180 : INC : DEY
    BNE -
    JMP start
BRANCH_THETA:
    ;X = 1: Copy the next byte Y times.
    LDA $0000,x : XBA : LDA $0000,x : INX : BNE +
    JSR IncrementBank
    +
-    
    STA $80211B
    edgecase1:
    STA $80211B
    BRA transfer
super_rare_edgecase3:
    XBA : BRA rare_return
BRANCH_IOTA:
    LDA $0000,x : XBA : INX : BNE +
    JSR IncrementBank
    +
    LDA $0000,x : XBA : INX : BNE +
    JSR IncrementBank
    +
    ;X = 2: Copy the next two bytes, one at a time, for the next Y bytes.
    rare_return:
    STA $80211B
    edgecase2:
    XBA
    edgecase3:
    STA $80211B;place word to repeatedly copy in multiplication register
    transfer:
    STY $25;set size
    LDA #$01 : STA $80211C
    LDA #$04 : STA $80420B;initiate DMA
    REP #$20
    LDA $22 : STA $802181;update destination address
    SEP #$20
++
    JMP NextByte;DMA autoincrements, so we can skip updating address section
Option4567:
    CMP #$C0
    AND #$20    ;X = 4: Copy Y bytes starting from a given address in the decompressed data.
    STA $5A;4F      ;X = 5: Copy and invert (EOR #$FF) Y bytes starting from a given address in the decompressed data.
    BCS +++    ;X = 6 or 7 branch
    LDA $0000,x : XBA : INX : BNE +
    JSR IncrementBank
    +
    LDA $0000,x : XBA : INX : BNE +
    JSR IncrementBank
    +
    REP #$21
    ADC $42 : STA $62;add starting offset(where we're decompressing to)
--
    SEP #$20
    LDA $5A
    BNE +    ;Inverted
-
    ;the non-inverted dictionnary copy uses MVN for speed
    ;it's faster as long as the data to copy is 4+ bytes long
    STX $50 
    LDX $62 
    REP #$20
    TYA : DEC   
    LDY $22 
    JSR $4372
    STY $22 : TYA : STA $802181 
    SEP #$20
    LDA $34 : PHA : PLB
    LDX $50 
    JMP NextByte
+
-
    LDA [$62];
    EOR #$FF
    STA $802180
    INC $62 : BNE ++
    INC $63
    ++
    DEY

    BNE -
    JMP start
+++
    ;X = 6: Copy Y bytes starting from a given number of bytes ago in the decompressed data.
    ;X = 7: Copy and invert (EOR #$FF) Y bytes starting from a given number of bytes ago in the decompressed data.

    TDC : XBA

    LDA $0000,x : INX : BNE +
    JSR IncrementBank
    +

    REP #$20
    STA $62
    LDA $22 : SBC $62 : STA $62

    BRA --
namespace off    
}

IncrementBank:
    LDX #$8000
    PHA : PHB : PLA : INC : STA $34 : PHA : PLB : PLA
    RTS

{;decompression to VRAM
VRAMdecomp:
print "vram: $",pc
namespace VRAM    
    PHP : PHB : PHD : REP #$10
    JSR setup
    ;overwrite some settings from "setup", since we're writing to the VRAM port now
    LDA #$18 : STA $31
    LDA #$01 : STA $30
    BRA NextByte
    
OddCheck:;if VRAM destination is odd, write first byte, then do transfer
    PHP : SEP #$20
    LSR : BCC ++
    LDA $0000,x : INX : BNE +
    JSR IncrementBank
    +
    STA $802119
    DEC $70 : DEC $18 : REP #$20 : INC $22 : INC $32
    DEY : BEQ +
    ++
    PLP : CLC : RTS
    +
    PLP : SEC : RTS    

End:
    PLD
    STZ !DecompFlag
    PLB : PLP : RTL

    Option0:;this one is here so one of the commands doesn't need a JMP to go back to the start (ugly, but saves those extra 3 cycles)
    REP #$20

    LDA $22 : JSR OddCheck : BCS start
    ; : LSR : BCC ++
    ; LDA $0000,x : INX : BNE +
    ; JSR IncrementBank
    ; +
    ; STA $802119;write first byte if VRAM destination is odd
    ; DEY : BEQ start
    ; DEC $18 : INC $22
    ; ++
    STX $32;address
    STY $35;size
    TXA : CLC : ADC $18 : ORA #$8000 : TAX : BCC +
    AND #$7FFF : STA $70;size of data in next bank
    TYA : SEC : SBC $70 : STA $35 : TAY;size of data in current bank
    SEP #$20
    LDA #$08 : STA $80420B
    PHX : JSR IncrementBank
    TYA : JSR OddCheck : PLX : BCS start
    ; LDA $0000,x : INX : BNE +
    ; JSR IncrementBank
    ; +
    ; STA $802119;write first byte if VRAM destination is odd
    ; DEY : BEQ start
    ; DEC $18 : INC $22
    ; REP #$20
    ; ++
    LDA $70 : STA $35
    LDA #$80 : STA $33
    ;INC $34
    +
    SEP #$20
    +
    LDA #$08 : STA $80420B
start:
    REP #$20
    LDA $22 : CLC : ADC $18 : STA $22;update destination address
    SEP #$20
    
NextByte:
    LDA $0000,x : INX : BNE +
    JSR IncrementBank
    +
    STA $08
    CMP #$FF : BEQ End
    CMP #$E0 : BCC ++
    ASL #3 : AND #$E0 : STA $0A
    LDA $08 : AND #$03 : XBA

    LDA $0000,x : INX : BNE +
    JSR IncrementBank
    +
    BRA +++
++
    AND #$E0 : STA $0A
    TDC : XBA
    LDA $08 : AND #$1F
+++
    TAY : INY : STY $18
    LDA $0A

    BMI Option4567

    BNE +
    JMP Option0 : +
    CMP #$20 : BEQ BRANCH_THETA
    CMP #$40 : BEQ BRANCH_IOTA

;X = 3: Store an ascending number (starting with the value of the next byte) Y times.
    LDA $0000,x : INX : BNE +
    JSR IncrementBank
    +
    PHA
    LDA $22 : LSR : PLA : BCS +
-
    
    STA $802118
    INC 
    DEY : BEQ ++
    +
    STA $802119
    INC : DEY : BNE -
    BNE -
    ++
    JMP start
BRANCH_THETA:
    ;X = 1: Copy the next byte Y times.
    LDA $0000,x : XBA : LDA $0000,x : INX : BNE +
    JSR IncrementBank
    +
-    
    BRA transfer
BRANCH_IOTA:;X = 2: Copy the next two bytes, one at a time, for the next Y bytes.
    LDA $0000,x : XBA : INX : BNE +
    JSR IncrementBank
    +
    LDA $0000,x : XBA : INX : BNE +
    JSR IncrementBank
    +
    transfer:
    PHA
    LDA $22 : LSR : PLA : BCS +
-
    STA $802118 : XBA : DEY : BEQ ++
    +
    STA $802119 : XBA : DEY : BNE -
++
    JMP start
Option4567:
    CMP #$C0
    AND #$20    ;X = 4: Copy Y bytes starting from a given address in the decompressed data.
    STA $5A : BCS +++      ;X = 5: Copy and invert (EOR #$FF) Y bytes starting from a given address in the decompressed data.
    ;X = 6 or 7 branch
    LDA $0000,x : XBA : INX : BNE +
    JSR IncrementBank
    +
    LDA $0000,x : XBA : INX : BNE +
    JSR IncrementBank
    +
    REP #$21
    ADC $42 : STA $62;add starting offset(where we're decompressing to)
-
    REP #$20
    ;    
    LDA $62 : INC $62 : LSR : STA $802116
    LDA $002139 : LDA $002139
    BCC ++ : XBA : ++
    SEP #$20
    PHA : LDA $5A : BEQ ++ : PLA : EOR #$FF : PHA : ++
    REP #$20 : LDA $22 : INC $22 : LSR : BCS ++
    STA $002116 : SEP #$20 : PLA
    STA $802118 : DEY : BNE -
    JMP NextByte
    ++
    STA $002116 : SEP #$20 : PLA
    STA $802119 : DEY : BNE -
    ++++
    JMP NextByte
+++
    ;X = 6: Copy Y bytes starting from a given number of bytes ago in the decompressed data.
    ;X = 7: Copy and invert (EOR #$FF) Y bytes starting from a given number of bytes ago in the decompressed data.
    TDC : XBA
    LDA $0000,x : INX : BNE +
    JSR IncrementBank
    +
    REP #$20
    STA $62
    LDA $22 : SBC $62 : STA $62

    BRA -
namespace off
}

; We don't care if we overwrite some of the "failed NTSC/PAL check" tilemap.
print PC
warnpc $80BC37

org $82E617
    JSR preserveDMA_library_background

org $82E62B
    JSL restoreDMA_library_background

;unused block of code so we don't use freespace in bank $82 :)
;preserves the settings and destination of used channel variables
org $82B5E8
preserveDMA_pause:
    JSR $8DBD  ; run hi-jacked instruction. fall through to below:

preserveDMAregisters:
    PHA
    LDA $4320 : STA $7EF600
    LDA $4330 : STA $7EF602
    LDA $4324 : STA $7EF606
    LDA $4334 : STA $7EF604
    LDA $4370 : STA $7EF608
    LDA $4374 : STA $7EF60A
    PLA
    RTS

preserveDMA_library_background:
    LDA $0000,y  ; run hi-jacked instruction
    BRA preserveDMAregisters

warnpc $82B62B

org !bank_80_free_space_start
 ;DP RAM is relocated to the DMA channels because faster cycles
 ;channel 2 is used to write a repeating word/byte
 ;direction is B->A, so the A address ($23-$24) becomes the destination address
 ;channel 3 is for direct copy, so A->b, with B being the WRAM port
 ;other channels are used as temp variables
setup:
    SEP #$20
    LDA #$01 : STA !DecompFlag
    TDC
    LDX $47
    STZ $85 : STZ $420C;disable HDMA channels
    LDA #$43 : XBA : TCD
    STZ $5B
    LDA $004C : STA $2181 : STA $42 : STA $22
    LDA $004D : STA $2182 : STA $43 : STA $23
    LDA $004E : STA $2183 : STA $64 : STA $24 : STA $73 : STA $74;$73/$74 are the argumants for the MVN
    LDA $0049 : STA $34
    ;channel 2/3 settings
    LDA #$34 : STA $21
    LDA #$81 : STA $20
    LDA #$80 : STA $31
    LDA #$00 : STA $30
    
    LDA #$54 : STA $72;write the MVN
    LDA #$60 : STA $75;write the RTS for the MVN
    LDA $0049 : PHA : PLB
    CLC
    RTS

;if decompression is busy, don't handle HDMA queue (this would overwrite the DMA channels we're using)
checkIfInLoading:
    JSR $91EE
    LDX !DecompFlag : BNE .check_stack
    JMP $95AA

;hyper-rare edge case where NMI interrupts a write to a write-twice register
.check_stack
    LDA $0D,s
    AND #$00FF
    CMP #$0080 ; confirm bank 80
    BNE +++
    LDA $0B,s
    CMP.w #DEFAULT_edgecase1 : BEQ +
    CMP.w #DEFAULT_edgecase2 : BEQ +
    CMP.w #DEFAULT_edgecase3 : BEQ ++
    BRA +++
+
    LDA.w #DEFAULT_rare_return : STA $0B,s : BRA +++
++
    LDA.w #DEFAULT_super_rare_edgecase3 : STA $0B,s
+++
    JMP $95C0

;IRQ also preserves DP
setDp:
    PHD
    REP #$20 : LDA #$0000 : TCD
    LDA $4211 : JMP return
ResetDp:
    PLD : STX $4207 : JMP return2
restoreDMA_unpause:
    JSL $82E97C  ; run hi-jacked instruction
    JSL UploadFXptr

    ; in case fast pause menu QoL is disabled, 
    ; add artificial lag to unpause black screen, to compensate for the accelerated decompression
    LDA !unpause_black_screen_lag_frames
    TAX
.loop:
    BEQ restoreDMA_registers
    LDA !nmi_counter
.wait_frame:
    CMP !nmi_counter  ; wait for frame counter to change
    BEQ .wait_frame
    LDA !nmi_counter
    DEX
    BRA .loop

    ; fall through to below
restoreDMA_registers:
    ;restore some DMA registers that could have been overwritten
    LDA $7EF600 : STA $4320
    LDA $7EF602 : STA $4330
    LDA $7EF606 : STA $4324
    LDA $7EF604 : STA $4334
    LDA $7EF608 : STA $4370
    LDA $7EF60A : STA $4374
    RTS

restoreDMA_library_background:
    JSL $80B119  ; run hi-jacked instruction
    JSR restoreDMA_registers
    RTL
warnpc !bank_80_free_space_end

org $82E97C
    PHP : PHB : REP #$30
    JSL $80A29C
    PEA $8F00 : PLB : PLB
    LDX $07BB  
    LDY $0016,x
    BPL +
-
    LDX $0000,y : INY #2 : JSR (commands,x) : BCC  -
+
    PLB : PLP : RTL
commands:
DW $E9E5, $E9F9, $EA2D, $EA4E, $EA66, $EA56, $EA5E, $E9E7

warnpc $82E9E5

org !bank_89_free_space_start
;change this to use VRAM write table to ensure it happens during blanking
UploadFXTilemap:
    LDA $ABF0,y : STA $1964
UploadFXptr:
    LDA $1964 : BEQ + : STA $4312 
    LDX $0330
    LDA #$0840 : STA $D0,x
    LDA #$8A00 : STA $D3,x
    LDA $1964  : STA $D2,x
    LDA #$5BE0 : STA $D5,x
    TXA : CLC : ADC #$0007
    STA $0330
+
    RTL

;replaced code at $89AC34 by the JSL
fx_hook:
    LDX $1966
    LDA $0009,X
    RTS

warnpc !bank_89_free_space_end
