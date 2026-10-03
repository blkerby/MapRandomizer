; Sprite Tube: select room data, load remapped graphics, and write OAM directly.
arch snes.cpu
lorom

!tube_drawn = $7EF4E4 ; Word, reset before each frame's sprite construction
!tube_active = $7EF4E6 ; Word, set by room load
!tube_x = $7EF4E8      ; Word, right column's room-relative pixel coordinate
!tube_attributes = $7EF4EA ; Word, body tile/palette/priority
!tube_table = $EAB000  ; 20-byte records, terminated by room header $FFFF
!tube_oam = $0500     ; Entries 100-127 of the low OAM buffer

; Both empty and populated rooms return through this enemy-init epilogue.
org $A08BE6
    JMP LoadTube
assert pc() == $A08BE9

; Preserve the room-specific graphics hook, then draw after ordinary sprites.
org $A088BD
    JSL FinishDrawing
assert pc() == $A088C1

; Reset the marker alongside the normal per-frame high-OAM clear.
org $828953
    JSL BeginFrame
assert pc() == $828957

org $80896E
    JML FinaliseOam
assert pc() == $808972

; The unrolled cleanup reaches this boundary after clearing entry 99.
org $808ABE
    JML CleanupTail
assert pc() == $808AC2

org $A0FA00
LoadTube:
    ; DB=A0, A/X/Y 16-bit. The original init saved DB/P for its epilogue.
    PHA
    PHX
    PHY
    JSL LoadRoomTube
    PLY
    PLX
    PLA
    PLB
    PLP
    RTL

BeginFrame:
    PHP
    REP #$20
    PHA
    LDA #$0000
    STA.l !tube_drawn
    PLA
    PLP
    JML $808B1A

FinishDrawing:
    ; Writing last replaces any ordinary sprite pieces allocated to 100-127.
    JSL $A088C4
    PHP
    REP #$30
    PHA
    PHX
    PHY
    PHB
    LDA.l !tube_active
    BEQ .restore
    LDA $12 : PHA
    LDA $14 : PHA
    LDA $16 : PHA
    SEP #$20
    LDA #$7E
    PHA
    PLB
    REP #$20
    LDA #$0001
    STA.l !tube_drawn
    JSR DrawTube
    PLA : STA $16
    PLA : STA $14
    PLA : STA $12
.restore:
    PLB
    PLY
    PLX
    PLA
    PLP
    RTL

; Each column has one packed XY word per row. Only X and the Y phase vary.
macro TubeCoordinates(column)
    !row #= 0
    while !row < 14
        STA.w !tube_oam+<column>+(!row*8)
        if !row < 13
            CLC
            ADC #$1000
        endif
        !row #= !row+1
    endwhile
endmacro

macro TubeAttributes(column)
    !row #= 0
    while !row < 14
        STA.w !tube_oam+<column>+(!row*8)+2
        !row #= !row+1
    endwhile
endmacro

DrawTube:
    ; Cull before truncating X to nine bits.
    LDA.l !tube_x
    SEC
    SBC $0911
    STA $12
    CLC
    ADC #$000F
    CMP #$011F ; Visible when right X is in [-15, 271]
    BCC .visible
    JMP HideTube
.visible:
    LDA $12
    SEC
    SBC #$0010
    STA $14
    ; Every row is a large OBJ. Each packed byte covers two rows.
    LDA #$00AA
    STA $16
    LDA $14
    AND #$0100
    BEQ .right_high
    LDA $16
    ORA #$0011
    STA $16
.right_high:
    LDA $12
    AND #$0100
    BEQ .packed_high
    LDA $16
    ORA #$0044
    STA $16
.packed_high:
    LDA $16
    XBA
    ORA $16
    STA $0589
    STA $058B
    STA $058D
    SEP #$20
    STA $058F
    REP #$20

    ; Repeat the upstream scrolling phase: first Y = 16 - (camera Y & 15).
    LDA $0915
    AND #$000F
    EOR #$FFFF
    INC A
    CLC
    ADC #$0010
    XBA
    STA $16
    LDA $14
    AND #$00FF
    ORA $16
    %TubeCoordinates(0)
    LDA $12
    AND #$00FF
    ORA $16
    %TubeCoordinates(4)
    LDA.l !tube_attributes
    ORA #$4000 ; Horizontal flip
    %TubeAttributes(0)
    LDA.l !tube_attributes
    %TubeAttributes(4)

    ; Match build-mosaic.rs: room-screen row 0 uses the joint, and row 15
    ; uses its vertical flip. First displayed row is (camera Y >> 4) + 1.
    ; Find row 0's OAM offset: 8 * ((15 - (camera Y >> 4)) & 15).
    LDA $0915
    EOR #$FFFF
    AND #$00F0
    LSR A
    TAX
    CPX #$0070 ; Fourteen displayed rows, eight OAM bytes per row
    BCS .bottom_joint
    LDA.l !tube_attributes
    ORA #$4002
    STA.w !tube_oam+2,X
    LDA.l !tube_attributes
    ORA #$0002
    STA.w !tube_oam+6,X
.bottom_joint:
    ; Row 15 immediately precedes row 0 in the repeating 16-row pattern.
    TXA
    SEC
    SBC #$0008
    AND #$0078
    TAX
    CPX #$0070
    BCS .done
    LDA.l !tube_attributes
    ORA #$C002
    STA.w !tube_oam+2,X
    LDA.l !tube_attributes
    ORA #$8002
    STA.w !tube_oam+6,X
.done:
    RTS

HideTube:
    SEP #$20
    LDA #$F0
    !entry #= 0
    while !entry < 28
        STA.w !tube_oam+(!entry*4)+1
        !entry #= !entry+1
    endwhile
    REP #$20
    RTS

FinaliseOam:
    ; Recreate the displaced PHP/REP/LDA, keeping the native jump table.
    PHP
    REP #$30
    LDA.l !tube_drawn
    BEQ .vanilla
    LDA $0590
    CMP #$0190 ; Only clear ordinary entries before the tube block
    BCC .clear_gap
.done:
    STZ $0590
    PLP
    RTL
.clear_gap:
    JML $808979
.vanilla:
    LDA $0590
    CMP #$0200
    BCS .done
    CMP #$0190
    BCC .clear_gap
    ; The four-byte tail hook spans two three-byte stores. Dispatch any
    ; start within entries 100-127 to an intact copy of that cleanup tail.
    LSR A
    STA $12
    LSR A
    ADC $12
    CLC
    ADC.w #VanillaCleanupTail-$012C
    STA $12
    LDA #$00F0
    SEP #$30
    JMP ($0012)

CleanupTail:
    ; A/X/Y are 8-bit here; the native dispatch saved P on the stack.
    PHA
    LDA.l !tube_drawn
    BEQ .vanilla
    PLA
    JML $808B12
.vanilla:
    PLA
    JML VanillaCleanupTail

VanillaCleanupTail:
    !entry #= 0
    while !entry < 28
        STA.w !tube_oam+(!entry*4)+1
        !entry #= !entry+1
    endwhile
    JML $808B12
assert pc() <= $A0FE00

org $B4F600
LoadRoomTube:
    LDA #$0000
    STA.l !tube_active
    LDX #$0000
.search:
    LDA.l !tube_table,X
    CMP #$FFFF
    BNE .check_room
    RTL
.check_room:
    CMP $079B ; Current room header, in bank 8F
    BEQ .found
    TXA
    CLC
    ADC #$0014
    TAX
    BRA .search
.found:
    LDA #$0001
    STA.l !tube_active
    LDA.l !tube_table+2,X
    STA.l !tube_x
    LDA.l !tube_table+4,X
    STA.l !tube_attributes

    ; Copy exactly five colors into the target palette. The first four
    ; follow BG palette 0 colors 4-7; the final tube color is opaque black.
    !color #= 0
    while !color < 5
        LDA.l !tube_table+10+(!color*2),X
        PHX
        TAX
        if !color < 4
            LDA.l $7EC208+(!color*2)
        else
            LDA #$0000
        endif
        STA.l $7EC200,X
        PLX
        !color #= !color+1
    endwhile

    ; Queue after normal enemy tile loading; VRAM destination is in words.
    LDA.l !tube_table+6,X
    PHA
    LDA.l !tube_table+8,X ; Bank byte, followed by zero padding
    PHA
    LDX $0330
    LDA #$0400 : STA $00D0,X
    PLA : STA $00D4,X
    PLA : STA $00D2,X
    LDA #$6D00 : STA $00D5,X
    TXA
    CLC
    ADC #$0007
    STA $0330
.done:
    RTL
assert pc() <= $B4FA00

; Customization replaces this default empty table and writes DMA payloads
; at $EAB800-$EAFFFF. No room is enabled until its record is installed.
org !tube_table
    dw $FFFF
