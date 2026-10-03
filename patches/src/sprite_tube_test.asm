; Temporary Sprite Tube test: load room graphics and write OAM directly.
arch snes.cpu
lorom

!tube_drawn = $7EF4E4 ; Word, reset before each frame's sprite construction
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
    LDX #$001E
.palette:
    LDA.l TubePalette,X
    STA.l $7EC3E0,X
    DEX
    DEX
    BPL .palette
    ; Retain the upstream substitution of BG palette 0 colors 4-7.
    LDX #$0006
.background_colors:
    LDA.l $7EC208,X
    STA.l $7EC3E8,X
    DEX
    DEX
    BPL .background_colors

    ; Queue after normal enemy tile loading. VRAM addresses are word offsets.
    LDX $0330
    LDA #$0400 : STA $00D0,X
    LDA.w #TubeGraphics : STA $00D2,X
    LDA #$00B4 : STA $00D4,X
    LDA #$6D00 : STA $00D5,X
    TXA
    CLC
    ADC #$0007
    STA $0330
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
    ; Right column at room X=$0080. Cull before truncating X to nine bits.
    LDA #$0080
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
    LDA #$6ED0 ; Tile $1D0, palette 7, OBJ priority 2, horizontal flip
    %TubeAttributes(0)
    LDA #$2ED0 ; Same tile/palette/priority, unflipped
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
    LDA #$6ED2
    STA.w !tube_oam+2,X
    LDA #$2ED2
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
    LDA #$EED2
    STA.w !tube_oam+2,X
    LDA #$AED2
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

org $B2FFC0
TubePalette:
    incbin "../../Mosaic/Projects/Base/Export/Enemies/F7D3.snes"
assert pc() == $B2FFE0

org $B4F600
TubeGraphics:
    incbin "../../Mosaic/Projects/Base/Export/Enemies/F7D3.gfx"
assert pc() == $B4FA00
