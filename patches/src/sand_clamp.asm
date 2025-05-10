lorom

; Function to clamp Samus' X position, used for sand room entry to ensure
; that Samus doesn't get stuck inside a wall.
;
; Note: we don't bother with adjusting subpixels.

incsrc "constants.asm"

!bank_85_free_space_start = $85A9B0   ; must match address in patch.rs
!bank_85_free_space_end = $85AA00

org !bank_85_free_space_start
; Input: X = min X position for sand entry
;        Y = max X position for sand entry
clamp_sand:
    ; Check if in a loading game state, and skip applying clamping if so:
    ; (This can apply with the West Sand Hole and East Sand Hole starting locations)
    LDA $0998
    CMP #$0006
    BEQ .end
    CMP #$001F
    BEQ .end

    ; Since door ASM is patched to run before scrolling starts (in load_plms_early.asm),
    ; the X screen may be based on the earlier room, so only look at lower 8 bits of X positions:
    SEP #$20

    TXA         ; A <- X = min sand position
    CMP $0AF6   ; compare with current X position
    BCC .no_clamp_min
    STA $0AF6   ; current X position <- min sand position
    STA $0B10   ; previous X position <-  min sand position
.no_clamp_min:

    TYA         ; A <- Y = max sand position
    CMP $0AF6   ; compare with current X position 
    BCS .end
    STA $0AF6   ; current X position <- max sand position
    STA $0B10   ; previous X position <-  max sand position

.end:
    REP #$20
    RTL

warnpc !bank_85_free_space_end