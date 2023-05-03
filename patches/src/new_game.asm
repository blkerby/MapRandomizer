;;; Based on https://github.com/theonlydude/RandomMetroidSolver/blob/master/patches/common/src/new_game.asm
;;; Skips intro and starts at Landing Site

arch snes.cpu
lorom

;;; CONSTANTS
!GameStartState = $7ED914
!current_save_slot = $7e0952
!area_explored_mask = $702600

;;; Hijack code that runs during initialization
org $82801d
    jsl startup

org $828067
    jsl gameplay_start

;;; Start in game state 1F (Post-Intro) instead of 1E (Intro)
org $82eeda
    db $1f

;;; CODE in bank A1
org $a1f210

startup:
    jsl check_new_game      : bne .end

    ; Initialize the load station and area, map-area:
    stz $078B : stz $079f : stz $1f5b

    ; Initialize areas explored
    lda #$0001
    sta !area_explored_mask

    ; Unlock Tourian statues room (to avoid camera glitching when entering from bottom, and also to ensure game is
    ; beatable since we don't take it into account as an obstacle in the item randomization logic)
    lda #$0004
    sta $7ED821

    ; Copy initial explored tiles from B5:F000 (to set map station tiles to explored)
    ; Also initialize these as revealed tiles (so that map station tiles will be taken into account in pause map scroll limits).
    ldx #$0600
.copy_explored
    dex
    dex
    lda $B5F000, X
    sta $7ECD52, X
    sta $702000, X
    txa
    bne .copy_explored

    ; If there are no existing save files, then clear map revealed tiles (persisted across deaths/reloads)
    lda $0954
    bne .skip_clear_revealed
    ldx #$0600
.clear_revealed
    dex
    dex
    lda #$0000
    sta $702000, X    
    txa
    bne .clear_revealed
.skip_clear_revealed:

    ; Do the same for the local-area explored tiles (TODO: maybe simplify this.)
    ldx #$0100
.copy_explored_crateria
    dex
    dex
    lda $B5F000, X
    sta $07F7, X
    txa
    bne .copy_explored_crateria


    lda #$0006  ; Start in game state 6 (Loading game data) instead of 0 (Intro) or 5 (File select map)
    sta !GameStartState
.end
    lda !GameStartState
    rtl

;;; zero flag set if we're starting a new game
check_new_game:
    ;; Make sure game mode is 1f
    lda $7e0998
    cmp #$001f : bne .end
    ;; check that Game time and frames is equal zero for new game
    ;; (Thanks Smiley and P.JBoy from metconst)
    lda $09DA
    ora $09DC
    ora $09DE
    ora $09E0
.end:
    rtl


gameplay_start:
    jsl check_new_game  : bne .end
    lda !current_save_slot
    jsl $818000
.end:
    rtl

warnpc $a1f300