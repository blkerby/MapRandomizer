;;; Based on https://github.com/theonlydude/RandomMetroidSolver/blob/master/patches/common/src/new_game.asm
;;; Skips intro and starts at Landing Site

arch snes.cpu
lorom

;;; CONSTANTS

incsrc "constants.asm"

!GameStartState = $7ED914
!current_save_slot = $7e0952
!area_explored_mask = $702600
!initial_area_explored_mask = $B5F600  ; must match address in patch/map_tiles.rs


;;; Hijack code that runs during initialization
org $82801d
    jsl startup

org $828067
    jsl gameplay_start

;;; Start in game state 1F (Post-Intro) instead of 1E (Intro)
org $82eeda
    db $1f

; Use Crateria load station 2 for starting location (unused by vanilla)
; Initialize it to load to the Ship; if using randomized start, this will be overwritten in patch.rs
org $80C4E1
    dw $91F8
    dw $896A
    dw $0000 
    dw $0400
    dw $0400
    dw $0040
    dw $0000

;;; CODE in bank A1
org $a1f210

startup:
    jsl check_new_game      : bne .end

    ; Initialize the load station and area:
    lda #$0002
    sta $078B
    sta $7ED916
    stz $079f
    stz $1F5B

    ; Unlock Tourian statues room (to avoid camera glitching when entering from bottom, and also to ensure game is
    ; beatable since we don't take it into account as an obstacle in the item randomization logic)
    lda #$0004
    sta $7ED821

    ; Initialize item collection times:
    lda #$0000
    ldx #$0050
.clear_item_times:
    sta !stat_item_collection_times, x
    dex
    dex
    bne .clear_item_times

    ; If there are no existing save files, then perform global initialization:
    lda $0954
    bne .skip_init

    ; Initialize areas explored
    lda !initial_area_explored_mask
    sta !area_explored_mask

    ; Initialize RTA timer & global stats
    lda #$0000
    sta !stat_timer
    sta !stat_timer+2
    sta !stat_saves
    sta !stat_deaths
    sta !stat_reloads
    sta !stat_loadbacks
    sta !stat_resets

    ; Copy initial revealed tiles from B5:F000 (e.g. to set map station tiles to revealed)
    ldx #$0600
.copy_revealed
    dex
    dex
    ; revealed tiles:
    lda $B5F000, X
    sta $702000, X
    ; partially revealed tiles:
    lda $B5F800, X
    sta $702700, X
    txa
    bne .copy_revealed

.skip_init:

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
    ; Fix BG2 size if starting in an ocean room:
    ; (not sure exactly how this gets messed up, probably something to do with the scrolling sky)
    lda $079B
    cmp #$93FE  ; west ocean
    beq .ocean
    cmp #$968F  ; homing geemer room
    beq .ocean
    cmp #$94FD  ; east ocean
    bne .skip
.ocean:
    lda #$0800
    sta $098E   ; set BG2 size to $800
.skip:

    jsl check_new_game  : bne .end
    
    ; Keep track of the vanilla area for the starting room
    lda $079f
    pha

    stz $079f  ; use save slot for area 0, regardless of what the starting area is
    lda !current_save_slot
    jsl $818000  ; save new game

    ; Restore the vanilla area for the starting room
    pla
    sta $079f

.end:
    rtl

warnpc $a1f400