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
!initial_area = $B5FE00  ; area used for the load station, not the map area
!initial_load_station = $B5FE02
!initial_items_collected = $B5FE04
!initial_items_equipped = $B5FE06
!initial_beams_collected = $B5FE08
!initial_beams_equipped = $B5FE0A
!initial_boss_bits = $B5FE0C
!initial_item_bits = $B5FE12
!initial_energy = $B5FE52
!initial_max_energy = $B5FE54
!initial_reserve_energy = $B5FE56
!initial_max_reserve_energy = $B5FE58
!initial_reserve_mode = $B5FE5A
!initial_missiles = $B5FE5C
!initial_max_missiles = $B5FE5E
!initial_supers = $B5FE60
!initial_max_supers = $B5FE62
!initial_power_bombs = $B5FE64
!initial_max_power_bombs = $B5FE66

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
    jsl check_new_game 
    beq .init
    jmp .end

.init:
    ; Initialize the load station and area:
    lda !initial_load_station 
    sta $078B
    sta $7ED916
    lda !initial_area
    sta $079f
    sta $7ED918
    sta $1F5B

    ; Initialize items/flags collected/equipped:
    lda !initial_items_collected
    sta $09A4
    lda !initial_items_equipped
    sta $09A2
    lda !initial_beams_collected
    sta $09A8
    lda !initial_beams_equipped
    sta $09A6
    lda !initial_boss_bits
    sta $7ED828
    lda !initial_boss_bits+2
    sta $7ED82A
    lda !initial_boss_bits+4
    sta $7ED82C
    lda !initial_energy
    sta $09C2
    lda !initial_max_energy
    sta $09C4
    lda !initial_reserve_energy
    sta $09D6
    lda !initial_max_reserve_energy
    sta $09D4
    lda !initial_reserve_mode
    sta $09C0
    lda !initial_missiles
    sta $09C6
    lda !initial_max_missiles
    sta $09C8
    lda !initial_supers
    sta $09CA
    lda !initial_max_supers
    sta $09CC
    lda !initial_power_bombs
    sta $09CE
    lda !initial_max_power_bombs
    sta $09D0
    ; item bits:
    ldx #$0040
.item_bits_loop:
    lda !initial_item_bits-2,x
    sta $7ED870-2,x
    dex
    dex
    bne .item_bits_loop

    ; Set items collected for escape (to make item collection rate show 100%, only applicable for "Escape" start):
    lda #$F32F
    sta $1F5D

    ; Unlock Tourian statues room (to avoid camera glitching when entering from bottom, and also to ensure game is
    ; beatable since we don't take it into account as an obstacle in the item randomization logic)
    lda #$0004
;    lda #$0044   ; set escape flag
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
    ldx #$0020
.clear_timers:
    sta !stat_pause_time-2, x
    dex
    dex
    bne .clear_timers

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
    sta $0998
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
    ; Fix BG2 size (which would be wrong if starting in certain rooms):
    lda #$0800
    sta $098E   ; set BG2 size to $800

    jsl check_new_game  : bne .end
    
    ; Keep track of the vanilla area for the starting room
    lda $079f
    pha

    lda !initial_area
    sta $079f
    lda !current_save_slot
    jsl $818000  ; save new game

    ; Restore the vanilla area for the starting room
    pla
    sta $079f

.end:
    rtl

warnpc $a1f400