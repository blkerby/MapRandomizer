;;; Skips intro and starts at Landing Site

arch snes.cpu
lorom

;;; Hijack code that runs during initialization
org $82801d
    jsl startup

;;; Start in game state 1F (Post-Intro) instead of 1E (Intro)
org $82eeda
    db $1f

;;; CODE in bank A1
org $a1f210

startup:
    lda #$0004  ; Unlock Tourian (to avoid camera glitching when entering from bottom, and also to ensure game is
    sta $7ED821 ; beatable since we don't take it into account as an obstacle in the item randomization logic)

    ; temporary extra stuff:
    lda #$FFFF
    sta $09A2   ; all items equipped
    sta $09A4   ; all items collected
    lda #$100b
    sta $09a6  ; all beams equipped except spazer
    lda #$100f
    sta $09a8   ; all beams collected
    lda #$05DB
    sta $09C2  ; health
    sta $09C4  ; max health
    lda #$00E6
    sta $09C6   ; missiles
    sta $09C8   ; max missiles
    lda #$0032
    sta $09CA   ; supers
    sta $09CC   ; max supers
    sta $09CE   ; power bombs
    sta $09D0   ; max power bombs
    lda #$0001
    sta $0789   ; area map collected

    lda #$0005  ; Start in loading game state 5 (Main) instead of 0 (Intro)
    rtl
