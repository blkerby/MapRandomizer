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
    lda #$FFFF
    sta $09A2
    sta $09A4   ; all items collected and equipped
    lda #$0005  ; Start in loading game state 5 (Main) instead of 0 (Intro)
    rtl
