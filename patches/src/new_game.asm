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

;;; Start in loading game state 5 (Main) instead of 0 (Intro)
startup:
    lda #$0005
    rtl
