; From https://github.com/theonlydude/RandomMetroidSolver/blob/master/patches/common/src/door_transition.asm

lorom
arch snes.cpu

;;; use this as exit door asm for croc, phantoon, draygon :
;;; bosses draw their tilemap on BG2, and a routine to draw enemy
;;; BG2 ($A0:9726) is also ran and at the end of every
;;; door transition. It uses $0e1e as flag to know if a VRAM transfer
;;; has to be done. If we exit during croc fight, the value can be
;;; non-0 and some garbage resulting from room tiles decompression
;;; of door transition is copied to BG2 tilemap in the next room.
org $8ff7f0
boss_exit_fix:
    stz $0e1e	; clear the flag to disable enemy BG2 tilemap routine
    rts
