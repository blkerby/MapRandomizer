lorom
arch snes.cpu

; immediately return from routines that would draw map icons
org $82B672
    rtl

org $82BB30
    rtl
