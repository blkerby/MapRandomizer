; Fix 32 sprite bug/crash that can occur during door transition
; Possible when leaving Kraid mid-fight, killing Shaktool with wave-plasma, etc.
; Documented by PJBoy: https://patrickjohnston.org/bank/B4#fBD97
; modified by nn357 to support maprando crashfixes dialog system.

arch snes.cpu
lorom

incsrc "constants.asm"

!bank_b4_free_space_start = $b4f4b8
!bank_b4_free_space_end = $b4f4540


;; hooks into vanilla code

org $b4bd9d 
  jmp sprite_crash
  nop #6
  
;; custom code

org !bank_b4_free_space_start
sprite_crash:
    sta $7eef78,x
    dex
    dex
    bne sprite_crash
    lda $7eef78
    beq .skip
    txa
    sta $7eef78
    lda !crash_toggles2
    and #$000f
    beq .default
    ;cmp #$0002
    ;beq .fix
.warn
    ;lda #$0046          ; bug ID [cannot display messagebox in door transition and it would be messy to implement otherwise] - id46 reserved for other uses
    ;jsl !bug_dialog
.fix
.skip    
    rtl
.default
    ;lda #$0046          ; bug ID [cannot display messagebox in door transition and it would be messy to implement otherwise] - id46 reserved for other uses
    ;jsl !bug_dialog
    jsl !kill_samus
    stz $05f5 ; enable sounds
    rtl

assert pc() <= !bank_b4_free_space_end

;;; $BD97: Clear sprite objects ;;;
{
; BUG: Doesn't clear $7E:EF78 due to wrong branch instruction,
; causes a crash during door transition if 32 sprite objects are created
;$B4:BD97 A2 FE 03    LDX #$03FE
;$B4:BD9A A9 00 00    LDA #$0000

;$B4:BD9D 9F 78 EF 7E STA $7EEF78,x[$7E:F376]
;$B4:BDA1 CA          DEX
;$B4:BDA2 CA          DEX
;$B4:BDA3 D0 F8       BNE $F8    [$BD9D]
;$B4:BDA5 6B          RTL
;}
