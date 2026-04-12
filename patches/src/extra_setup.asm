lorom

incsrc "constants.asm"


!bank_b8_free_space_start = $B88000
!bank_b8_free_space_end = $B88100

!bank_8f_free_space_start = $8FFE80
!bank_8f_free_space_end = $8FFF00


; hook extra setup ASM to run right before normal setup ASM
; (careful: escape.asm hijacks the instruction after this)
org $8FE893
    jsr run_extra_setup_asm_wrapper

org !bank_b8_free_space_start

run_extra_setup_asm:
    ; get extra setup ASM pointer to run in bank B5 (using pointer in room state almost completely unused by vanilla, only for X-ray override in BT Room in escape)
    ldx $07bb      ; x <- room state pointer
    lda $8F0010,x
    tax            ; x <- extra room data pointer
    lda $B80001,x  ; a <- [extra room data pointer + 1]
    beq .skip
    sta $1F68         ; write setup ASM pointer temporarily to $1F68, so we can jump to it with JSR. (Is there a less awkward way to do this?)
    ldx #$0000
    jsr ($1F68, x)

.skip:
    ; run hi-jacked instructions
    ldx $07BB
    lda $0018,x
    rtl

assert pc() <= !bank_b8_free_space_end

org !bank_8f_free_space_start

run_extra_setup_asm_wrapper:
    jsl run_extra_setup_asm
    rts

assert pc() <= !bank_8f_free_space_end