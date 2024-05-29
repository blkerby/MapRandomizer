; Any low bank could be used here:
!bank_85_free_space_start = $859B00
!bank_85_free_space_end = $859B20

org $91E604
    jsl hook_enable_hyper_beam
    nop : nop

org !bank_85_free_space_start
hook_enable_hyper_beam:
    ; run hi-jacked instructions:
    sta $0A76
    stz $0DC0
    ; fix vanilla bug that can cause slowlock:
    stz $0CD0
    stz $0CD6
    stz $0CD8
    stz $0CDA
    stz $0CDC
    stz $0CDE
    stz $0CE0    
    rtl

warnpc !bank_85_free_space_end