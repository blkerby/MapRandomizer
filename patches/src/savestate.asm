; SD2SNES Savestate code
; by acmlm, total, Myria
;
; adapted by Stag Shot
arch 65816
lorom

incsrc "constants.asm"

org $80ffd8
    db 8            ; 256KB SRAM (ensure patch applied after map_progress_maintain)

!bank_85_free_space_start = $85c000
!bank_85_free_space_end = $85c450

!REG_4200_NMI = $84
!IH_CONTROLLER_PRI = $8B
!IH_CONTROLLER_PRI_NEW = $8F
!IH_CONTROLLER_PRI_PREV = $97
!IH_CONTROLLER_SEC = $8D
!IH_CONTROLLER_SEC_NEW = $91
!IH_CONTROLLER_SEC_PREV = $99

!MUSIC_QUEUE_ENTRIES = $0619
!MUSIC_QUEUE_TIMERS = $0629
!MUSIC_QUEUE_NEXT = $0639
!MUSIC_QUEUE_START = $063B
!MUSIC_ENTRY = $063D
!MUSIC_TIMER = $063F
!MUSIC_DATA = $07F3
!MUSIC_TRACK = $07F5
!SOUND_TIMER = $0686
!DISABLE_SOUNDS = $05F5
!SAMUS_HEALTH_WARNING = $0A6A

!SRAM_SAVED_SP = $707F00
!SRAM_MUSIC_DATA = $707F02
!SRAM_MUSIC_TRACK = $707F04
!SRAM_SOUND_TIMER = $707F06
!SRAM_DMA_BANK = $707F80

!MUSIC_ROUTINE = $808FC1

macro wram_to_sram(wram_addr, size, sram_addr)
    dw $0000|$4312, <sram_addr>&$FFFF                            ; VRAM address >> 1.
    dw $0000|$4314, ((<sram_addr>>>16)&$FF)|((<size>&$FF)<<8)    ; A addr = $70xxxx, size = $xx00
    dw $0000|$4316, (<size>>>8)&$FF                              ; size = $80xx ($8000), unused bank reg = $00.
    dw $0000|$2181, <wram_addr>&$FFFF                            ; WRAM addr = $xx0000
    dw $1000|$2183, ((<wram_addr>>>16)&$FF)-$7E                  ; WRAM addr = $7Exxxx  (bank is relative to $7E)
    dw $1000|$420B, $02                                          ; Trigger DMA on channel 1
endmacro

macro vram_to_sram(vram_addr, size, sram_addr)
    dw $0000|$2116, <vram_addr>>>1                               ; VRAM address >> 1.
    dw $9000|$213A, $0000                                        ; VRAM dummy read.
    dw $0000|$4312, <sram_addr>&$FFFF                            ; A addr = $xx0000
    dw $0000|$4314, ((<sram_addr>>>16)&$FF)|((<size>&$FF)<<8)    ; A addr = $70xxxx, size = $xx00
    dw $0000|$4316, (<size>>>8)&$FF                              ; size = $80xx ($0000), unused bank reg = $00.
    dw $1000|$420B, $02                                          ; Trigger DMA on channel 1
endmacro

macro sram_to_wram(wram_addr, size, sram_addr)
    dw $0000|$4312, <sram_addr>&$FFFF                            ; A addr = $xx0000
    dw $0000|$4314, ((<sram_addr>>>16)&$FF)|((<size>&$FF)<<8)    ; A addr = $70xxxx, size = $xx00
    dw $0000|$4316, (<size>>>8)&$FF                              ; size = $80xx ($8000), unused bank reg = $00.
    dw $0000|$2181, <wram_addr>&$FFFF                            ; WRAM addr = $xx0000
    dw $1000|$2183, ((<wram_addr>>>16)&$FF)-$7E                  ; WRAM addr = $7Exxxx  (bank is relative to $7E)
    dw $1000|$420B, $02                                          ; Trigger DMA on channel 1
endmacro

macro sram_to_vram(vram_addr, size, sram_addr)
    dw $0000|$2116, <vram_addr>>>1                               ; VRAM address >> 1.
    dw $0000|$4312, <sram_addr>&$FFFF                            ; A addr = $xx0000
    dw $0000|$4314, ((<sram_addr>>>16)&$FF)|((<size>&$FF)<<8)    ; A addr = $70xxxx, size = $xx00
    dw $0000|$4316, (<size>>>8)&$FF                              ; size = $80xx ($0000), unused bank reg = $00.
    dw $1000|$420B, $02                                          ; Trigger DMA on channel 1
endmacro

macro sram_to_sram(sram_src, size, sram_dest)
    LDX #<size>
-
    LDA <sram_src>-2,X
    STA <sram_dest>-2,X
    DEX : DEX
    BNE -
endmacro

macro a8() ; A = 8-bit
    SEP #$20
endmacro

macro a16() ; A = 16-bit
    REP #$20
endmacro

macro i8() ; X/Y = 8-bit
    SEP #$10
endmacro

macro i16() ; X/Y = 16-bit
    REP #$10
endmacro

macro ai8() ; A + X/Y = 8-bit
    SEP #$30
endmacro

macro ai16() ; A + X/Y = 16-bit
    REP #$30
endmacro

org !bank_85_free_space_start
; *****************************************
; These jumps must remain at this address, as the fast_reload.asm controller hook hard references them.
    jmp save_state
    jmp load_state
; This setting must remain at this address for patch.rs (0 = unlimited, 1 = limited)
savestate_limited:  skip 2
; *****************************************

; These can be modified to do game-specific things before and after saving and loading
; Both A and X/Y are 16-bit here
pre_load_state:
{  
    LDA !MUSIC_DATA : STA !SRAM_MUSIC_DATA
    LDA !MUSIC_TRACK : STA !SRAM_MUSIC_TRACK
    LDA !SOUND_TIMER : STA !SRAM_SOUND_TIMER
    RTS
}

post_load_state:
{
    JSR post_load_music
    
    ; If sounds are not enabled, the game won't clear the sounds
    LDA !DISABLE_SOUNDS : PHA
    STZ !DISABLE_SOUNDS
    JSL $82BE17 ; Cancel sound effects
    PLA : STA !DISABLE_SOUNDS
    
    ; Reload BG3 (minimap) tiles, except in credits
    LDA $0998
    CMP #$0027
    BEQ .skip_bg3
    JSL $85A290
.skip_bg3
    RTS
}

post_load_music:
{
    LDY !MUSIC_TRACK
    LDA !MUSIC_QUEUE_NEXT : CMP !MUSIC_QUEUE_START : BEQ .music_queue_empty

    DEC #2 : AND #$000E : TAX
    LDA !MUSIC_QUEUE_ENTRIES,X : BMI .queued_music_data
    TXA : TAY : CMP !MUSIC_QUEUE_START : BEQ .no_music_data

  .music_queue_data_search
    DEC #2 : AND #$000E : TAX
    LDA !MUSIC_QUEUE_ENTRIES,X : BMI .queued_music_data
    TXA : CMP !MUSIC_QUEUE_START : BNE .music_queue_data_search

  .no_music_data
;    LDA !sram_music_toggle : CMP #$0002 : BPL .fast_off_preset_off

    ; No data found in queue, check if we need to insert it
    LDA !SRAM_MUSIC_DATA : CMP !MUSIC_DATA : BEQ .music_queue_increase_timer

    ; Insert queued music data
    DEX #2 : TXA : AND #$000E : TAX
    LDA #$FF00 : CLC : ADC !MUSIC_DATA : STA !MUSIC_QUEUE_ENTRIES,X
    LDA #$0008 : STA !MUSIC_QUEUE_TIMERS,X

  .queued_music_data
;   LDA !sram_music_toggle : CMP #$0002 : BMI .queued_music_data_clear_track

    ; There is music data in the queue, assume it was loaded
    LDA !MUSIC_QUEUE_ENTRIES,X : STA !MUSIC_DATA
    BRA .fast_off_preset_off

  .music_queue_empty
;    LDA !sram_music_toggle : CMP #$0002 : BPL .fast_off_preset_off
    LDA !SRAM_MUSIC_DATA : CMP !MUSIC_DATA : BNE .clear_track_load_data
    JMP .check_track

  .clear_track_load_data
    TDC : JSL !MUSIC_ROUTINE
    LDA #$FF00 : CLC : ADC !MUSIC_DATA : JSL !MUSIC_ROUTINE
    BRA .load_track

  .fast_off_preset_off
    ; Treat music as already loaded
    STZ !MUSIC_QUEUE_TIMERS : STZ !MUSIC_QUEUE_TIMERS+$2
    STZ !MUSIC_QUEUE_TIMERS+$4 : STZ !MUSIC_QUEUE_TIMERS+$6
    STZ !MUSIC_QUEUE_TIMERS+$8 : STZ !MUSIC_QUEUE_TIMERS+$A
    STZ !MUSIC_QUEUE_TIMERS+$C : STZ !MUSIC_QUEUE_TIMERS+$E
    STZ !MUSIC_QUEUE_NEXT : STZ !MUSIC_QUEUE_START
    STZ !MUSIC_ENTRY : STZ !MUSIC_TIMER
    STZ !SOUND_TIMER : STY !MUSIC_TRACK
    BRA .done

  .music_queue_increase_timer
    ; Data is correct, but we may need to increase our sound timer
    LDA !SRAM_SOUND_TIMER : CMP !MUSIC_TIMER : BMI .done
    STA !MUSIC_TIMER : STA !SOUND_TIMER
    BRA .done

  .queued_music_data_clear_track
    ; Insert clear track before queued music data and start queue there
    DEX #2 : TXA : AND #$000E : STA !MUSIC_QUEUE_START : TAX
    STZ !MUSIC_QUEUE_ENTRIES,X : STZ !MUSIC_ENTRY

    ; Clear all timers before this point
  .music_clear_timer_loop
    TXA : DEC #2 : AND #$000E : TAX
    STZ !MUSIC_QUEUE_TIMERS,X : CPX !MUSIC_QUEUE_NEXT : BNE .music_clear_timer_loop

    ; Set timer on the clear track command
    LDX !MUSIC_QUEUE_START

  .queued_music_prepare_set_timer
    LDA !SRAM_SOUND_TIMER : BNE .queued_music_set_timer
    INC

  .queued_music_set_timer
    STA !MUSIC_QUEUE_TIMERS,X : STA !SOUND_TIMER : STA !MUSIC_TIMER
    BRA .done

  .check_track
    LDA !SRAM_MUSIC_TRACK : CMP !MUSIC_TRACK : BEQ .done

  .load_track
    LDA !MUSIC_TRACK : JSL !MUSIC_ROUTINE

  .done
    RTS
}

; These restored registers are game-specific and needs to be updated for different games
register_restore_return:
{
    %a8()
    ; restore status register
    LDA !REG_4200_NMI : STA $4200
    %a16()
    ; run NMI once to avoid corrupted palette/flashing
    JSL $808338
    RTL
}

save_state:
{
    LDA.l savestate_limited
    BEQ .saves_left
    LDA !savestate_state
    BIT #!savestate_save_used_mask
    BEQ .saves_left
.no_saves
    ; Clear inputs
    TDC : STA !IH_CONTROLLER_PRI : STA !IH_CONTROLLER_PRI_NEW
    RTL
    
.saves_left
    %ai8()
    PHB
    TDC : PHA : PLB

    TAX : TXY
  .save_dma_regs
    ; Store DMA registers to SRAM
    LDA $4300,X : STA !SRAM_DMA_BANK,X
    INX
    INY : CPY #$0B : BNE .save_dma_regs
    CPX #$7B : BEQ .done
    TXA : CLC : ADC #$05 : TAX
    LDY #$00
    BRA .save_dma_regs

  .done
    %ai16()
    ; Mark the savestate as existing and consume Limited mode's save opportunity
    LDA !savestate_state
    ORA #!savestate_exists_mask|!savestate_save_used_mask
    STA !savestate_state

    LDX #save_write_table
    ; fallthrough to run_vm
}

run_vm:
{
    PHK : PLB
    JMP vm
}

save_write_table:
    ; Turn PPU off
    dw $1000|$2100, $80
    dw $1000|$4200, $00
    ; Single address, B bus -> A bus.  B address = reflector to WRAM ($2180).
    dw $0000|$4310, $8080  ; direction = B->A, byte reg, B addr = $2180

    ; Copy WRAM segments, uses $710000-$747FFF
    %wram_to_sram($7E0000, $8000, $710000)
    %wram_to_sram($7E8000, $8000, $720000)
    %wram_to_sram($7F0000, $8000, $730000)
    %wram_to_sram($7F8000, $8000, $740000)

    ; Address pair, B bus -> A bus.  B address = VRAM read ($2139).
    dw $0000|$4310, $3981  ; direction = B->A, word reg, B addr = $2139
    dw $1000|$2115, $0080  ; VRAM address increment mode.

    ; Copy VRAM segments, uses $750000-$767FFF
    %vram_to_sram($0000, $8000, $750000)
    %vram_to_sram($8000, $8000, $760000)

    ; Copy CGRAM, uses SRAM $771000-$7711FF
    dw $1000|$2121, $00    ; CGRAM address
    dw $0000|$4310, $3B80  ; direction = B->A, byte reg, B addr = $213B
    dw $0000|$4312, $1000  ; A addr = $xx0000
    dw $0000|$4314, $0077  ; A addr = $77xxxx, size = $xx00
    dw $0000|$4316, $0002  ; size = $02xx ($0200), unused bank reg = $00.
    dw $1000|$420B, $02    ; Trigger DMA on channel 1

    ; Done
    dw $0000, save_return

save_return:
{
    PEA $0000 : PLB : PLB

    %ai16()
    ; Clear inputs
    TDC : STA !IH_CONTROLLER_PRI : STA !IH_CONTROLLER_PRI_NEW

    ; Save fix_transition_bad_tiles SRAM to prevent possible door transition corruption
    %sram_to_sram($704000, $400, $770200)

    ; Save temporary tilemap segment, uses $770000-$771000
    %sram_to_sram($703000, $1000, $770000)
    
    PLB
    TSC : STA !SRAM_SAVED_SP
    JMP register_restore_return
}

load_clear_inputs:
    ; Clear inputs and prevent repeated loads
    TDC : STA !IH_CONTROLLER_PRI : STA !IH_CONTROLLER_PRI_NEW
    LDA $82FE7A : STA !IH_CONTROLLER_PRI_PREV
    RTS
    
load_state:
{
    LDA !savestate_state
    BIT #!savestate_exists_mask
    BEQ .no_loads

.loads_left
    JSR pre_load_state

    %a8()
    PHB
    TDC : PHA : PLB
    LDX #load_write_table
    JMP run_vm

.no_loads
    JSR load_clear_inputs
    RTL
}

load_write_table:
    ; Disable HDMA
    dw $1000|$420C, $00
    ; Turn PPU off
    dw $1000|$2100, $80
    dw $1000|$4200, $00
    ; Single address, A bus -> B bus.  B address = reflector to WRAM ($2180).
    dw $0000|$4310, $8000  ; direction = A->B, B addr = $2180

    ; Copy WRAM segments, uses $710000-$747FFF
    %sram_to_wram($7E0000, $8000, $710000)
    %sram_to_wram($7E8000, $8000, $720000)
    %sram_to_wram($7F0000, $8000, $730000)
    %sram_to_wram($7F8000, $8000, $740000)

    ; Address pair, A bus -> B bus.  B address = VRAM write ($2118).
    dw $0000|$4310, $1801  ; direction = A->B, B addr = $2118
    dw $1000|$2115, $0080  ; VRAM address increment mode.

    ; Copy VRAM segments, uses $750000-$767FFF
    %sram_to_vram($0000, $8000, $750000)
    %sram_to_vram($8000, $8000, $760000)

    ; Copy CGRAM, uses SRAM $771000-$7711FF
    dw $1000|$2121, $00    ; CGRAM address
    dw $0000|$4310, $2200  ; direction = A->B, byte reg, B addr = $2122
    dw $0000|$4312, $1000  ; A addr = $xx0000
    dw $0000|$4314, $0077  ; A addr = $77xxxx, size = $xx00
    dw $0000|$4316, $0002  ; size = $02xx ($0200), unused bank reg = $00.
    dw $1000|$420B, $02    ; Trigger DMA on channel 1

    ; Done
    dw $0000, load_return

load_return:
{
    %ai16()
    PLB
    LDA !SRAM_SAVED_SP : TCS
    
    ; Restore fix_transition_bad_tiles SRAM to prevent possible door transition corruption
    %sram_to_sram($770200, $400, $704000)
    
    ; Restore temporary tilemap segment
    %sram_to_sram($770000, $1000, $703000)
    
    ; Clear inputs and prevent repeated loads
    JSR load_clear_inputs
    
    ; clear frame held counters
    TDC
    %ai8()
    PHB
    PHA : PLB
    TAX : TXY
  .load_dma_regs
    ; Load DMA registers from SRAM
    LDA !SRAM_DMA_BANK,X : STA $4300,X
    INX
    INY : CPY #$0B : BNE .load_dma_regs
    CPX #$7B : BEQ .load_dma_regs_done
    TXA : CLC : ADC #$05 : TAX
    LDY #$00
    BRA .load_dma_regs

  .load_dma_regs_done
    ; Restore registers and return.
    %ai16()
    PLB
    JSR post_load_state

    JMP register_restore_return
}

vm:
{
    ; Data format: xx xx yy yy
    ; xxxx = little-endian address to write to .vm's bank
    ; yyyy = little-endian value to write
    ; If xxxx has high bit set, read and discard instead of write.
    ; If xxxx has bit 12 set ($1000), byte instead of word.
    ; If yyyy has $DD in the low half, it means that this operation is a byte
    ; write instead of a word write.  If xxxx is $0000, end the VM.
    %ai16()
    ; Read address to write to
    LDA.w $0000,X : BEQ .vm_done
    TAY
    INX #2
    ; Check for byte mode
    BIT.w #$1000 : BEQ .vm_word_mode
    AND.w #$EFFF : TAY
    %a8()
  .vm_word_mode
    ; Read value
    LDA.w $0000,X
    INX #2
  .vm_write
    ; Check for read mode (high bit of address)
    CPY.w #$8000 : BCS .vm_read
    STA $0000,Y
    BRA vm
  .vm_read
    ; "Subtract" $8000 from Y by taking advantage of bank wrapping.
    LDA $8000,Y
    BRA vm
  .vm_done
    ; A, X and Y are 16-bit at exit.
    ; Return to caller.  The word in the table after the terminator is the
    ; code address to return to.
    JMP ($0002,X)
}

assert pc() <= !bank_85_free_space_end
