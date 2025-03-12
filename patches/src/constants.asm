; Global stats (stored in SRAM, not specific to any save file)
!stat_timer = $701E10
!stat_saves = $701E14
!stat_deaths = $701E16
!stat_reloads = $701E18
!stat_loadbacks = $701E1A
!stat_resets = $701E1C
!stat_final_time = $701E1E
!stat_pause_time = $701E22
!stat_area0_time = $701E26
!stat_area1_time = $701E2A
!stat_area2_time = $701E2E
!stat_area3_time = $701E32
!stat_area4_time = $701E36
!stat_area5_time = $701E3A
!stat_area6_time = $701E3E

; Local stats (stored to normal RAM that goes into save files)
!stat_item_collection_times = $7efe06  ; must match address in patch.rs

; Additional save data:
!num_disabled_etanks = $09EC

!spin_lock_enabled = $1F70
!last_samus_map_x = $1F72
!last_samus_map_y = $1F74
!loadback_ready = $1F7A
!nmi_timeronly = $1F7C
!previous_room = $1F7E
!nmi_counter = $1F80
!nmi_pause = $1F81
!nmi_area0 = $1F82
!nmi_area1 = $1F83
!nmi_area2 = $1F84
!nmi_area3 = $1F85
!nmi_area4 = $1F86
!nmi_area5 = $1F87
!nmi_area6 = $1F88

; Objectives
!objectives_max = $0014
!objectives_num = $82FFFC ; bits 0-15
!objectives_addrs = $8FEBC0
!objectives_bitmasks #= !objectives_addrs+(2*!objectives_max)

; Settings:
; target number of frames for the pause menu black screen to lag
; (to compensate for VRAM decompression being faster): 
!unpause_black_screen_lag_frames = $dfff07
