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

!spin_lock_enabled = $1F70
!last_samus_map_x = $1F72
!last_samus_map_y = $1F74
!loadback_ready = $1F7A
!nmi_timeronly = $1F7C
