; Global stats (stored in SRAM, not specific to any save file)
!stat_timer = $701E10
!stat_saves = $701E14
!stat_deaths = $701E16
!stat_reloads = $701E18
!stat_loadbacks = $701E1A
!stat_resets = $701E1C
!stat_final_time = $701E1E

; Local stats (stored to normal RAM that goes into save files)
!stat_item_collection_times = $7efe06  ; must match address in patch.rs
