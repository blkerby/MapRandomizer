/* tslint:disable */
/* eslint-disable */
/**
*/
export function set_panic_hook(): void;
/**
* @param {any} inventory
* @param {number} proficiency
* @param {boolean} can_manage_reserves
* @returns {boolean}
*/
export function can_defeat_phantoon(inventory: any, proficiency: number, can_manage_reserves: boolean): boolean;
/**
* @param {any} inventory
* @param {number} proficiency
* @param {boolean} can_manage_reserves
* @param {boolean} can_be_very_patient
* @returns {boolean}
*/
export function can_defeat_draygon(inventory: any, proficiency: number, can_manage_reserves: boolean, can_be_very_patient: boolean): boolean;
/**
* @param {any} inventory
* @param {number} proficiency
* @param {boolean} can_manage_reserves
* @param {boolean} can_be_very_patient
* @returns {boolean}
*/
export function can_defeat_ridley(inventory: any, proficiency: number, can_manage_reserves: boolean, can_be_very_patient: boolean): boolean;
/**
* @param {any} inventory
* @param {number} proficiency
* @param {boolean} second_phase
* @param {boolean} can_manage_reserves
* @returns {boolean}
*/
export function can_defeat_botwoon(inventory: any, proficiency: number, second_phase: boolean, can_manage_reserves: boolean): boolean;
/**
* @param {any} inventory
* @param {number} proficiency
* @param {boolean} supers_double
* @param {boolean} can_manage_reserves
* @param {boolean} can_be_very_patient
* @param {boolean} r_mode
* @returns {boolean}
*/
export function can_defeat_mother_brain_2(inventory: any, proficiency: number, supers_double: boolean, can_manage_reserves: boolean, can_be_very_patient: boolean, r_mode: boolean): boolean;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly can_defeat_phantoon: (a: number, b: number, c: number, d: number) => void;
  readonly can_defeat_draygon: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly can_defeat_ridley: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly can_defeat_botwoon: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly can_defeat_mother_brain_2: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly set_panic_hook: () => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
